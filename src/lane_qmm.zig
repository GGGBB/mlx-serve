//! Row-exact 4-bit matmul on the M5 tensor units for 1..MAX_ROWS rows, ported
//! from TensorFold's `lane_qmm.py` (MIT, see NOTICE). Every 64- (or 32-) input
//! group runs one fixed 16-row `matmul2d` over the packed 4-bit codes, then
//! `C = s * P + b * XS` in group order in fp32, the K slices summed in slice
//! order; the slice count follows the weight's shape only. A row's bits never
//! depend on how many rows ride with it, so a drafted window verifies with the
//! one-row step's bits. Weights whose N is a multiple of 64 take the 64-column
//! tile two simdgroups run together (`COOP`), the rest 32 columns (`NARROW`),
//! both over MLX's packed layout or a tiled copy (TILED: each column tile's
//! group one contiguous block); all four give the same bits.
const std = @import("std");
const mlx = @import("mlx.zig");

pub const MAX_ROWS = 16;

const HEADER =
    \\#include <metal_tensor>
    \\#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
    \\using namespace mpp::tensor_ops;
    \\
;

// XS (a row's sum over a group) is summed in the kernel in a fixed order per
// row: lane 2r + h adds half h of row r, and the two halves join low + high.
const ROW_SUMS =
    \\  const int xr = lane >> 1, xh = lane & 1;
    \\  const bool xlive = xr < M;
    \\  const device bfloat* xp = X + size_t(xlive ? xr : 0) * K + xh * (GS / 2);
    \\
;
const ROW_SUM_G =
    \\    float half_sum = 0.0f;
    \\    if (xlive) for (int i = 0; i < GS / 2; i++) half_sum += float(xp[g * GS + i]);
    \\    const float other = simd_shuffle_xor(half_sum, ushort(1));
    \\    const float row_sum = xh ? other + half_sum : half_sum + other;
    \\
;

// 32 output columns a simdgroup; SBt holds (s, b) bf16 pairs group-major [K/GS][N][2].
const NARROW = 
    \\  const ushort lane = thread_index_in_simdgroup;
    \\  const ushort sg = simdgroup_index_in_threadgroup;     // K slice
    \\  const short qid = lane >> 2;
    \\  const short fm = (qid & 4) | ((lane >> 1) & 3);       // fragment row of this lane (and fm + 8)
    \\  const short fn = ((qid & 2) | (lane & 1)) * 4;        // first of its four fragment columns
    \\  const int M = mdims[0];
    \\  constexpr int KG = K / GS;
    \\  constexpr int NF = 2;
    \\  const int n0 = threadgroup_position_in_grid.x * 32;
    \\  const int g_begin = (sg * KG) / SK;
    \\  const int g_end = ((sg + 1) * KG) / SK;
    \\  constexpr auto desc = matmul2d_descriptor(16, 32, GS, false, true, false, matmul2d_descriptor::mode::multiply);
    \\  matmul2d<desc, execution_simdgroup> op;
    \\  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tA((device bfloat*)X, dextents<int32_t, 2>(K, M));
    \\  tensor<device uint4b_format, dextents<int32_t, 2>, tensor_inline> tB((device uchar*)W, dextents<int32_t, 2>(K, N));
    \\  float C[NF * 8];
    \\  for (int i = 0; i < NF * 8; i++) C[i] = 0.0f;
    \\  const device uint4* sbv = (const device uint4*)SBt;
    \\  bool colok[NF];
    \\  for (int f = 0; f < NF; f++) colok[f] = n0 + f * 16 + fn < N;
++ "\n" ++ ROW_SUMS ++
    \\  for (int g = g_begin; g < g_end; g++) {
++ "\n" ++ ROW_SUM_G ++
    \\    const float xs0 = simd_shuffle(row_sum, ushort(2 * fm));
    \\    const float xs1 = simd_shuffle(row_sum, ushort(2 * (fm + 8)));
    \\    float s[NF][4], bb[NF][4];
    \\    for (int f = 0; f < NF; f++) {
    \\      const uint4 q = colok[f] ? sbv[(size_t(g) * N + n0 + f * 16 + fn) / 4] : uint4(0);
    \\      const vec<bfloat, 8> v = as_type<vec<bfloat, 8>>(q);
    \\      for (int j = 0; j < 4; j++) { s[f][j] = float(v[2 * j]); bb[f][j] = float(v[2 * j + 1]); }
    \\    }
    \\    auto a = tA.slice(g * GS, 0);
    \\#if TILED
    \\    tensor<device uint4b_format, dextents<int32_t, 2>, tensor_inline> b(
    \\        (device uchar*)W + (int64_t)(threadgroup_position_in_grid.x * KG + g) * (32 * GS / 2), dextents<int32_t, 2>(GS, 32));
    \\#else
    \\    auto b = tB.slice(g * GS, n0);
    \\#endif
    \\    auto P = op.template get_destination_cooperative_tensor<decltype(a), decltype(b), float>();
    \\    op.run(a, b, P);
    \\    for (int f = 0; f < NF; f++)
    \\      for (int r = 0; r < 2; r++)
    \\        for (int j = 0; j < 4; j++) {
    \\          const int i = f * 8 + r * 4 + j;
    \\          C[i] = fma(s[f][j], P[i], fma(bb[f][j], r ? xs1 : xs0, C[i]));
    \\        }
    \\  }
    \\  // K slices are added in slice order
    \\  threadgroup float part[(SK > 1 ? SK - 1 : 1) * NF * 8 * 32];
    \\  if (SK > 1) {
    \\    if (sg > 0) for (int i = 0; i < NF * 8; i++) part[((sg - 1) * NF * 8 + i) * 32 + lane] = C[i];
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    if (sg == 0)
    \\      for (int s2 = 1; s2 < SK; s2++) for (int i = 0; i < NF * 8; i++) C[i] += part[((s2 - 1) * NF * 8 + i) * 32 + lane];
    \\  }
    \\  if (sg == 0)
    \\    for (int f = 0; f < NF; f++)
    \\      for (int r = 0; r < 2; r++) {
    \\        const int m = fm + 8 * r;
    \\        const int n = n0 + f * 16 + fn;
    \\        if (m < M && n < N)
    \\          for (int j = 0; j < 4; j++) Y[m * N + n + j] = static_cast<bfloat>(C[f * 8 + r * 4 + j]);
    \\      }
    \\
;

// 64 output columns run by two simdgroups together (N % 64 == 0); a pair per K slice.
const COOP =
    \\  const ushort sg = simdgroup_index_in_threadgroup;
    \\  const ushort lane = thread_index_in_simdgroup;
    \\  const ushort slice = sg >> 1;
    \\  const int M = mdims[0];
    \\  constexpr int KG = K / GS;
    \\  const int n0 = threadgroup_position_in_grid.x * 64;
    \\  const int g_begin = (slice * KG) / SK;
    \\  const int g_end = ((slice + 1) * KG) / SK;
    \\  constexpr auto desc = matmul2d_descriptor(16, 64, GS, false, true, false, matmul2d_descriptor::mode::multiply);
    \\  matmul2d<desc, execution_simdgroups<2>> op;
    \\  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tA((device bfloat*)X, dextents<int32_t, 2>(K, M));
    \\  tensor<device uint4b_format, dextents<int32_t, 2>, tensor_inline> tB((device uchar*)W, dextents<int32_t, 2>(K, N));
    \\  auto a0 = tA.slice(0, 0);
    \\  auto b0 = tB.slice(0, 0);
    \\  auto P = op.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();
    \\  constexpr int CAP = 16;                                   // 16 x 64 outputs over 64 threads
    \\  short ecol[CAP], erow[CAP];
    \\  for (int i = 0; i < CAP; i++) { auto ids = P.get_multidimensional_index(i); ecol[i] = ids[0]; erow[i] = ids[1]; }
    \\  float C[CAP];
    \\  for (int i = 0; i < CAP; i++) C[i] = 0.0f;
    \\  const device uint* sbw = (const device uint*)SBt;
++ "\n" ++ ROW_SUMS ++
    \\  for (int g = g_begin; g < g_end; g++) {
++ "\n" ++ ROW_SUM_G ++
    \\    auto a = tA.slice(g * GS, 0);
    \\#if TILED
    \\    tensor<device uint4b_format, dextents<int32_t, 2>, tensor_inline> b(
    \\        (device uchar*)W + (int64_t)(threadgroup_position_in_grid.x * KG + g) * (64 * GS / 2), dextents<int32_t, 2>(GS, 64));
    \\#else
    \\    auto b = tB.slice(g * GS, n0);
    \\#endif
    \\    op.run(a, b, P);
    \\    for (int i = 0; i < CAP; i++) {
    \\      const vec<bfloat, 2> sb = as_type<vec<bfloat, 2>>(sbw[size_t(g) * N + n0 + ecol[i]]);
    \\      const float xs = simd_shuffle(row_sum, ushort(2 * erow[i]));
    \\      C[i] = fma(float(sb[0]), P[i], fma(float(sb[1]), xs, C[i]));
    \\    }
    \\  }
    \\  // K slices are added in slice order
    \\  threadgroup float part[(SK > 1 ? SK - 1 : 1) * 16 * 64];
    \\  const ushort tip = ushort(thread_position_in_threadgroup.x) - slice * 64;
    \\  if (SK > 1) {
    \\    if (slice > 0) for (int i = 0; i < CAP; i++) part[((slice - 1) * CAP + i) * 64 + tip] = C[i];
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    if (slice == 0)
    \\      for (int s2 = 1; s2 < SK; s2++) for (int i = 0; i < CAP; i++) C[i] += part[((s2 - 1) * CAP + i) * 64 + tip];
    \\  }
    \\  if (slice == 0)
    \\    for (int i = 0; i < CAP; i++) {
    \\      const int m = erow[i], n = n0 + ecol[i];
    \\      if (m < M) Y[m * N + n] = static_cast<bfloat>(C[i]);
    \\    }
    \\
;

/// K slices for an (n, k) weight: fixed by the shape, never by the row count.
fn splitK(n: c_int, k: c_int) c_int {
    const tiles = @divTrunc(n + 31, 32);
    var sk: c_int = 1;
    while (sk < 8 and tiles * sk < 1024 and @divTrunc(@divTrunc(k, 64), sk * 2) >= 8) sk *= 2;
    return sk;
}

/// The pick follows the weight's shape only: 64-column tiles where N allows.
fn coopFor(n: c_int) bool {
    return @rem(n, 64) == 0;
}

/// Bytes the tiled weight copies may still take: a machine decision the scheduler
/// makes when the drafter binds. A weight past it is read in MLX's layout (same bits).
pub var tile_budget: u64 = 0;

const KernelKey = struct { coop: bool, tiled: bool, k: c_int, n: c_int, gs: c_int, sk: c_int };
var kernels: std.AutoHashMapUnmanaged(KernelKey, mlx.mlx_fast_metal_kernel) = .{};
const PlanKey = struct { coop: bool, rows: c_int, n: c_int, k: c_int };
var plans: std.AutoHashMapUnmanaged(PlanKey, mlx.mlx_fast_metal_kernel_config) = .{};
var mdims_cache: [MAX_ROWS + 1]mlx.mlx_array = @splat(.{ .ctx = null });

/// Arrays derived from a weight and built on first use: (s, b) pairs
/// group-major per (scales, biases), a tiled copy per packed weight. Keyed by
/// the data the handles point at (a handle's address is reused once its caller
/// frees it) plus the shape (a joined weight and its first part start at the
/// same address); an entry holds its sources alive, so no key is reused.
const DerivedKey = struct { a: usize, b: usize, n: c_int, w: c_int };
const Derived = struct { src_a: mlx.mlx_array, src_b: mlx.mlx_array, out: mlx.mlx_array };
var packed_scales: std.AutoHashMapUnmanaged(DerivedKey, Derived) = .{};
var tiled_weights: std.AutoHashMapUnmanaged(DerivedKey, Derived) = .{};

/// Frees every derived copy (a model unload); the next call rebuilds its own.
pub fn release() void {
    for ([_]*std.AutoHashMapUnmanaged(DerivedKey, Derived){ &packed_scales, &tiled_weights }) |map| {
        var it = map.valueIterator();
        while (it.next()) |e| {
            _ = mlx.mlx_array_free(e.src_a);
            if (e.src_b.ctx != null) _ = mlx.mlx_array_free(e.src_b);
            _ = mlx.mlx_array_free(e.out);
        }
        map.clearAndFree(std.heap.c_allocator);
    }
    tile_budget = 0;
}

fn dataKey(a: mlx.mlx_array, b: ?mlx.mlx_array) !DerivedKey {
    // Loaded weights are evaluated already (a no-op); a data read needs it.
    try mlx.check(mlx.mlx_array_eval(a));
    const shape = mlx.getShape(a);
    const ap: usize = switch (mlx.mlx_array_dtype(a)) {
        .uint32 => @intFromPtr(mlx.mlx_array_data_uint32(a) orelse return error.UnreadableWeight),
        else => @intFromPtr(mlx.mlx_array_data_bfloat16(a) orelse return error.UnreadableWeight),
    };
    var bp: usize = 0;
    if (b) |bb| {
        try mlx.check(mlx.mlx_array_eval(bb));
        bp = @intFromPtr(mlx.mlx_array_data_bfloat16(bb) orelse return error.UnreadableWeight);
    }
    return .{ .a = ap, .b = bp, .n = shape[0], .w = shape[1] };
}

fn remember(map: *std.AutoHashMapUnmanaged(DerivedKey, Derived), key: DerivedKey, a: mlx.mlx_array, b: ?mlx.mlx_array, out: mlx.mlx_array) !void {
    var held_a = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(held_a);
    try mlx.check(mlx.mlx_array_set(&held_a, a));
    var held_b: mlx.mlx_array = .{ .ctx = null };
    if (b) |bb| {
        held_b = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_array_set(&held_b, bb));
    }
    try map.put(std.heap.c_allocator, key, .{ .src_a = held_a, .src_b = held_b, .out = out });
}

fn packedFor(sc: mlx.mlx_array, bi: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    const key = try dataKey(sc, bi);
    if (packed_scales.get(key)) |e| return e.out;
    var st = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(st);
    try mlx.check(mlx.mlx_transpose(&st, sc, s));
    var bt = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(bt);
    try mlx.check(mlx.mlx_transpose(&bt, bi, s));
    const pair = [_]mlx.mlx_array{ st, bt };
    const vec = mlx.mlx_vector_array_new_data(&pair, 2);
    defer _ = mlx.mlx_vector_array_free(vec);
    var stacked = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(stacked);
    try mlx.check(mlx.mlx_stack_axis(&stacked, vec, -1, s));
    var sbt = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(sbt);
    try mlx.check(mlx.mlx_contiguous(&sbt, stacked, false, s));
    try remember(&packed_scales, key, sc, bi, sbt);
    return sbt;
}

/// MLX's packed [N, K / 8] regrouped [N / nt][K / gs][nt columns x a group's words].
fn tiledFor(w: mlx.mlx_array, nt: c_int, gs: c_int, s: mlx.mlx_stream) !?mlx.mlx_array {
    const key = try dataKey(w, null);
    if (tiled_weights.get(key)) |e| return e.out;
    const n = key.n;
    const kw = key.w;
    const bytes: u64 = @as(u64, @intCast(n)) * @as(u64, @intCast(kw)) * 4;
    if (bytes > tile_budget) return null;
    const wg = @divExact(gs * 4, 32);
    var r4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(r4);
    try mlx.check(mlx.mlx_reshape(&r4, w, &[_]c_int{ @divExact(n, nt), nt, @divExact(kw, wg), wg }, 4, s));
    var tr = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(tr);
    try mlx.check(mlx.mlx_transpose_axes(&tr, r4, &[_]c_int{ 0, 2, 1, 3 }, 4, s));
    var c = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(c);
    try mlx.check(mlx.mlx_contiguous(&c, tr, false, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_reshape(&out, c, &[_]c_int{ n, kw }, 2, s));
    try remember(&tiled_weights, key, w, null, out);
    tile_budget -= bytes;
    return out;
}

fn kernelFor(key: KernelKey) !mlx.mlx_fast_metal_kernel {
    if (kernels.get(key)) |k| return k;
    const a = std.heap.c_allocator;
    const consts = try std.fmt.allocPrint(a, "#define TILED {d}\n  constexpr int K = {d};\n  constexpr int N = {d};\n  constexpr int GS = {d};\n  constexpr int SK = {d};\n", .{ @intFromBool(key.tiled), key.k, key.n, key.gs, key.sk });
    defer a.free(consts);
    const source = try std.mem.concatWithSentinel(a, u8, &.{ consts, if (key.coop) COOP else NARROW }, 0);
    defer a.free(source);
    const name = try std.fmt.allocPrintSentinel(a, "msv_lane_qmm_{s}{s}_k{d}_n{d}_g{d}_s{d}", .{ if (key.coop) "coop" else "narrow", if (key.tiled) "_tiled" else "", key.k, key.n, key.gs, key.sk }, 0);
    defer a.free(name);
    const in_names = [_][*:0]const u8{ "X", "W", "SBt", "mdims" };
    const out_names = [_][*:0]const u8{"Y"};
    const in_vec = mlx.mlx_vector_string_new_data(&in_names, in_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&out_names, out_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const k = mlx.mlx_fast_metal_kernel_new(name.ptr, in_vec, out_vec, source.ptr, HEADER, true, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    try kernels.put(a, key, k);
    return k;
}

fn planFor(key: PlanKey) !mlx.mlx_fast_metal_kernel_config {
    if (plans.get(key)) |p| return p;
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    const sk = splitK(key.n, key.k);
    const width: c_int = if (key.coop) 64 else 32;
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ key.rows, key.n }, 2, .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, @divTrunc(key.n + width - 1, width) * width * sk, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, width * sk, 1, 1));
    try plans.put(std.heap.c_allocator, key, cfg);
    return cfg;
}

/// A 4-bit affine bf16 matrix [N, K / 8] in groups of 64 or 32, K % 64 == 0, N % 4 == 0.
pub fn fits(w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, bits: u32, group_size: u32) bool {
    if (bits != 4 or (group_size != 64 and group_size != 32) or bi.ctx == null) return false;
    if (mlx.mlx_array_dtype(w) != .uint32 or mlx.mlx_array_dtype(sc) != .bfloat16 or mlx.mlx_array_dtype(bi) != .bfloat16) return false;
    const ws = mlx.getShape(w);
    return ws.len == 2 and @rem(ws[1] * 8, 64) == 0 and @rem(ws[0], 4) == 0;
}

/// `x [..., K] @ w.T` for 1..MAX_ROWS rows on the tensor units, or null
/// outside the kernel (the caller's other row-exact kernels take it).
pub fn qmm(x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, bits: u32, group_size: u32, s: mlx.mlx_stream) !?mlx.mlx_array {
    if (!mlx.streamIsGpu(s) or !fits(w, sc, bi, bits, group_size) or mlx.mlx_array_dtype(x) != .bfloat16) return null;
    const xs = mlx.getShape(x);
    if (xs.len == 0 or xs.len > 8) return null;
    const ws = mlx.getShape(w);
    const k = xs[xs.len - 1];
    if (k * 4 != ws[1] * 32) return null;
    var rows: c_int = 1;
    for (xs[0 .. xs.len - 1]) |d| rows *= d;
    if (rows < 1 or rows > MAX_ROWS) return null;
    const n = ws[0];
    const gs: c_int = @intCast(group_size);
    const coop = coopFor(n);
    const width: c_int = if (coop) 64 else 32;
    const tiled_w: ?mlx.mlx_array = if (@rem(n, width) == 0) try tiledFor(w, width, gs, s) else null;
    const tiled = tiled_w != null;

    var x2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x2);
    try mlx.check(mlx.mlx_reshape(&x2, x, &[_]c_int{ rows, k }, 2, s));
    var xc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(xc);
    try mlx.check(mlx.mlx_contiguous(&xc, x2, false, s));
    const ri: usize = @intCast(rows);
    if (mdims_cache[ri].ctx == null) {
        const d = [_]i32{rows};
        mdims_cache[ri] = mlx.mlx_array_new_data(&d, &[_]c_int{1}, 1, .int32);
    }
    const sbt = try packedFor(sc, bi, s);
    const wk = tiled_w orelse w;
    const mk = try kernelFor(.{ .coop = coop, .tiled = tiled, .k = k, .n = n, .gs = gs, .sk = splitK(n, k) });
    const mcfg = try planFor(.{ .coop = coop, .rows = rows, .n = n, .k = k });
    const ins = [_]mlx.mlx_array{ xc, wk, sbt, mdims_cache[ri] };
    const in_vec = mlx.mlx_vector_array_new_data(&ins, ins.len);
    defer _ = mlx.mlx_vector_array_free(in_vec);
    var outs = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outs);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outs, mk, in_vec, mcfg, s));
    var y2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(y2);
    try mlx.check(mlx.mlx_vector_array_get(&y2, outs, 0));
    var shape: [8]c_int = undefined;
    @memcpy(shape[0 .. xs.len - 1], xs[0 .. xs.len - 1]);
    shape[xs.len - 1] = n;
    var y = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&y, y2, &shape, xs.len, s));
    return y;
}

const testing = std.testing;

fn randBf16(shape: []const c_int, scale: f32, seed: u64, s: mlx.mlx_stream) !mlx.mlx_array {
    var key = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(key);
    try mlx.check(mlx.mlx_random_key(&key, seed));
    var f = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f);
    try mlx.check(mlx.mlx_random_normal(&f, shape.ptr, shape.len, .float32, 0.0, scale, key, s));
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&out, f, .bfloat16, s));
    return out;
}

test "lane_qmm: every row of an R-row call equals its one-row call bit for bit, tiled or not, and the product is the fp32 one" {
    if (!@import("transformer.zig").naxAvailable()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    errdefer {
        var buf: [512]u8 = undefined;
        if (mlx.takeError(&buf)) |msg| std.debug.print("[lane_qmm] mlx: {s}\n", .{msg});
    }
    defer release();
    // 64-column tiles (MLP, down projection), 32-column ones (a joined GDN
    // projection: N % 64 == 32), and a ragged untiled one (GDN a/b: 48 columns).
    for ([_][2]c_int{ .{ 1024, 5120 }, .{ 5120, 1024 }, .{ 16480, 1024 }, .{ 48, 5120 } }, 0..) |sh, si| for ([_]u32{ 64, 32 }) |gs| {
        const wf = try randBf16(&.{ sh[0], sh[1] }, 0.02, 100 + si, s);
        defer _ = mlx.mlx_array_free(wf);
        var triple = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(triple);
        try mlx.check(mlx.mlx_quantize(&triple, wf, mlx.mlx_optional_int.some(@intCast(gs)), mlx.mlx_optional_int.some(4), "affine", .{}, s));
        var w = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(w);
        var sc = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sc);
        var bi = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(bi);
        try mlx.check(mlx.mlx_vector_array_get(&w, triple, 0));
        try mlx.check(mlx.mlx_vector_array_get(&sc, triple, 1));
        try mlx.check(mlx.mlx_vector_array_get(&bi, triple, 2));
        const x = try randBf16(&.{ MAX_ROWS, sh[1] }, 1.0, 7 + si, s);
        defer _ = mlx.mlx_array_free(x);
        // fp32 truth: the dequantized weight times x in fp32.
        var wd = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(wd);
        try mlx.check(mlx.mlx_dequantize(&wd, w, sc, bi, mlx.mlx_optional_int.some(@intCast(gs)), mlx.mlx_optional_int.some(4), "affine", .{ .ctx = null }, .{ .value = .float32, .has_value = true }, s));
        var xf = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(xf);
        try mlx.check(mlx.mlx_astype(&xf, x, .float32, s));
        var wt = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(wt);
        try mlx.check(mlx.mlx_transpose(&wt, wd, s));
        var truth = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(truth);
        try mlx.check(mlx.mlx_matmul(&truth, xf, wt, s));
        tile_budget = 0;
        const plain = (try qmm(x, w, sc, bi, 4, gs, s)).?;
        defer _ = mlx.mlx_array_free(plain);
        tile_budget = std.math.maxInt(u64);
        const all = (try qmm(x, w, sc, bi, 4, gs, s)).?;
        defer _ = mlx.mlx_array_free(all);
        try testing.expect(try bitEqual(all, plain, s));
        // Parity: relative RMS error vs fp32 truth at bf16 output rounding.
        {
            var af = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(af);
            try mlx.check(mlx.mlx_astype(&af, all, .float32, s));
            var d = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(d);
            try mlx.check(mlx.mlx_subtract(&d, af, truth, s));
            const err = try meanSquare(d, s);
            const ref = try meanSquare(truth, s);
            try testing.expect(@sqrt(err / ref) < 1e-2);
        }
        var r: c_int = 1;
        while (r <= MAX_ROWS) : (r += 1) {
            var xr = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(xr);
            try mlx.check(mlx.mlx_slice(&xr, x, &[_]c_int{ 0, 0 }, 2, &[_]c_int{ r, sh[1] }, 2, &[_]c_int{ 1, 1 }, 2, s));
            const part = (try qmm(xr, w, sc, bi, 4, gs, s)).?;
            defer _ = mlx.mlx_array_free(part);
            // Row r-1 of the r-row call == row r-1 of the one-row call == row r-1 of the full call.
            var last = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(last);
            try mlx.check(mlx.mlx_slice(&last, xr, &[_]c_int{ r - 1, 0 }, 2, &[_]c_int{ r, sh[1] }, 2, &[_]c_int{ 1, 1 }, 2, s));
            const one = (try qmm(last, w, sc, bi, 4, gs, s)).?;
            defer _ = mlx.mlx_array_free(one);
            var pr = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(pr);
            try mlx.check(mlx.mlx_slice(&pr, part, &[_]c_int{ r - 1, 0 }, 2, &[_]c_int{ r, sh[0] }, 2, &[_]c_int{ 1, 1 }, 2, s));
            var fr = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(fr);
            try mlx.check(mlx.mlx_slice(&fr, all, &[_]c_int{ r - 1, 0 }, 2, &[_]c_int{ r, sh[0] }, 2, &[_]c_int{ 1, 1 }, 2, s));
            try testing.expect(try bitEqual(pr, one, s));
            try testing.expect(try bitEqual(fr, one, s));
        }
    };
}

test "lane_qmm: the tile budget bounds the tiled copies, and a weight past it keeps its bits" {
    if (!@import("transformer.zig").naxAvailable()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    defer release();
    var ws: [2][3]mlx.mlx_array = undefined;
    for (&ws, 0..) |*t, i| {
        const wf = try randBf16(&.{ 1024, 1024 }, 0.02, 300 + i, s);
        defer _ = mlx.mlx_array_free(wf);
        var triple = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(triple);
        try mlx.check(mlx.mlx_quantize(&triple, wf, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(4), "affine", .{}, s));
        for (t, 0..) |*a, j| {
            a.* = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_vector_array_get(a, triple, j));
        }
    }
    defer for (ws) |t| for (t) |a| {
        _ = mlx.mlx_array_free(a);
    };
    const x = try randBf16(&.{ 3, 1024 }, 1.0, 9, s);
    defer _ = mlx.mlx_array_free(x);
    const one_copy: u64 = 1024 * 1024 / 2;
    tile_budget = one_copy + one_copy / 2;
    var ys: [2]mlx.mlx_array = undefined;
    for (ws, 0..) |t, i| ys[i] = (try qmm(x, t[0], t[1], t[2], 4, 64, s)).?;
    defer for (ys) |y| {
        _ = mlx.mlx_array_free(y);
    };
    try testing.expectEqual(@as(u32, 1), tiled_weights.count());
    try testing.expectEqual(one_copy / 2, tile_budget);
    release();
    for (ws, 0..) |t, i| {
        const plain = (try qmm(x, t[0], t[1], t[2], 4, 64, s)).?;
        defer _ = mlx.mlx_array_free(plain);
        try testing.expect(try bitEqual(ys[i], plain, s));
    }
    try testing.expectEqual(@as(u32, 0), tiled_weights.count());
}

fn meanSquare(a: mlx.mlx_array, s: mlx.mlx_stream) !f32 {
    var sq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sq);
    try mlx.check(mlx.mlx_square(&sq, a, s));
    var m = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(m);
    try mlx.check(mlx.mlx_mean(&m, sq, false, s));
    try mlx.check(mlx.mlx_array_eval(m));
    var v: f32 = 0;
    try mlx.check(mlx.mlx_array_item_float32(&v, m));
    return v;
}

fn bitEqual(a: mlx.mlx_array, b: mlx.mlx_array, s: mlx.mlx_stream) !bool {
    var e = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(e);
    try mlx.check(mlx.mlx_array_equal(&e, a, b, false, s));
    try mlx.check(mlx.mlx_array_eval(e));
    var v: bool = false;
    try mlx.check(mlx.mlx_array_item_bool(&v, e));
    return v;
}
