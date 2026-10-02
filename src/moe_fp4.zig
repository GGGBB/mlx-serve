//! MoE decode over fp4 expert banks (MXFP4 / NVFP4), read in place: one
//! dispatch for gate+up+SwiGLU across the top-k experts, one for down and the
//! score-weighted sum. Each simdgroup owns ROWS output rows and keeps its 16
//! values of x in registers across them (MLX's `fp_qmv_fast` shape), so x is
//! read once per row group rather than once per row.
const std = @import("std");
const mlx = @import("mlx.zig");
const xfm = @import("transformer.zig");

const ROWS: c_int = 4;
const SGS: c_int = 2;
/// K and the expert width must be whole 512-value blocks (16 values x 32 lanes).
const BLOCK: c_int = 512;

/// e2m1 nibble -> its value times 2^-14, by bit placement into a half.
pub const E2M1_HEADER =
    \\inline float mlxserve_e2m1(uint c) {
    \\  return float(as_type<half>(ushort(((c & 0x7u) << 9) | ((c & 0x8u) << 12))));
    \\}
    \\
;
/// Group scale -> its value times 2^-8: e4m3 (NVFP4) by bit placement into a
/// half, e8m0 (MXFP4) as MLX's `fp8_e8m0` reads it (the byte is a float
/// exponent, 0 is 2^-127). With the e2m1 2^-14, one 2^22 multiply restores both.
const NV_SCALE =
    \\inline float mlxserve_fp4_scale(uint b) {
    \\  return float(as_type<half>(ushort(((b & 0x7Fu) << 7) | ((b & 0x80u) << 8))));
    \\}
    \\
;
const MX_SCALE =
    \\inline float mlxserve_fp4_scale(uint b) {
    \\  return as_type<float>(b == 0u ? 0x400000u : (b << 23)) * 0.00390625f;
    \\}
    \\
;

const DOT16 =
    \\inline float mlxserve_fp4_dot16(uint2 w, thread const float* x) {
    \\  float a = 0.0f, b = 0.0f;
    \\  for (int j = 0; j < 8; j += 2) {
    \\    a += x[j] * mlxserve_e2m1((w.x >> (4 * j)) & 0xFu) + x[j + 1] * mlxserve_e2m1((w.x >> (4 * j + 4)) & 0xFu);
    \\    b += x[8 + j] * mlxserve_e2m1((w.y >> (4 * j)) & 0xFu) + x[9 + j] * mlxserve_e2m1((w.y >> (4 * j + 4)) & 0xFu);
    \\  }
    \\  return a + b;
    \\}
;

// grid (32, N/ROWS, TOPK), threadgroup (32, SGS, 1). The e2m1 and scale decodes
// each scale by a power of two; 2^22 folds both back once per row.
const GATEUP_SOURCE =
    \\const uint lane = thread_index_in_simdgroup;
    \\const int row0 = int(thread_position_in_grid.y) * ROWS;
    \\const uint e = thread_position_in_grid.z;
    \\constexpr int KW2 = K / 16;
    \\constexpr int KG = K / GS;
    \\const size_t row = size_t(inds[e]) * N + row0;
    \\const device uint2* gw = (const device uint2*)wg_q + row * KW2 + lane;
    \\const device uint2* uw = (const device uint2*)wu_q + row * KW2 + lane;
    \\float ag[ROWS] = {0.0f};
    \\float au[ROWS] = {0.0f};
    \\for (int k = 0; k < K; k += 512) {
    \\  const int kk = k + int(lane) * 16;
    \\  float xr[16];
    \\  for (int i = 0; i < 16; ++i) xr[i] = float(x[kk + i]);
    \\  const int gi = kk / GS;
    \\  for (int r = 0; r < ROWS; ++r) {
    \\    const size_t sg = (row + r) * KG + gi;
    \\    ag[r] += mlxserve_fp4_dot16(gw[r * KW2 + k / 16], xr) * mlxserve_fp4_scale(uint(g_scales[sg]));
    \\    au[r] += mlxserve_fp4_dot16(uw[r * KW2 + k / 16], xr) * mlxserve_fp4_scale(uint(u_scales[sg]));
    \\  }
    \\}
    \\for (int r = 0; r < ROWS; ++r) {
    \\  const float g = simd_sum(ag[r]) * 4194304.0f;
    \\  const float u = simd_sum(au[r]) * 4194304.0f;
    \\  if (lane == 0) {
    \\    const T gt = T(g);
    \\    y[size_t(e) * N + row0 + r] = (gt * sigtab[as_type<ushort>(gt)]) * T(u);
    \\  }
    \\}
;

// grid (32, H/ROWS, 1), threadgroup (32, SGS, 1): every expert's down row,
// weighted by its routing score, accumulated in the same registers.
const DOWNRED_SOURCE =
    \\const uint lane = thread_index_in_simdgroup;
    \\const int row0 = int(thread_position_in_grid.y) * ROWS;
    \\constexpr int KW2 = I / 16;
    \\constexpr int KG = I / GS;
    \\float acc[ROWS] = {0.0f};
    \\for (int e = 0; e < TOPK; ++e) {
    \\  const size_t row = size_t(inds[e]) * H + row0;
    \\  const device uint2* dw = (const device uint2*)wd_q + row * KW2 + lane;
    \\  const float sc = float(scores[e]);
    \\  for (int k = 0; k < I; k += 512) {
    \\    const int kk = k + int(lane) * 16;
    \\    float xr[16];
    \\    for (int i = 0; i < 16; ++i) xr[i] = float(act[size_t(e) * I + kk + i]);
    \\    const int gi = kk / GS;
    \\    for (int r = 0; r < ROWS; ++r)
    \\      acc[r] += sc * mlxserve_fp4_dot16(dw[r * KW2 + k / 16], xr) * mlxserve_fp4_scale(uint(d_scales[(row + r) * KG + gi]));
    \\  }
    \\}
    \\for (int r = 0; r < ROWS; ++r) {
    \\  const float v = simd_sum(acc[r]) * 4194304.0f;
    \\  if (lane == 0) y[row0 + r] = T(v);
    \\}
;

const Kernels = struct { gateup: ?mlx.mlx_fast_metal_kernel = null, downred: ?mlx.mlx_fast_metal_kernel = null };
/// Indexed by format: nvfp4, mxfp4.
var kernels: [2]Kernels = .{ .{}, .{} };
var engaged = false;

fn makeKernel(name: [*:0]const u8, ins: []const [*:0]const u8, source: [*:0]const u8, header: [*:0]const u8) !mlx.mlx_fast_metal_kernel {
    const outs = [_][*:0]const u8{"y"};
    const in_vec = mlx.mlx_vector_string_new_data(ins.ptr, ins.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&outs, outs.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const k = mlx.mlx_fast_metal_kernel_new(name, in_vec, out_vec, source, header, true, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    return k;
}

fn apply(kernel: mlx.mlx_fast_metal_kernel, inputs: []const mlx.mlx_array, out_shape: []const c_int, dt: mlx.mlx_dtype, grid: [3]c_int, tmpl: []const struct { [*:0]const u8, c_int }, s: mlx.mlx_stream) !mlx.mlx_array {
    const c = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(c);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, out_shape.ptr, out_shape.len, dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, grid[0], grid[1], grid[2]));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, 32, SGS, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(c, "T", dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "ROWS", ROWS));
    for (tmpl) |t| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, t[0], t[1]));
    const v = mlx.mlx_vector_array_new_data(inputs.ptr, inputs.len);
    defer _ = mlx.mlx_vector_array_free(v);
    var o = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(o);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&o, kernel, v, c, s));
    var y = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_vector_array_get(&y, o, 0));
    return y;
}

pub const Bank = struct { w: mlx.mlx_array, s: mlx.mlx_array };

/// y [hidden] = sum_k scores[k] * down_k(silu(gate_k(x)) * up_k(x)) for one
/// token. `x` [hidden] bf16/f16, banks [E, out, in/8] u32 + [E, out, in/GS] u8,
/// `inds` [TOPK] u32, `scores` [TOPK] in x's dtype, `sigtab` the SwiGLU table.
/// Null outside the kernels' set (caller keeps its path).
pub fn decode(s: mlx.mlx_stream, x: mlx.mlx_array, gate: Bank, up: Bank, down: Bank, inds: mlx.mlx_array, scores: mlx.mlx_array, sigtab: mlx.mlx_array, mode: @import("model.zig").QuantMode, group_size: u32) !?mlx.mlx_array {
    if (!mlx.streamIsGpu(s)) return null;
    const ki: usize = switch (mode) {
        .nvfp4 => if (group_size == 16) 0 else return null,
        .mxfp4 => if (group_size == 32) 1 else return null,
        else => return null,
    };
    const dt = mlx.mlx_array_dtype(x);
    if (dt != .bfloat16 and dt != .float16) return null;
    if (mlx.mlx_array_dtype(inds) != .uint32 or mlx.mlx_array_dtype(scores) != dt) return null;
    const gsh = mlx.getShape(gate.w);
    const dsh = mlx.getShape(down.w);
    if (gsh.len != 3 or dsh.len != 3 or !std.mem.eql(c_int, gsh, mlx.getShape(up.w))) return null;
    const inter = gsh[1];
    const hidden = gsh[2] * 8;
    if (dsh[1] != hidden or dsh[2] * 8 != inter) return null;
    if (@rem(hidden, BLOCK) != 0 or @rem(inter, BLOCK) != 0 or @rem(inter, ROWS * SGS) != 0 or @rem(hidden, ROWS * SGS) != 0) return null;
    if (mlx.mlx_array_size(x) != @as(usize, @intCast(hidden))) return null;
    const topk: c_int = @intCast(mlx.mlx_array_size(inds));

    const full_header = [2][:0]const u8{ E2M1_HEADER ++ NV_SCALE ++ DOT16, E2M1_HEADER ++ MX_SCALE ++ DOT16 };
    if (kernels[ki].gateup == null) {
        const names = [2][*:0]const u8{ "mlxserve_moe_fp4_gateup_nv", "mlxserve_moe_fp4_gateup_mx" };
        const ins = [_][*:0]const u8{ "x", "wg_q", "g_scales", "wu_q", "u_scales", "inds", "sigtab" };
        kernels[ki].gateup = try makeKernel(names[ki], &ins, GATEUP_SOURCE, full_header[ki].ptr);
    }
    if (kernels[ki].downred == null) {
        const names = [2][*:0]const u8{ "mlxserve_moe_fp4_downred_nv", "mlxserve_moe_fp4_downred_mx" };
        const ins = [_][*:0]const u8{ "act", "wd_q", "d_scales", "inds", "scores" };
        kernels[ki].downred = try makeKernel(names[ki], &ins, DOWNRED_SOURCE, full_header[ki].ptr);
    }
    const gs: c_int = @intCast(group_size);
    const act = try apply(kernels[ki].gateup.?, &.{ x, gate.w, gate.s, up.w, up.s, inds, sigtab }, &.{ topk, inter }, dt, .{ 32, @divExact(inter, ROWS), topk }, &.{ .{ "K", hidden }, .{ "N", inter }, .{ "GS", gs } }, s);
    defer _ = mlx.mlx_array_free(act);
    const y = try apply(kernels[ki].downred.?, &.{ act, down.w, down.s, inds, scores }, &.{hidden}, dt, .{ 32, @divExact(hidden, ROWS), 1 }, &.{ .{ "I", inter }, .{ "H", hidden }, .{ "GS", gs }, .{ "TOPK", topk } }, s);
    if (!engaged) {
        engaged = true;
        @import("log.zig").info("[moe] fp4 decode kernels engaged: {s} topk={d} inter={d} hidden={d}\n", .{ @tagName(mode), topk, inter, hidden });
    }
    return y;
}

const testing = std.testing;

fn randBf16(rnd: std.Random, shape: []const c_int, scale: f32, s: mlx.mlx_stream) !mlx.mlx_array {
    var n: usize = 1;
    for (shape) |d| n *= @intCast(d);
    const buf = try testing.allocator.alloc(f32, n);
    defer testing.allocator.free(buf);
    for (buf) |*v| v.* = (rnd.float(f32) - 0.5) * scale;
    const f = mlx.mlx_array_new_data(buf.ptr, shape.ptr, @intCast(shape.len), .float32);
    defer _ = mlx.mlx_array_free(f);
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&out, f, .bfloat16, s));
    return out;
}

fn quant(w: mlx.mlx_array, gs: c_int, mode: [:0]const u8, s: mlx.mlx_stream) !struct { b: Bank, deq: mlx.mlx_array } {
    var pair = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(pair);
    try mlx.check(mlx.mlx_quantize(&pair, w, mlx.mlx_optional_int.some(gs), mlx.mlx_optional_int.some(4), mode, .{}, s));
    var b: Bank = .{ .w = mlx.mlx_array_new(), .s = mlx.mlx_array_new() };
    try mlx.check(mlx.mlx_vector_array_get(&b.w, pair, 0));
    try mlx.check(mlx.mlx_vector_array_get(&b.s, pair, 1));
    var deq = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_dequantize(&deq, b.w, b.s, .{ .ctx = null }, mlx.mlx_optional_int.some(gs), mlx.mlx_optional_int.some(4), mode, .{ .ctx = null }, .{ .value = .float32, .has_value = true }, s));
    return .{ .b = b, .deq = deq };
}

test "moe fp4 decode: gate+up+SwiGLU and the weighted down match fp32 truth (mxfp4, nvfp4)" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(0x4D1E0F4);
    const rnd = prng.random();
    const E: c_int = 16;
    const I: c_int = 1024;
    const H: c_int = 1536;
    const TOPK: c_int = 8;
    const Fmt = struct { name: [:0]const u8, gs: c_int, mode: @import("model.zig").QuantMode };
    for ([_]Fmt{ .{ .name = "mxfp4", .gs = 32, .mode = .mxfp4 }, .{ .name = "nvfp4", .gs = 16, .mode = .nvfp4 } }) |f| {
        var q: [3]@TypeOf(try quant(.{ .ctx = null }, 0, "", s)) = undefined;
        const shapes = [3][3]c_int{ .{ E, I, H }, .{ E, I, H }, .{ E, H, I } };
        for (&q, shapes) |*slot, sh| {
            const w = try randBf16(rnd, &sh, 0.2, s);
            defer _ = mlx.mlx_array_free(w);
            slot.* = try quant(w, f.gs, f.name, s);
        }
        defer for (q) |e| {
            _ = mlx.mlx_array_free(e.b.w);
            _ = mlx.mlx_array_free(e.b.s);
            _ = mlx.mlx_array_free(e.deq);
        };
        const x = try randBf16(rnd, &.{H}, 2.0, s);
        defer _ = mlx.mlx_array_free(x);
        const idx = [8]u32{ 3, 0, 15, 7, 9, 1, 12, 5 };
        const inds = mlx.mlx_array_new_data(&idx, &[_]c_int{TOPK}, 1, .uint32);
        defer _ = mlx.mlx_array_free(inds);
        const scores = try randBf16(rnd, &.{TOPK}, 0.5, s);
        defer _ = mlx.mlx_array_free(scores);
        const sigtab = try xfm.swigluSigTable(s, .bfloat16, std.heap.c_allocator);

        const ours = (try decode(s, x, q[0].b, q[1].b, q[2].b, inds, scores, sigtab, f.mode, @intCast(f.gs))) orelse return error.KernelDeclined;
        defer _ = mlx.mlx_array_free(ours);

        // f32 truth over the dequantized banks: y = sum_k s_k * Wd_k (silu(Wg_k x) * Wu_k x).
        var tr = std.ArrayList(mlx.mlx_array).empty;
        defer {
            for (tr.items) |a| _ = mlx.mlx_array_free(a);
            tr.deinit(testing.allocator);
        }
        const T = struct {
            fn op(list: *std.ArrayList(mlx.mlx_array), a: mlx.mlx_array) !mlx.mlx_array {
                try list.append(testing.allocator, a);
                return a;
            }
        };
        var x32 = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_astype(&x32, x, .float32, s));
        _ = try T.op(&tr, x32);
        var xc = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_reshape(&xc, x32, &.{ 1, H, 1 }, 3, s));
        _ = try T.op(&tr, xc);
        var mats: [3]mlx.mlx_array = undefined;
        for (&mats, q) |*m, e| {
            m.* = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_take_axis(m, e.deq, inds, 0, s));
            _ = try T.op(&tr, m.*);
        }
        var g = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_matmul(&g, mats[0], xc, s));
        _ = try T.op(&tr, g);
        var u = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_matmul(&u, mats[1], xc, s));
        _ = try T.op(&tr, u);
        var sg = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_sigmoid(&sg, g, s));
        _ = try T.op(&tr, sg);
        var a = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_multiply(&a, g, sg, s));
        _ = try T.op(&tr, a);
        var a2 = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_multiply(&a2, a, u, s));
        _ = try T.op(&tr, a2);
        var d = mlx.mlx_array_new(); // [K, H, 1]
        try mlx.check(mlx.mlx_matmul(&d, mats[2], a2, s));
        _ = try T.op(&tr, d);
        var s32 = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_astype(&s32, scores, .float32, s));
        _ = try T.op(&tr, s32);
        var s3 = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_reshape(&s3, s32, &.{ TOPK, 1, 1 }, 3, s));
        _ = try T.op(&tr, s3);
        var wd = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_multiply(&wd, d, s3, s));
        _ = try T.op(&tr, wd);
        var truth = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_sum_axis(&truth, wd, 0, false, s));
        _ = try T.op(&tr, truth);
        var o32 = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_astype(&o32, ours, .float32, s));
        _ = try T.op(&tr, o32);
        try mlx.check(mlx.mlx_array_eval(o32));
        try mlx.check(mlx.mlx_array_eval(truth));
        const n: usize = @intCast(H);
        const po = mlx.mlx_array_data_float32(o32).?;
        const pt = mlx.mlx_array_data_float32(truth).?;
        var se: f64 = 0;
        var st: f64 = 0;
        for (0..n) |i| {
            se += (po[i] - pt[i]) * (po[i] - pt[i]);
            st += pt[i] * pt[i];
        }
        const rel = @sqrt(se / st);
        std.debug.print("[moe-fp4] {s}: rel rms err vs fp32 truth {d:.5}\n", .{ f.name, rel });
        try testing.expect(rel < 0.02);
    }
}
