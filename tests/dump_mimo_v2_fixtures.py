#!/usr/bin/env python3
"""Engine oracle for mimo_v2 on a TINY random model.

Writes a random MiMo checkpoint in the RELEASE's storage format (FP8 e4m3 fused
QKV in tensor-parallel slabs with rank-local 128x128 tiles, FP8 dense MLP,
per-expert MXFP4, unfolded attention_value_scale), runs the checkpoint's own
`modeling_mimo_v2.py` in f32 on the decoded weights, and dumps the logits.
`tests/convert_mimo_v2.py` then packs the tiny release and the Zig test
`mimo_v2 fixture` compares our forward against these logits.

  venv/bin/python tests/dump_mimo_v2_fixtures.py --ref <dir holding modeling_mimo_v2.py> --out ~/claude-tmp/mimo-tiny
  venv/bin/python tests/convert_mimo_v2.py --src ~/claude-tmp/mimo-tiny/src --dst ~/claude-tmp/mimo-tiny/pack --bits 16
  MIMO_V2_MODEL=~/claude-tmp/mimo-tiny/pack MIMO_V2_FIXTURE=~/claude-tmp/mimo-tiny/fixture.safetensors \\
      zig build test -Dtest-filter="mimo_v2 fixture"

Every expert runs (top-k = all), so a near-tie in expert selection cannot fail
the comparison; selection itself is exercised by the live model.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

BLOCK = 128
TP = 2
E2M1 = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float32)

CONFIG = {
    "architectures": ["MiMoV2ForCausalLM"], "model_type": "mimo_v2",
    "add_full_attention_sink_bias": False, "add_swa_attention_sink_bias": True,
    "attention_bias": False, "attention_chunk_size": 128, "attention_dropout": 0.0,
    "attention_projection_layout": "fused_qkv", "attention_value_scale": 0.707,
    "bos_token_id": None, "eos_token_id": 1, "pad_token_id": 0,
    "head_dim": 32, "v_head_dim": 16, "swa_head_dim": 32, "swa_v_head_dim": 16,
    "hidden_act": "silu", "hidden_size": 128, "intermediate_size": 256,
    "hybrid_layer_pattern": [0, 1, 1, 0], "moe_layer_freq": [0, 1, 1, 1],
    "layernorm_epsilon": 1e-6, "max_position_embeddings": 4096,
    "moe_intermediate_size": 64, "n_group": 1, "n_routed_experts": 8, "n_shared_experts": None,
    "norm_topk_prob": True, "num_attention_heads": 8, "swa_num_attention_heads": 8,
    "num_experts_per_tok": 8, "num_hidden_layers": 4, "num_key_value_heads": 2,
    "swa_num_key_value_heads": 4, "num_nextn_predict_layers": 0, "partial_rotary_factor": 0.334,
    "rope_parameters": {"partial_rotary_factor": 0.334, "rope_theta": 10000000.0, "rope_type": "default"},
    "rope_theta": 10000000.0, "swa_rope_theta": 10000.0, "routed_scaling_factor": None,
    "scoring_func": "sigmoid", "sliding_window": 8, "sliding_window_size": 8,
    "tie_word_embeddings": False, "topk_group": 1, "topk_method": "noaux_tc",
    "vocab_size": 512, "dtype": "float32",
    "quantization_config": {"activation_scheme": "dynamic", "fmt": "e4m3", "quant_method": "fp8",
                            "store_dtype": "mxfp4", "mxfp4_block_size": 32, "weight_block_size": [128, 128]},
}


def fp8_blocks(w: torch.Tensor):
    """e4m3 codes + f32 scale per 128x128 tile (scale = amax / 448)."""
    n, k = w.shape
    rb, cb = -(-n // BLOCK), -(-k // BLOCK)
    scale = torch.zeros(rb, cb)
    codes = torch.zeros_like(w)
    for i in range(rb):
        for j in range(cb):
            t = w[i * BLOCK:(i + 1) * BLOCK, j * BLOCK:(j + 1) * BLOCK]
            sc = t.abs().max().clamp(min=1e-12) / 448.0
            scale[i, j] = sc
            codes[i * BLOCK:(i + 1) * BLOCK, j * BLOCK:(j + 1) * BLOCK] = t / sc
    return codes.to(torch.float8_e4m3fn), scale


def fp8_decode(codes: torch.Tensor, scale: torch.Tensor):
    n, k = codes.shape
    s = scale.repeat_interleave(BLOCK, 0)[:n].repeat_interleave(BLOCK, 1)[:, :k]
    return codes.to(torch.float32) * s


def encode_qkv(q, k, v):
    """Storage = TP rank slabs [q_r | k_r | v_r], each tiled on its own; returns
    (codes, scale, decoded contiguous [q | k | v])."""
    codes, scales, dec = [], [], {0: [], 1: [], 2: []}
    for r in range(TP):
        parts = [t.chunk(TP, 0)[r] for t in (q, k, v)]
        c, s = fp8_blocks(torch.cat(parts))
        codes.append(c)
        scales.append(s)
        d = fp8_decode(c, s)
        o = 0
        for p, t in enumerate(parts):
            dec[p].append(d[o:o + t.shape[0]])
            o += t.shape[0]
    return torch.cat(codes), torch.cat(scales), torch.cat([torch.cat(dec[p]) for p in range(3)])


def mxfp4_encode(w: torch.Tensor):
    """OCP MXFP4 per 32-element group: U8 bytes (low nibble first) + E8M0 scale."""
    rows, cols = w.shape
    g = w.reshape(rows, cols // 32, 32)
    amax = g.abs().amax(-1, keepdim=True).clamp(min=1e-12)
    e = torch.ceil(torch.log2(amax / 6.0)).clamp(-127, 127)
    x = g / torch.exp2(e)
    mag = (x.abs().unsqueeze(-1) - E2M1).abs().argmin(-1)
    code = (mag | ((x < 0) & (mag > 0)).to(torch.int64) << 3).reshape(rows, cols)
    packed = (code[:, 0::2] | (code[:, 1::2] << 4)).to(torch.uint8)
    return packed, (e.squeeze(-1) + 127).to(torch.uint8)


def mxfp4_decode(packed: torch.Tensor, scale: torch.Tensor):
    lut = torch.cat([E2M1, -E2M1])
    p = packed.to(torch.int64)
    vals = torch.stack([lut[p & 0xF], lut[p >> 4]], -1).reshape(packed.shape[0], -1)
    return vals * torch.exp2(scale.to(torch.float32) - 127).repeat_interleave(32, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, type=Path, help="dir with modeling_mimo_v2.py + configuration_mimo_v2.py")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    pkg = args.out / "mimo_ref"
    pkg.mkdir(parents=True, exist_ok=True)
    for f in ("modeling_mimo_v2.py", "configuration_mimo_v2.py"):
        (pkg / f).write_text((args.ref / f).read_text())
    (pkg / "__init__.py").write_text("")
    sys.path.insert(0, str(args.out))
    from mimo_ref.configuration_mimo_v2 import MiMoV2Config
    from mimo_ref.modeling_mimo_v2 import MiMoV2ForCausalLM

    c = CONFIG
    h = c["hidden_size"]
    stored, ref = {}, {}

    def rnd(*shape, std=0.08):
        return torch.randn(*shape) * std

    def bf16(t):
        return t.to(torch.bfloat16)

    for name, shape in (("model.embed_tokens.weight", (c["vocab_size"], h)), ("lm_head.weight", (c["vocab_size"], h))):
        t = bf16(rnd(*shape, std=0.5))
        stored[name], ref[name] = t, t.float()
    stored["model.norm.weight"] = bf16(1 + rnd(h, std=0.1))
    ref["model.norm.weight"] = stored["model.norm.weight"].float()

    for li in range(c["num_hidden_layers"]):
        pre = f"model.layers.{li}."
        sliding = c["hybrid_layer_pattern"][li] == 1
        kvh = c["swa_num_key_value_heads"] if sliding else c["num_key_value_heads"]
        for n in ("input_layernorm", "post_attention_layernorm"):
            stored[pre + n + ".weight"] = bf16(1 + rnd(h, std=0.1))
            ref[pre + n + ".weight"] = stored[pre + n + ".weight"].float()
        q = rnd(c["num_attention_heads"] * c["head_dim"], h)
        k = rnd(kvh * c["head_dim"], h)
        v = rnd(kvh * c["v_head_dim"], h)
        codes, scale, dec = encode_qkv(q, k, v)
        stored[pre + "self_attn.qkv_proj.weight"], stored[pre + "self_attn.qkv_proj.weight_scale_inv"] = codes, scale
        ref[pre + "self_attn.qkv_proj.weight"] = dec
        o = bf16(rnd(h, c["num_attention_heads"] * c["v_head_dim"]))
        stored[pre + "self_attn.o_proj.weight"], ref[pre + "self_attn.o_proj.weight"] = o, o.float()
        if sliding:
            sink = bf16(rnd(c["num_attention_heads"], std=1.0))
            stored[pre + "self_attn.attention_sink_bias"], ref[pre + "self_attn.attention_sink_bias"] = sink, sink.float()
        if c["moe_layer_freq"][li]:
            gate = bf16(rnd(c["n_routed_experts"], h, std=0.3))
            bias = rnd(c["n_routed_experts"], std=0.1)
            stored[pre + "mlp.gate.weight"], ref[pre + "mlp.gate.weight"] = gate, gate.float()
            stored[pre + "mlp.gate.e_score_correction_bias"] = ref[pre + "mlp.gate.e_score_correction_bias"] = bias
            for e in range(c["n_routed_experts"]):
                for p, shape in (("gate_proj", (c["moe_intermediate_size"], h)), ("up_proj", (c["moe_intermediate_size"], h)),
                                 ("down_proj", (h, c["moe_intermediate_size"]))):
                    base = f"{pre}mlp.experts.{e}.{p}"
                    packed, sc = mxfp4_encode(rnd(*shape, std=0.15))
                    stored[base + ".weight"], stored[base + ".weight_scale"] = packed, sc
                    ref[base + ".weight"] = mxfp4_decode(packed, sc)
        else:
            for p, shape in (("gate_proj", (c["intermediate_size"], h)), ("up_proj", (c["intermediate_size"], h)),
                             ("down_proj", (h, c["intermediate_size"]))):
                codes, scale = fp8_blocks(rnd(*shape))
                stored[f"{pre}mlp.{p}.weight"], stored[f"{pre}mlp.{p}.weight_scale_inv"] = codes, scale
                ref[f"{pre}mlp.{p}.weight"] = fp8_decode(codes, scale)

    src = args.out / "src"
    src.mkdir(parents=True, exist_ok=True)
    shard = "model_pp0_ep0_shard0.safetensors"
    save_file({k: v.contiguous() for k, v in stored.items()}, str(src / shard))
    (src / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {k: shard for k in stored}}, indent=1))
    (src / "config.json").write_text(json.dumps(c, indent=1))

    cfg = MiMoV2Config(**{k: v for k, v in c.items() if k != "quantization_config"})
    cfg._attn_implementation = "eager"
    model = MiMoV2ForCausalLM(cfg).float().eval()
    missing, unexpected = model.load_state_dict(ref, strict=False)
    missing = [m for m in missing if "rotary_emb" not in m]
    assert not missing and not unexpected, (missing, unexpected)

    def ref_logits(x):
        # Explicit masks (the reference takes a per-layer-type dict): causal, and
        # transformers' sliding definition, a query and the window-1 keys before it.
        n = x.shape[1]
        q, kv = torch.arange(n)[:, None], torch.arange(n)[None, :]
        neg = torch.tensor(float("-inf"))
        full_mask = torch.where(kv <= q, 0.0, neg)[None, None]
        swa_mask = torch.where((kv <= q) & (kv > q - c["sliding_window"]), 0.0, neg)[None, None]
        masks = {"full_attention": full_mask, "sliding_window_attention": swa_mask}
        return model(x, attention_mask=masks, use_cache=False).logits[0]

    T = 40
    ids = torch.randint(2, c["vocab_size"], (1, T))
    with torch.no_grad():
        full = ref_logits(ids)
    fixture = {"input_ids": ids[0].to(torch.int32), "logits_full": full.float()}
    save_file({k: v.contiguous() for k, v in fixture.items()}, str(args.out / "fixture.safetensors"))
    print(f"wrote {src} and {args.out / 'fixture.safetensors'} (T={T})")


if __name__ == "__main__":
    main()
