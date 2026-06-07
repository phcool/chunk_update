from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ["HF_HOME"] = "/home/vrintern/tmp/.hf-cache"
os.environ["HF_XET_CACHE"] = "/home/vrintern/tmp/.hf-xet-cache"
os.environ["HF_MODULES_CACHE"] = "/home/vrintern/tmp/.hf-modules-cache"
os.environ["HF_DATASETS_CACHE"] = "/home/vrintern/tmp/.hf-cache/datasets"
os.environ["XDG_CACHE_HOME"] = "/home/vrintern/tmp/.cache"

import torch

from hybrid_quant.metrics import inference_mode
from hybrid_quant.modeling import (
    load_nemotron,
    new_runtime_caches,
    prepare_attention_kv_caches,
    reset_model_sequence_state,
)
from hybrid_quant.mxfp8_fused import (
    enable_mxfp8_fused_state_cache,
    initialize_mxfp8_fused_state_cache,
    quantize_current_mamba_states,
)


def cuda_sync(device):
    if torch.cuda.is_available() and torch.device(device).type == "cuda":
        torch.cuda.synchronize(torch.device(device))


def timed(stats: dict[str, float], key: str, device, fn):
    cuda_sync(device)
    start = time.perf_counter()
    out = fn()
    cuda_sync(device)
    stats[key] = stats.get(key, 0.0) + time.perf_counter() - start
    stats[key + "_calls"] = stats.get(key + "_calls", 0) + 1
    return out


def profile_mamba_modules(model):
    for layer in getattr(model.model, "layers", []):
        mamba = getattr(layer, "mamba", None)
        if mamba is not None and getattr(mamba, "_mxfp8_fused_patched", False):
            mamba._mxfp8_profile = {}


def collect_mamba_profile(model):
    total: dict[str, float] = {}
    layer_count = 0
    for layer in getattr(model.model, "layers", []):
        mamba = getattr(layer, "mamba", None)
        prof = getattr(mamba, "_mxfp8_profile", None) if mamba is not None else None
        if prof is None:
            continue
        layer_count += 1
        for key, value in prof.items():
            total[key] = total.get(key, 0.0) + value
    total["profiled_mamba_layers"] = layer_count
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/home/vrintern/tmp/models/Nemotron-Flash-1B")
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--prefill-length", type=int, default=2048)
    parser.add_argument("--decode-length", type=int, default=64)
    parser.add_argument("--mamba-group-size", type=int, default=32)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    model, _ = load_nemotron(
        args.model_path,
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )
    enable_mxfp8_fused_state_cache(model, group_size=args.mamba_group_size)
    initialize_mxfp8_fused_state_cache(model)

    vocab = model.config.vocab_size
    mem = int(getattr(model.config, "num_memory_tokens", 0) or 0)
    reset_model_sequence_state(model)
    prepare_attention_kv_caches(
        model,
        batch_size=args.batch_size,
        max_seqlen=args.prefill_length + args.decode_length + 1,
    )
    caches = new_runtime_caches(
        model,
        batch_size=args.batch_size,
        max_seqlen=args.prefill_length + args.decode_length + 1,
        kv_mode="normal",
    )
    stats: dict[str, float] = {}
    input_ids = torch.randint(0, vocab, (args.batch_size, args.prefill_length), device=model.device)
    with inference_mode():
        caches.mamba_inference_params.seqlen_offset = 0
        out = timed(
            stats,
            "prefill_forward_s",
            device,
            lambda: model(
                input_ids=input_ids,
                position_ids=None,
                past_key_values=caches.past_key_values,
                fla_past_key_values=caches.fla_past_key_values,
                mamba_inference_params=caches.mamba_inference_params,
                use_cache=True,
                calc_logits_for_entire_prompt=False,
            ),
        )
        timed(
            stats,
            "initial_mxfp8_state_quantize_s",
            device,
            lambda: quantize_current_mamba_states(model, caches.mamba_inference_params),
        )
        profile_mamba_modules(model)
        next_ids = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
        for i in range(args.decode_length):
            caches.mamba_inference_params.seqlen_offset = args.prefill_length + i
            pos = torch.full(
                (args.batch_size, 1),
                mem + args.prefill_length + i,
                device=model.device,
                dtype=torch.long,
            )
            out = timed(
                stats,
                "decode_model_forward_s",
                device,
                lambda: model(
                    input_ids=next_ids,
                    position_ids=pos,
                    past_key_values=caches.past_key_values,
                    fla_past_key_values=caches.fla_past_key_values,
                    mamba_inference_params=caches.mamba_inference_params,
                    use_cache=True,
                    calc_logits_for_entire_prompt=False,
                ),
            )
            next_ids = timed(
                stats,
                "decode_argmax_s",
                device,
                lambda: torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True),
            )

    mamba_profile = collect_mamba_profile(model)
    mamba_time_s = sum(v for k, v in mamba_profile.items() if k.endswith("_s"))
    decode_outer_s = stats["decode_model_forward_s"] + stats["decode_argmax_s"]
    sections = {}
    for key, value in sorted(mamba_profile.items()):
        if key.endswith("_s"):
            sections[key] = {
                "seconds": value,
                "pct_of_mamba_profiled_time": value / mamba_time_s * 100.0 if mamba_time_s else None,
                "pct_of_decode_outer_time": value / decode_outer_s * 100.0 if decode_outer_s else None,
            }
    result = {
        "batch_size": args.batch_size,
        "prefill_length": args.prefill_length,
        "decode_length": args.decode_length,
        "mamba_group_size": args.mamba_group_size,
        "outer_timings_s": stats,
        "decode_ms_per_token_outer": decode_outer_s / args.decode_length * 1000.0,
        "tokens_per_second_outer": args.batch_size * args.decode_length / decode_outer_s,
        "mamba_profile_total_s": mamba_time_s,
        "mamba_sections": sections,
        "raw_mamba_profile": mamba_profile,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
