from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import torch

from hybrid_quant.modeling import load_nemotron, new_runtime_caches, reset_model_sequence_state
from hybrid_quant.mxfp8_fused import (
    enable_mxfp8_fused_state_cache,
    initialize_mxfp8_fused_state_cache,
    quantize_current_mamba_states,
)


def parse_ints(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def tensor_bytes(x: torch.Tensor) -> int:
    return x.numel() * x.element_size()


def tree_tensor_bytes(obj: Any) -> int:
    if isinstance(obj, torch.Tensor):
        return tensor_bytes(obj)
    if isinstance(obj, dict):
        return sum(tree_tensor_bytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(tree_tensor_bytes(v) for v in obj)
    return 0


def mamba_state_breakdown(inference_params) -> dict[str, int]:
    conv_bytes = 0
    ssm_bytes = 0
    other_bytes = 0
    state_dict = getattr(inference_params, "key_value_memory_dict", {}) or {}
    for value in state_dict.values():
        if isinstance(value, tuple) and len(value) >= 2:
            conv_bytes += tree_tensor_bytes(value[0])
            ssm_bytes += tree_tensor_bytes(value[1])
            for extra in value[2:]:
                other_bytes += tree_tensor_bytes(extra)
        else:
            other_bytes += tree_tensor_bytes(value)
    return {
        "mamba_conv_state_bytes": conv_bytes,
        "mamba_ssm_state_bytes": ssm_bytes,
        "mamba_other_state_bytes": other_bytes,
        "mamba_total_state_bytes": conv_bytes + ssm_bytes + other_bytes,
    }


def fused_sidecar_breakdown(model) -> dict[str, int]:
    q_bytes = 0
    scale_bytes = 0
    cache_count = 0
    for layer in getattr(model.model, "layers", []):
        mamba = getattr(layer, "mamba", None)
        if mamba is None:
            continue
        for cache in getattr(mamba, "_mxfp8_fused_caches", {}).values():
            cache_count += 1
            q_bytes += tensor_bytes(cache.q_state)
            scale_bytes += tensor_bytes(cache.scale_e8m0)
    return {
        "mxfp8_cache_count": cache_count,
        "mxfp8_q_state_bytes": q_bytes,
        "mxfp8_e8m0_scale_bytes": scale_bytes,
        "mxfp8_sidecar_total_bytes": q_bytes + scale_bytes,
    }


def initialize_mamba_states(model, inference_params, batch_size: int):
    for layer in getattr(model.model, "layers", []):
        mamba = getattr(layer, "mamba", None)
        if mamba is not None and hasattr(mamba, "_get_states_from_cache"):
            mamba._get_states_from_cache(inference_params, batch_size)


def cuda_mem(device: torch.device) -> dict[str, int]:
    if device.type != "cuda":
        return {}
    torch.cuda.synchronize(device)
    return {
        "cuda_allocated_bytes": torch.cuda.memory_allocated(device),
        "cuda_reserved_bytes": torch.cuda.memory_reserved(device),
        "cuda_max_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "cuda_max_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }


def to_mib(value: int | float | None) -> float | None:
    return None if value is None else float(value) / 1024.0 / 1024.0


def measure_one(model, batch_size: int, max_seqlen: int, mode: str, device: torch.device, group_size: int):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    reset_model_sequence_state(model)
    if mode == "mxfp8_fused":
        initialize_mxfp8_fused_state_cache(model)

    before = cuda_mem(device)
    caches = new_runtime_caches(model, batch_size=batch_size, max_seqlen=max_seqlen, kv_mode="normal")
    initialize_mamba_states(model, caches.mamba_inference_params, batch_size)
    after_cache = cuda_mem(device)
    normal_breakdown = mamba_state_breakdown(caches.mamba_inference_params)

    if mode == "mxfp8_fused":
        quantize_current_mamba_states(model, caches.mamba_inference_params)
    after_quant = cuda_mem(device)
    sidecar = fused_sidecar_breakdown(model) if mode == "mxfp8_fused" else {
        "mxfp8_cache_count": 0,
        "mxfp8_q_state_bytes": 0,
        "mxfp8_e8m0_scale_bytes": 0,
        "mxfp8_sidecar_total_bytes": 0,
    }

    normal_mamba_bytes = normal_breakdown["mamba_total_state_bytes"]
    theoretical_replacement_bytes = (
        normal_breakdown["mamba_conv_state_bytes"]
        + normal_breakdown["mamba_other_state_bytes"]
        + sidecar["mxfp8_sidecar_total_bytes"]
    )
    current_impl_bytes = normal_mamba_bytes + sidecar["mxfp8_sidecar_total_bytes"]
    theoretical_saving = normal_mamba_bytes - theoretical_replacement_bytes

    row = {
        "mode": mode,
        "batch_size": batch_size,
        "max_seqlen": max_seqlen,
        **normal_breakdown,
        **sidecar,
        "theoretical_mxfp8_replacement_state_bytes": theoretical_replacement_bytes if mode == "mxfp8_fused" else None,
        "current_impl_state_plus_sidecar_bytes": current_impl_bytes if mode == "mxfp8_fused" else normal_mamba_bytes,
        "theoretical_state_saving_bytes": theoretical_saving if mode == "mxfp8_fused" else 0,
        "theoretical_state_saving_mib": to_mib(theoretical_saving if mode == "mxfp8_fused" else 0),
        "mamba_total_state_mib": to_mib(normal_mamba_bytes),
        "mxfp8_sidecar_total_mib": to_mib(sidecar["mxfp8_sidecar_total_bytes"]),
        "theoretical_mxfp8_replacement_state_mib": to_mib(theoretical_replacement_bytes if mode == "mxfp8_fused" else None),
        "current_impl_state_plus_sidecar_mib": to_mib(current_impl_bytes if mode == "mxfp8_fused" else normal_mamba_bytes),
        "cuda_before": before,
        "cuda_after_cache": after_cache,
        "cuda_after_quant": after_quant,
    }
    if before and after_cache:
        row["cuda_cache_alloc_delta_bytes"] = after_cache["cuda_allocated_bytes"] - before["cuda_allocated_bytes"]
        row["cuda_cache_alloc_delta_mib"] = to_mib(row["cuda_cache_alloc_delta_bytes"])
    if after_cache and after_quant:
        row["cuda_quant_alloc_delta_bytes"] = after_quant["cuda_allocated_bytes"] - after_cache["cuda_allocated_bytes"]
        row["cuda_quant_alloc_delta_mib"] = to_mib(row["cuda_quant_alloc_delta_bytes"])

    del caches
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/home/vrintern/tmp/models/Nemotron-Flash-1B")
    parser.add_argument("--output", default="results/state_memory_sweep.json")
    parser.add_argument("--batch-sizes", default="1,32,64,128,256")
    parser.add_argument("--sequence-lengths", default="2048,8192")
    parser.add_argument("--modes", default="normal,mxfp8_fused")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--mamba-group-size", type=int, default=32)
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

    rows = []
    for batch_size in parse_ints(args.batch_sizes):
        for seqlen in parse_ints(args.sequence_lengths):
            for mode in [x.strip() for x in args.modes.split(",") if x.strip()]:
                rows.append(measure_one(model, batch_size, seqlen, mode, device, args.mamba_group_size))

    payload = {
        "note": (
            "Mamba recurrent state memory is sequence-length independent; sequence length is passed to cache init "
            "to verify this. Current mxfp8_fused implementation keeps the original BF16 ssm_state tensor and adds "
            "an FP8+E8M0 sidecar. theoretical_mxfp8_replacement_* reports memory if BF16 ssm_state were replaced."
        ),
        "model_path": args.model_path,
        "dtype": args.dtype,
        "mamba_group_size": args.mamba_group_size,
        "results": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
