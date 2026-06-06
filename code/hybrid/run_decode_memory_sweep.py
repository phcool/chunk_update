from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys

_CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_CODE_ROOT / "shared"))
from typing import Any

import torch

from hybrid_quant.modeling import (
    load_nemotron,
    new_runtime_caches,
    prepare_attention_kv_caches,
    reset_model_sequence_state,
)
from hybrid_quant.mxfp8_fused import (
    enable_mxfp8_fused_state_cache,
    initialize_mxfp8_fused_state_cache,
    replace_mamba_states_with_mxfp8,
)


def parse_ints(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def tensor_bytes(x: torch.Tensor) -> int:
    return x.numel() * x.element_size()


def tree_tensor_bytes(obj: Any) -> int:
    if isinstance(obj, torch.Tensor):
        return tensor_bytes(obj)
    if hasattr(obj, "q_state") and hasattr(obj, "scale_e8m0"):
        return tensor_bytes(obj.q_state) + tensor_bytes(obj.scale_e8m0)
    if isinstance(obj, dict):
        return sum(tree_tensor_bytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(tree_tensor_bytes(v) for v in obj)
    return 0


def initialize_mamba_states(model, inference_params, batch_size: int):
    for layer in getattr(model.model, "layers", []):
        mamba = getattr(layer, "mamba", None)
        if mamba is not None and hasattr(mamba, "_get_states_from_cache"):
            mamba._get_states_from_cache(inference_params, batch_size)


def mamba_state_bytes(inference_params) -> int:
    return tree_tensor_bytes(getattr(inference_params, "key_value_memory_dict", {}))


def cuda_snapshot(device: torch.device) -> dict[str, int]:
    if device.type != "cuda":
        return {}
    torch.cuda.synchronize(device)
    return {
        "allocated_bytes": torch.cuda.memory_allocated(device),
        "reserved_bytes": torch.cuda.memory_reserved(device),
        "max_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "max_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }


def mib(x: int | float | None) -> float | None:
    return None if x is None else float(x) / 1024.0 / 1024.0


def measure(model, batch_size: int, seq_len: int, mode: str, device: torch.device, group_size: int):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    reset_model_sequence_state(model)
    if mode == "mxfp8_decode_native":
        initialize_mxfp8_fused_state_cache(model)

    before = cuda_snapshot(device)
    prepare_attention_kv_caches(model, batch_size=batch_size, max_seqlen=seq_len + 1)
    caches = new_runtime_caches(model, batch_size=batch_size, max_seqlen=seq_len + 1, kv_mode="normal")
    initialize_mamba_states(model, caches.mamba_inference_params, batch_size)
    after_normal_cache = cuda_snapshot(device)
    normal_state_bytes = mamba_state_bytes(caches.mamba_inference_params)

    if mode == "mxfp8_decode_native":
        replace_mamba_states_with_mxfp8(model, caches.mamba_inference_params)
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    after_mode_cache = cuda_snapshot(device)
    mode_state_bytes = mamba_state_bytes(caches.mamba_inference_params)

    row = {
        "mode": mode,
        "batch_size": batch_size,
        "sequence_length": seq_len,
        "before_allocated_mib": mib(before.get("allocated_bytes")),
        "after_normal_cache_allocated_mib": mib(after_normal_cache.get("allocated_bytes")),
        "after_mode_cache_allocated_mib": mib(after_mode_cache.get("allocated_bytes")),
        "after_normal_cache_reserved_mib": mib(after_normal_cache.get("reserved_bytes")),
        "after_mode_cache_reserved_mib": mib(after_mode_cache.get("reserved_bytes")),
        "normal_state_mib": mib(normal_state_bytes),
        "mode_state_mib": mib(mode_state_bytes),
        "allocated_delta_vs_normal_cache_mib": mib(
            after_mode_cache.get("allocated_bytes", 0) - after_normal_cache.get("allocated_bytes", 0)
        ),
        "state_delta_vs_normal_mib": mib(mode_state_bytes - normal_state_bytes),
        "snapshots": {
            "before": before,
            "after_normal_cache": after_normal_cache,
            "after_mode_cache": after_mode_cache,
        },
    }

    del caches
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/home/vrintern/tmp/models/Nemotron-Flash-1B")
    parser.add_argument("--output", default="results/decode_memory_sweep.json")
    parser.add_argument("--batch-sizes", default="1,32,64,128,256")
    parser.add_argument("--sequence-lengths", default="2048,8192")
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
        for seq_len in parse_ints(args.sequence_lengths):
            rows.append(measure(model, batch_size, seq_len, "normal", device, args.mamba_group_size))
            rows.append(measure(model, batch_size, seq_len, "mxfp8_decode_native", device, args.mamba_group_size))

    payload = {
        "note": (
            "Decode-cache total memory sweep. mxfp8_decode_native replaces BF16 Mamba ssm_state entries "
            "with MXFP8 E4M3+E8M0 caches after cache initialization, so allocated_delta_vs_normal_cache_mib "
            "estimates total CUDA allocated memory difference for decode cache setup. This does not modify prefill."
        ),
        "model_path": args.model_path,
        "dtype": args.dtype,
        "mamba_group_size": args.mamba_group_size,
        "results": rows,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
