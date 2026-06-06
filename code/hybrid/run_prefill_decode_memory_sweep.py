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
from hybrid_quant.quant import QuantizedTensor


def parse_csv(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def parse_ints(value: str) -> list[int]:
    return [int(x) for x in parse_csv(value)]


def tensor_bytes(x: torch.Tensor) -> int:
    return x.numel() * x.element_size()


def tree_bytes(obj: Any) -> int:
    if isinstance(obj, torch.Tensor):
        return tensor_bytes(obj)
    if isinstance(obj, QuantizedTensor):
        scale = 0 if obj.scale is None else tensor_bytes(obj.scale)
        zero = 0 if obj.zero is None else tensor_bytes(obj.zero)
        return tensor_bytes(obj.q) + scale + zero
    if hasattr(obj, "q_state") and hasattr(obj, "scale_e8m0"):
        return tensor_bytes(obj.q_state) + tensor_bytes(obj.scale_e8m0)
    if isinstance(obj, dict):
        return sum(tree_bytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(tree_bytes(v) for v in obj)
    return 0


def attention_kv_bytes(past_key_values) -> int:
    return tree_bytes(getattr(past_key_values, "key_cache", [])) + tree_bytes(
        getattr(past_key_values, "value_cache", [])
    )


def mamba_state_bytes(inference_params) -> int:
    return tree_bytes(getattr(inference_params, "key_value_memory_dict", {}))


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


def run_prefill(model, caches, batch_size: int, seq_len: int, chunk_size: int):
    vocab = model.config.vocab_size
    mem = int(getattr(model.config, "num_memory_tokens", 0) or 0)
    input_ids = torch.randint(0, vocab, (batch_size, seq_len), device=model.device)
    out = None
    with torch.inference_mode():
        for start in range(0, seq_len, chunk_size):
            end = min(start + chunk_size, seq_len)
            caches.mamba_inference_params.seqlen_offset = start
            if start == 0:
                pos = None
            else:
                pos_values = torch.arange(mem + start, mem + end, device=model.device, dtype=torch.long)
                pos = pos_values.unsqueeze(0).expand(batch_size, -1)
            out = model(
                input_ids=input_ids[:, start:end],
                position_ids=pos,
                past_key_values=caches.past_key_values,
                fla_past_key_values=caches.fla_past_key_values,
                mamba_inference_params=caches.mamba_inference_params,
                use_cache=True,
                calc_logits_for_entire_prompt=False,
            )
    return out


def measure_one(model, batch_size: int, seq_len: int, kv_mode: str, mamba_mode: str, device: torch.device, chunk_size: int):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    reset_model_sequence_state(model)
    initialize_mxfp8_fused_state_cache(model)
    before = cuda_snapshot(device)
    try:
        prepare_attention_kv_caches(model, batch_size=batch_size, max_seqlen=seq_len + 1)
        caches = new_runtime_caches(model, batch_size=batch_size, max_seqlen=seq_len + 1, kv_mode=kv_mode)
        run_prefill(model, caches, batch_size=batch_size, seq_len=seq_len, chunk_size=chunk_size)
        after_prefill = cuda_snapshot(device)
        normal_mamba_bytes = mamba_state_bytes(caches.mamba_inference_params)
        if mamba_mode == "mxfp8_decode_native":
            replace_mamba_states_with_mxfp8(model, caches.mamba_inference_params)
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        after_mode = cuda_snapshot(device)
        row = {
            "status": "ok",
            "batch_size": batch_size,
            "sequence_length": seq_len,
            "kv_mode": kv_mode,
            "mamba_mode": mamba_mode,
            "prefill_chunk_size": chunk_size,
            "before_allocated_mib": mib(before.get("allocated_bytes")),
            "after_prefill_allocated_mib": mib(after_prefill.get("allocated_bytes")),
            "after_mode_allocated_mib": mib(after_mode.get("allocated_bytes")),
            "after_prefill_reserved_mib": mib(after_prefill.get("reserved_bytes")),
            "after_mode_reserved_mib": mib(after_mode.get("reserved_bytes")),
            "prefill_peak_allocated_mib": mib(after_prefill.get("max_allocated_bytes")),
            "mode_peak_allocated_mib": mib(after_mode.get("max_allocated_bytes")),
            "allocated_delta_after_mamba_mode_mib": mib(
                after_mode.get("allocated_bytes", 0) - after_prefill.get("allocated_bytes", 0)
            ),
            "kv_cache_mib": mib(attention_kv_bytes(caches.past_key_values)),
            "mamba_state_before_mode_mib": mib(normal_mamba_bytes),
            "mamba_state_after_mode_mib": mib(mamba_state_bytes(caches.mamba_inference_params)),
            "snapshots": {
                "before": before,
                "after_prefill": after_prefill,
                "after_mode": after_mode,
            },
        }
        del caches
    except torch.cuda.OutOfMemoryError as exc:
        row = {
            "status": "oom",
            "batch_size": batch_size,
            "sequence_length": seq_len,
            "kv_mode": kv_mode,
            "mamba_mode": mamba_mode,
            "prefill_chunk_size": chunk_size,
            "error": str(exc),
        }
    except Exception as exc:
        row = {
            "status": "error",
            "batch_size": batch_size,
            "sequence_length": seq_len,
            "kv_mode": kv_mode,
            "mamba_mode": mamba_mode,
            "prefill_chunk_size": chunk_size,
            "error": str(exc),
        }
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/home/vrintern/tmp/models/Nemotron-Flash-1B")
    parser.add_argument("--output", default="results/prefill_decode_memory_sweep.json")
    parser.add_argument("--batch-sizes", default="1,32,64,128,256")
    parser.add_argument("--sequence-lengths", default="2048,8192")
    parser.add_argument("--kv-modes", default="normal,fp8,int4")
    parser.add_argument("--mamba-modes", default="normal,mxfp8_decode_native")
    parser.add_argument("--prefill-chunk-size", type=int, default=256)
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
            for kv_mode in parse_csv(args.kv_modes):
                for mamba_mode in parse_csv(args.mamba_modes):
                    rows.append(measure_one(model, batch_size, seq_len, kv_mode, mamba_mode, device, args.prefill_chunk_size))
                    print(json.dumps(rows[-1]), flush=True)

    payload = {
        "note": (
            "Real decode memory sweep after chunked prefill. KV cache is populated by prefill. "
            "mxfp8_decode_native replaces BF16 Mamba ssm_state with MXFP8 cache after prefill, before decode."
        ),
        "dtype": args.dtype,
        "prefill_chunk_size": args.prefill_chunk_size,
        "results": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
