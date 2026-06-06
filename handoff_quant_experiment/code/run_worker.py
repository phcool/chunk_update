from __future__ import annotations

import argparse
import json
import os
import traceback
from pathlib import Path

os.environ["HF_HOME"] = "/home/vrintern/tmp/.hf-cache"
os.environ["HF_XET_CACHE"] = "/home/vrintern/tmp/.hf-xet-cache"
os.environ["HF_MODULES_CACHE"] = "/home/vrintern/tmp/.hf-modules-cache"
os.environ["HF_DATASETS_CACHE"] = "/home/vrintern/tmp/.hf-cache/datasets"
os.environ["XDG_CACHE_HOME"] = "/home/vrintern/tmp/.cache"

import torch

from hybrid_quant.data import load_wikitext_texts, token_segments
from hybrid_quant.metrics import cached_teacher_forcing_ppl, latency_benchmark
from hybrid_quant.modeling import load_nemotron


def write_jsonl(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_int_list(value: str):
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/home/vrintern/tmp/models/Nemotron-Flash-1B")
    parser.add_argument("--dataset-path", default="/home/vrintern/tmp/datasets/wikitext")
    parser.add_argument("--output", required=True)
    parser.add_argument("--kv-mode", required=True, choices=["normal", "fp8", "int4"])
    parser.add_argument("--mamba-mode", required=True, choices=["normal", "mxfp8", "mxfp8_fused"])
    parser.add_argument("--metrics", default="ppl,latency")
    parser.add_argument("--context-lengths", default="2048,8192")
    parser.add_argument("--batch-sizes", default="1,32")
    parser.add_argument("--decode-length", type=int, default=256)
    parser.add_argument("--ppl-max-eval-tokens", type=int, default=8192)
    parser.add_argument("--ppl-split", default="test")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--device", default=None)
    parser.add_argument("--ready-file", default=None)
    parser.add_argument("--latency-warmup", type=int, default=1)
    parser.add_argument("--latency-repeats", type=int, default=3)
    parser.add_argument("--prefill-chunk-size", type=int, default=0)
    parser.add_argument("--mamba-group-size", type=int, default=32)
    args = parser.parse_args()

    output = Path(args.output)
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    cuda_debug = {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_available": torch.cuda.is_available(),
    }
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.set_device(torch.device(device))
    metrics = {x.strip() for x in args.metrics.split(",") if x.strip()}
    context_lengths = parse_int_list(args.context_lengths)
    batch_sizes = parse_int_list(args.batch_sizes)

    base = {
        "worker_pid": os.getpid(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device": device,
        **cuda_debug,
        "model_path": args.model_path,
        "dataset_path": args.dataset_path,
        "kv_mode": args.kv_mode,
        "mamba_state_mode": args.mamba_mode,
        "dtype": args.dtype,
        "attn_implementation": args.attn_implementation,
    }

    try:
        model, tokenizer = load_nemotron(
            args.model_path,
            device=device,
            dtype=args.dtype,
            attn_implementation=args.attn_implementation,
        )
        if args.ready_file:
            ready_path = Path(args.ready_file)
            ready_path.parent.mkdir(parents=True, exist_ok=True)
            ready_path.write_text(json.dumps({**base, "status": "ready"}), encoding="utf-8")

        if "ppl" in metrics:
            texts = load_wikitext_texts(args.dataset_path, split=args.ppl_split)
            for ctx in context_lengths:
                segs = token_segments(
                    tokenizer,
                    texts,
                    context_length=ctx,
                    max_eval_tokens=args.ppl_max_eval_tokens,
                )
                result = cached_teacher_forcing_ppl(
                    model,
                    segs,
                    kv_mode=args.kv_mode,
                    mamba_mode=args.mamba_mode,
                    context_length=ctx,
                    mamba_group_size=args.mamba_group_size,
                )
                write_jsonl(output, {**base, "metric": "ppl", "context_length": ctx, **result})

        if "latency" in metrics:
            for ctx in context_lengths:
                for batch_size in batch_sizes:
                    try:
                        result = latency_benchmark(
                            model,
                            kv_mode=args.kv_mode,
                            mamba_mode=args.mamba_mode,
                            batch_size=batch_size,
                            prefill_length=ctx,
                            decode_length=args.decode_length,
                            warmup=args.latency_warmup,
                            repeats=args.latency_repeats,
                            mamba_group_size=args.mamba_group_size,
                            prefill_chunk_size=args.prefill_chunk_size,
                        )
                        row = {
                            **base,
                            "metric": "latency",
                            "prefill_length": ctx,
                            "batch_size": batch_size,
                            **result,
                        }
                    except torch.cuda.OutOfMemoryError as exc:
                        torch.cuda.empty_cache()
                        row = {
                            **base,
                            "metric": "latency",
                            "prefill_length": ctx,
                            "batch_size": batch_size,
                            "status": "oom",
                            "error": str(exc),
                        }
                    write_jsonl(output, row)
    except Exception as exc:
        write_jsonl(output, {**base, "status": "error", "error": str(exc), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
