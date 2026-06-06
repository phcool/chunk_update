from __future__ import annotations

import argparse
import gc
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

from hybrid_quant.modeling import load_nemotron
from hybrid_quant.mxfp8_fused import enable_mxfp8_fused_state_cache, quantize_current_mamba_states
from run_decode_ppl_latency_mamba_refresh import (
    MambaChunkRefresh,
    cuda_sync,
    parse_ints,
    run_prefill,
    setup_caches,
)


def mib(x: int | float) -> float:
    return float(x) / 1024.0 / 1024.0


def measure_one(model, mode: str, batch_size: int, context_length: int, decode_length: int, prefill_chunk_size: int, refresh_interval: int):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(model.device)

    vocab = model.config.vocab_size
    mem = int(getattr(model.config, "num_memory_tokens", 0) or 0)
    input_ids = torch.randint(0, vocab, (batch_size, context_length), device=model.device)
    caches = setup_caches(model, batch_size, context_length + decode_length + 1, mode)
    with torch.inference_mode():
        out = run_prefill(model, caches, input_ids, context_length, prefill_chunk_size)
        if mode != "normal":
            quantize_current_mamba_states(model, caches.mamba_inference_params)
        cuda_sync()
        after_prefill_alloc = torch.cuda.memory_allocated(model.device)
        after_prefill_reserved = torch.cuda.memory_reserved(model.device)
        torch.cuda.reset_peak_memory_stats(model.device)

        next_ids = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
        refresher = MambaChunkRefresh(model, caches.mamba_inference_params, refresh_interval) if mode == "mxfp8_refresh256" else None
        start = time.perf_counter()
        for i in range(decode_length):
            if refresher is not None and i % refresh_interval == 0:
                refresher.start()
            caches.mamba_inference_params.seqlen_offset = context_length + i
            pos = torch.full((batch_size, 1), mem + context_length + i, device=model.device, dtype=torch.long)
            out = model(
                input_ids=next_ids,
                position_ids=pos,
                past_key_values=caches.past_key_values,
                fla_past_key_values=caches.fla_past_key_values,
                mamba_inference_params=caches.mamba_inference_params,
                use_cache=True,
                calc_logits_for_entire_prompt=False,
            )
            next_ids = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
            if refresher is not None and (i + 1) % refresh_interval == 0:
                refresher.refresh()
        if refresher is not None:
            refresher.finish()
        cuda_sync()
        elapsed = time.perf_counter() - start
        after_decode_alloc = torch.cuda.memory_allocated(model.device)
        decode_peak_alloc = torch.cuda.max_memory_allocated(model.device)
        decode_peak_reserved = torch.cuda.max_memory_reserved(model.device)

    del caches
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "mode": mode,
        "batch_size": batch_size,
        "context_length": context_length,
        "decode_length": decode_length,
        "refresh_interval": refresh_interval if mode == "mxfp8_refresh256" else None,
        "after_prefill_allocated_mib": mib(after_prefill_alloc),
        "after_prefill_reserved_mib": mib(after_prefill_reserved),
        "after_decode_allocated_mib": mib(after_decode_alloc),
        "decode_peak_allocated_mib": mib(decode_peak_alloc),
        "decode_peak_reserved_mib": mib(decode_peak_reserved),
        "decode_peak_minus_after_prefill_mib": mib(decode_peak_alloc - after_prefill_alloc),
        "decode_latency_per_token_s": elapsed / decode_length,
        "tokens_per_s": batch_size * decode_length / elapsed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/home/vrintern/tmp/models/Nemotron-Flash-1B")
    parser.add_argument("--output", default="results/decode_refresh_memory.json")
    parser.add_argument("--modes", default="normal,mxfp8_fused,mxfp8_refresh256")
    parser.add_argument("--context-lengths", default="2048,8192")
    parser.add_argument("--batch-sizes", default="1,32")
    parser.add_argument("--decode-length", type=int, default=1000)
    parser.add_argument("--refresh-interval", type=int, default=256)
    parser.add_argument("--prefill-chunk-size", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    args = parser.parse_args()

    torch.cuda.set_device(torch.device(args.device))
    model, _ = load_nemotron(args.model_path, device=args.device, dtype=args.dtype, attn_implementation=args.attn_implementation)
    if any(x.strip() != "normal" for x in args.modes.split(",")):
        enable_mxfp8_fused_state_cache(model, group_size=32)

    rows = []
    for ctx in parse_ints(args.context_lengths):
        for bs in parse_ints(args.batch_sizes):
            for mode in [x.strip() for x in args.modes.split(",") if x.strip()]:
                row = measure_one(model, mode, bs, ctx, args.decode_length, args.prefill_chunk_size, args.refresh_interval)
                rows.append(row)
                print(json.dumps(row), flush=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"results": rows}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
