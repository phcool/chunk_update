from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path

os.environ["HF_HOME"] = "/home/vrintern/tmp/.hf-cache"
os.environ["HF_XET_CACHE"] = "/home/vrintern/tmp/.hf-xet-cache"
os.environ["HF_MODULES_CACHE"] = "/home/vrintern/tmp/.hf-modules-cache"
os.environ["HF_DATASETS_CACHE"] = "/home/vrintern/tmp/.hf-cache/datasets"
os.environ["XDG_CACHE_HOME"] = "/home/vrintern/tmp/.cache"

import matplotlib.pyplot as plt
import torch

from hybrid_quant.modeling import load_nemotron
from hybrid_quant.mxfp8_fused import enable_mxfp8_fused_state_cache, quantize_current_mamba_states
from run_decode_ppl_latency_mamba_refresh import MambaChunkRefresh, cuda_sync, run_prefill, setup_caches


def mib(x: int | float) -> float:
    return float(x) / 1024.0 / 1024.0


def trace_mode(model, mode: str, batch_size: int, context_length: int, decode_length: int, prefill_chunk_size: int, refresh_interval: int):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(model.device)

    vocab = model.config.vocab_size
    mem = int(getattr(model.config, "num_memory_tokens", 0) or 0)
    input_ids = torch.randint(0, vocab, (batch_size, context_length), device=model.device)
    caches = setup_caches(model, batch_size, context_length + decode_length + 1, mode)
    trace = []
    refresh_events = []

    with torch.inference_mode():
        out = run_prefill(model, caches, input_ids, context_length, prefill_chunk_size)
        if mode != "normal":
            quantize_current_mamba_states(model, caches.mamba_inference_params)
        cuda_sync()
        baseline = torch.cuda.memory_allocated(model.device)
        trace.append({"token": 0, "allocated_mib": mib(baseline), "delta_mib": 0.0, "event": "after_prefill"})

        next_ids = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
        refresher = MambaChunkRefresh(model, caches.mamba_inference_params, refresh_interval) if mode == "mxfp8_refresh256" else None
        for i in range(decode_length):
            if refresher is not None and i % refresh_interval == 0:
                refresher.start()
                cuda_sync()
                trace.append(
                    {
                        "token": i,
                        "allocated_mib": mib(torch.cuda.memory_allocated(model.device)),
                        "delta_mib": mib(torch.cuda.memory_allocated(model.device) - baseline),
                        "event": "refresh_start",
                    }
                )
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
                cuda_sync()
                before = torch.cuda.memory_allocated(model.device)
                refresher.refresh()
                cuda_sync()
                after = torch.cuda.memory_allocated(model.device)
                refresh_events.append(
                    {
                        "token": i + 1,
                        "before_refresh_mib": mib(before),
                        "after_refresh_mib": mib(after),
                        "before_delta_mib": mib(before - baseline),
                        "after_delta_mib": mib(after - baseline),
                    }
                )
                trace.append(
                    {
                        "token": i + 1,
                        "allocated_mib": mib(after),
                        "delta_mib": mib(after - baseline),
                        "event": "refresh_done",
                    }
                )
            cuda_sync()
            alloc = torch.cuda.memory_allocated(model.device)
            trace.append({"token": i + 1, "allocated_mib": mib(alloc), "delta_mib": mib(alloc - baseline), "event": "decode"})
        if refresher is not None:
            refresher.finish()
    peak = torch.cuda.max_memory_allocated(model.device)
    del caches
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "mode": mode,
        "batch_size": batch_size,
        "context_length": context_length,
        "decode_length": decode_length,
        "baseline_mib": mib(baseline),
        "peak_mib": mib(peak),
        "peak_delta_mib": mib(peak - baseline),
        "trace": trace,
        "refresh_events": refresh_events,
    }


def plot_results(results, output_png: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=160)
    for ax, field, ylabel in [
        (axes[0], "allocated_mib", "Allocated memory (MiB)"),
        (axes[1], "delta_mib", "Delta from after-prefill (MiB)"),
    ]:
        for result in results:
            xs = [p["token"] for p in result["trace"] if p["event"] in {"after_prefill", "decode", "refresh_done"}]
            ys = [p[field] for p in result["trace"] if p["event"] in {"after_prefill", "decode", "refresh_done"}]
            ax.plot(xs, ys, label=result["mode"], linewidth=1.6)
            if result["mode"] == "mxfp8_refresh256":
                for ev in result["refresh_events"]:
                    ax.axvline(ev["token"], color="tab:red", alpha=0.18, linewidth=0.9)
        ax.set_xlabel("Decode token")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend()
    title = f"Decode memory trace, ctx={results[0]['context_length']}, bs={results[0]['batch_size']}, decode={results[0]['decode_length']}"
    fig.suptitle(title)
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/home/vrintern/tmp/models/Nemotron-Flash-1B")
    parser.add_argument("--output-json", default="results/decode_memory_trace_ctx2048_bs32.json")
    parser.add_argument("--output-png", default="results/decode_memory_trace_ctx2048_bs32.png")
    parser.add_argument("--modes", default="normal,mxfp8_fused,mxfp8_refresh256")
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--decode-length", type=int, default=1000)
    parser.add_argument("--refresh-interval", type=int, default=256)
    parser.add_argument("--prefill-chunk-size", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    args = parser.parse_args()

    torch.cuda.set_device(torch.device(args.device))
    model, _ = load_nemotron(args.model_path, device=args.device, dtype=args.dtype, attn_implementation=args.attn_implementation)
    enable_mxfp8_fused_state_cache(model, group_size=32)
    results = []
    for mode in [x.strip() for x in args.modes.split(",") if x.strip()]:
        row = trace_mode(model, mode, args.batch_size, args.context_length, args.decode_length, args.prefill_chunk_size, args.refresh_interval)
        results.append(row)
        print(json.dumps({k: v for k, v in row.items() if k not in {"trace", "refresh_events"}}), flush=True)
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps({"results": results}, indent=2), encoding="utf-8")
    plot_results(results, Path(args.output_png))


if __name__ == "__main__":
    main()
