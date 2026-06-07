from __future__ import annotations

import argparse
import gc
import json
import math
import os
import time
from pathlib import Path
import sys

_CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_CODE_ROOT / "shared"))
sys.path.insert(0, str(_CODE_ROOT / "hybrid" / "chunk_update"))

os.environ["HF_HOME"] = "/home/vrintern/tmp/.hf-cache"
os.environ["HF_XET_CACHE"] = "/home/vrintern/tmp/.hf-xet-cache"
os.environ["HF_MODULES_CACHE"] = "/home/vrintern/tmp/.hf-modules-cache"
os.environ["HF_DATASETS_CACHE"] = "/home/vrintern/tmp/.hf-cache/datasets"
os.environ["XDG_CACHE_HOME"] = "/home/vrintern/tmp/.cache"

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.utils.generation import InferenceParams
from transformers import AutoTokenizer

from hybrid_quant.data import load_wikitext_texts
from hybrid_quant.mxfp8_fused import (
    enable_mxfp8_fused_state_cache,
    initialize_mxfp8_fused_state_cache,
    quantize_current_mamba_states,
    replace_mamba_states_with_mxfp8,
)
from run_decode_ppl_latency_mamba_refresh import MambaChunkRefresh, cuda_sync


MODES = {
    "normal",
    "mxfp8_fused",
    "mxfp8_refresh256",
    "mxfp8_fused_native",
    "mxfp8_refresh256_native",
}


def parse_csv(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def parse_ints(value: str) -> list[int]:
    return [int(x) for x in parse_csv(value)]


def mib(x: int | float) -> float:
    return float(x) / 1024.0 / 1024.0


def load_model(model_path: str, tokenizer_path: str, device: str, dtype: str):
    torch_dtype = {"bfloat16": torch.bfloat16, "bf16": torch.bfloat16, "float16": torch.float16, "fp16": torch.float16}[dtype]
    model = MambaLMHeadModel.from_pretrained(model_path, device=device, dtype=torch_dtype).eval()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def new_inference_params(model, batch_size: int, max_seqlen: int, dtype: torch.dtype):
    inference_params = InferenceParams(max_seqlen=max_seqlen, max_batch_size=batch_size)
    inference_params.key_value_memory_dict = model.backbone.allocate_inference_cache(
        batch_size=batch_size,
        max_seqlen=max_seqlen,
        dtype=dtype,
    )
    return inference_params


def token_windows(tokenizer, texts, total_length: int, max_windows: int):
    ids = tokenizer("\n\n".join(texts), add_special_tokens=False, return_tensors="pt").input_ids[0]
    windows = []
    pos = 0
    while pos + total_length <= ids.numel() and len(windows) < max_windows:
        windows.append(ids[pos : pos + total_length].clone())
        pos += total_length - 1
    return windows


def setup_mode(model, mode: str):
    if mode != "normal":
        enable_mxfp8_fused_state_cache(model, group_size=32)
        initialize_mxfp8_fused_state_cache(model)


def is_refresh_mode(mode: str) -> bool:
    return mode in {"mxfp8_refresh256", "mxfp8_refresh256_native"}


def activate_mxfp8_state(model, inference_params, mode: str):
    if mode == "normal":
        return
    if mode.endswith("_native"):
        replace_mamba_states_with_mxfp8(model, inference_params)
    else:
        quantize_current_mamba_states(model, inference_params)


def prefill(model, input_ids, inference_params, context_length: int, step_prefill: bool = False):
    if not step_prefill:
        inference_params.seqlen_offset = 0
        return model(input_ids[:, :context_length], inference_params=inference_params, num_last_tokens=1)
    out = None
    for i in range(context_length):
        inference_params.seqlen_offset = i
        out = model(input_ids[:, i : i + 1], inference_params=inference_params)
    return out


def should_step_prefill(batch_size: int, context_length: int) -> bool:
    return batch_size >= 16 and context_length >= 8192


def decode_ppl(model, windows, mode: str, context_length: int, decode_length: int, refresh_interval: int):
    setup_mode(model, mode)
    total_nll = 0.0
    total_tokens = 0
    start = time.perf_counter()
    with torch.inference_mode():
        for ids in windows:
            ids = ids.to(model.lm_head.weight.device).unsqueeze(0)
            inference_params = new_inference_params(model, 1, context_length + decode_length + 1, model.lm_head.weight.dtype)
            prefill(model, ids, inference_params, context_length, step_prefill=False)
            activate_mxfp8_state(model, inference_params, mode)
            refresher = MambaChunkRefresh(model, inference_params, refresh_interval) if is_refresh_mode(mode) else None
            for i in range(decode_length):
                if refresher is not None and i % refresh_interval == 0:
                    refresher.start()
                inference_params.seqlen_offset = context_length + i
                out = model(ids[:, context_length + i : context_length + i + 1], inference_params=inference_params)
                loss = F.cross_entropy(out.logits[:, -1, :], ids[:, context_length + i + 1], reduction="sum")
                total_nll += float(loss.item())
                total_tokens += 1
                if refresher is not None and (i + 1) % refresh_interval == 0:
                    refresher.refresh()
            if refresher is not None:
                refresher.finish()
                refresher.clear()
            del inference_params
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    cuda_sync()
    elapsed = time.perf_counter() - start
    return {
        "metric": "decode_ppl",
        "mode": mode,
        "context_length": context_length,
        "decode_length": decode_length,
        "tokens": total_tokens,
        "ppl": math.exp(total_nll / max(total_tokens, 1)),
        "nll": total_nll,
        "elapsed_s": elapsed,
        "tokens_per_s": total_tokens / elapsed if elapsed > 0 else None,
    }


def latency_once(model, mode: str, batch_size: int, context_length: int, decode_length: int, refresh_interval: int):
    setup_mode(model, mode)
    device = model.lm_head.weight.device
    input_ids = torch.randint(0, model.config.vocab_size, (batch_size, context_length), device=device)
    inference_params = new_inference_params(model, batch_size, context_length + decode_length + 1, model.lm_head.weight.dtype)
    with torch.inference_mode():
        out = prefill(
            model,
            input_ids,
            inference_params,
            context_length,
            step_prefill=should_step_prefill(batch_size, context_length),
        )
        activate_mxfp8_state(model, inference_params, mode)
        next_ids = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
        refresher = MambaChunkRefresh(model, inference_params, refresh_interval) if is_refresh_mode(mode) else None
        cuda_sync()
        start = time.perf_counter()
        for i in range(decode_length):
            if refresher is not None and i % refresh_interval == 0:
                refresher.start()
            inference_params.seqlen_offset = context_length + i
            out = model(next_ids, inference_params=inference_params)
            next_ids = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
            if refresher is not None and (i + 1) % refresh_interval == 0:
                refresher.refresh()
        if refresher is not None:
            refresher.finish()
            refresher.clear()
        cuda_sync()
        elapsed = time.perf_counter() - start
    del inference_params
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return elapsed


def latency_benchmark(model, mode: str, batch_size: int, context_length: int, decode_length: int, refresh_interval: int, warmup: int, repeats: int):
    samples = []
    for i in range(warmup + repeats):
        elapsed = latency_once(model, mode, batch_size, context_length, decode_length, refresh_interval)
        if i >= warmup:
            samples.append(elapsed)
    decode_s = sum(samples) / len(samples)
    return {
        "metric": "latency",
        "mode": mode,
        "context_length": context_length,
        "batch_size": batch_size,
        "decode_length": decode_length,
        "decode_latency_s": decode_s,
        "decode_latency_per_token_s": decode_s / decode_length,
        "tokens_per_s": batch_size * decode_length / decode_s,
        "repeats": repeats,
    }


def memory_trace(model, modes: list[str], batch_size: int, context_length: int, decode_length: int, refresh_interval: int):
    rows = []
    for mode in modes:
        setup_mode(model, mode)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(model.lm_head.weight.device)
        device = model.lm_head.weight.device
        input_ids = torch.randint(0, model.config.vocab_size, (batch_size, context_length), device=device)
        inference_params = new_inference_params(model, batch_size, context_length + decode_length + 1, model.lm_head.weight.dtype)
        trace = []
        refresh_events = []
        with torch.inference_mode():
            out = prefill(
                model,
                input_ids,
                inference_params,
                context_length,
                step_prefill=should_step_prefill(batch_size, context_length),
            )
            activate_mxfp8_state(model, inference_params, mode)
            cuda_sync()
            baseline = torch.cuda.memory_allocated(device)
            prefill_peak = torch.cuda.max_memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
            trace.append({"token": 0, "allocated_mib": mib(baseline), "delta_mib": 0.0, "event": "after_prefill"})
            next_ids = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
            refresher = MambaChunkRefresh(model, inference_params, refresh_interval) if is_refresh_mode(mode) else None
            for i in range(decode_length):
                if refresher is not None and i % refresh_interval == 0:
                    refresher.start()
                inference_params.seqlen_offset = context_length + i
                out = model(next_ids, inference_params=inference_params)
                next_ids = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
                if refresher is not None and (i + 1) % refresh_interval == 0:
                    before = torch.cuda.memory_allocated(device)
                    refresher.refresh()
                    cuda_sync()
                    after = torch.cuda.memory_allocated(device)
                    refresh_events.append({"token": i + 1, "before_delta_mib": mib(before - baseline), "after_delta_mib": mib(after - baseline)})
                cuda_sync()
                alloc = torch.cuda.memory_allocated(device)
                trace.append({"token": i + 1, "allocated_mib": mib(alloc), "delta_mib": mib(alloc - baseline), "event": "decode"})
            if refresher is not None:
                refresher.finish()
                refresher.clear()
        decode_peak = torch.cuda.max_memory_allocated(device)
        rows.append(
            {
                "metric": "memory_trace",
                "mode": mode,
                "context_length": context_length,
                "batch_size": batch_size,
                "decode_length": decode_length,
                "baseline_mib": mib(baseline),
                "prefill_peak_mib": mib(prefill_peak),
                "prefill_peak_delta_mib": mib(prefill_peak - baseline),
                "decode_peak_mib": mib(decode_peak),
                "decode_peak_delta_mib": mib(decode_peak - baseline),
                "peak_mib": mib(decode_peak),
                "peak_delta_mib": mib(decode_peak - baseline),
                "trace": trace,
                "refresh_events": refresh_events,
            }
        )
        del inference_params
        gc.collect()
        torch.cuda.empty_cache()
    return rows


def plot_trace(rows, output_png: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=160)
    for ax, field, ylabel in [
        (axes[0], "allocated_mib", "Allocated memory (MiB)"),
        (axes[1], "delta_mib", "Delta from after-prefill (MiB)"),
    ]:
        for row in rows:
            xs = [p["token"] for p in row["trace"]]
            ys = [p[field] for p in row["trace"]]
            ax.plot(xs, ys, label=row["mode"], linewidth=1.4)
            if row["mode"] == "mxfp8_refresh256":
                for ev in row["refresh_events"]:
                    ax.axvline(ev["token"], color="tab:red", alpha=0.18, linewidth=0.9)
        ax.set_xlabel("Decode token")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend()
    fig.suptitle(f"Mamba2-1.3B decode memory trace, ctx={rows[0]['context_length']}, bs={rows[0]['batch_size']}")
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/home/vrintern/tmp/models/mamba2-1.3b")
    parser.add_argument("--tokenizer-path", default="/home/vrintern/tmp/models/gpt-neox-tokenizer")
    parser.add_argument("--dataset-path", default="/home/vrintern/tmp/datasets/wikitext")
    parser.add_argument("--output", default="results/mamba2_1_3b_quant_refresh_summary.json")
    parser.add_argument("--output-dir", default="results/mamba2_1_3b_quant_refresh")
    parser.add_argument("--modes", default="normal,mxfp8_fused,mxfp8_refresh256")
    parser.add_argument("--metrics", default="ppl,latency,memory")
    parser.add_argument("--context-lengths", default="2048,8192")
    parser.add_argument("--batch-sizes", default="1,32")
    parser.add_argument("--memory-batch-size", type=int, default=32)
    parser.add_argument("--decode-length", type=int, default=1000)
    parser.add_argument("--refresh-interval", type=int, default=256)
    parser.add_argument("--ppl-windows", type=int, default=2)
    parser.add_argument("--latency-warmup", type=int, default=1)
    parser.add_argument("--latency-repeats", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()

    torch.cuda.set_device(torch.device(args.device))
    model, tokenizer = load_model(args.model_path, args.tokenizer_path, args.device, args.dtype)
    modes = parse_csv(args.modes)
    metrics = set(parse_csv(args.metrics))
    rows = []
    output_dir = Path(args.output_dir)
    texts = None
    if "ppl" in metrics:
        texts = load_wikitext_texts(args.dataset_path, split="test")
    for ctx in parse_ints(args.context_lengths):
        if "ppl" in metrics:
            windows = token_windows(tokenizer, texts, ctx + args.decode_length + 1, args.ppl_windows)
            for mode in modes:
                row = decode_ppl(model, windows, mode, ctx, args.decode_length, args.refresh_interval)
                row["ppl_windows"] = len(windows)
                rows.append(row)
                print(json.dumps({k: v for k, v in row.items() if k != "trace"}), flush=True)
        if "latency" in metrics:
            for bs in parse_ints(args.batch_sizes):
                for mode in modes:
                    row = latency_benchmark(model, mode, bs, ctx, args.decode_length, args.refresh_interval, args.latency_warmup, args.latency_repeats)
                    rows.append(row)
                    print(json.dumps(row), flush=True)
        if "memory" in metrics:
            trace_rows = memory_trace(model, modes, args.memory_batch_size, ctx, args.decode_length, args.refresh_interval)
            rows.extend(trace_rows)
            trace_json = output_dir / f"memory_trace_ctx{ctx}_bs{args.memory_batch_size}.json"
            trace_png = output_dir / f"memory_trace_ctx{ctx}_bs{args.memory_batch_size}.png"
            trace_json.parent.mkdir(parents=True, exist_ok=True)
            trace_json.write_text(json.dumps({"results": trace_rows}, indent=2), encoding="utf-8")
            plot_trace(trace_rows, trace_png)
            for row in trace_rows:
                print(json.dumps({k: v for k, v in row.items() if k not in {"trace", "refresh_events"}}), flush=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps({"results": rows}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
