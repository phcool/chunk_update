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
from typing import Any

os.environ["HF_HOME"] = "/home/vrintern/tmp/.hf-cache"
os.environ["HF_XET_CACHE"] = "/home/vrintern/tmp/.hf-xet-cache"
os.environ["HF_MODULES_CACHE"] = "/home/vrintern/tmp/.hf-modules-cache"
os.environ["HF_DATASETS_CACHE"] = "/home/vrintern/tmp/.hf-cache/datasets"
os.environ["XDG_CACHE_HOME"] = "/home/vrintern/tmp/.cache"

import torch
import torch.nn.functional as F
from einops import rearrange

from causal_conv1d import causal_conv1d_fn
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

from hybrid_quant.data import load_wikitext_texts
from hybrid_quant.modeling import (
    load_nemotron,
    new_runtime_caches,
    prepare_attention_kv_caches,
    reset_model_sequence_state,
)
from hybrid_quant.mxfp8_fused import (
    MXFP8StateCache,
    clone_state_cache,
    dequantize_state_cache,
    enable_mxfp8_fused_state_cache,
    initialize_mxfp8_fused_state_cache,
    quantize_current_mamba_states,
    quantize_state_into_cache,
)


MODES = {"normal", "mxfp8_fused", "mxfp8_refresh256"}


def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def parse_ints(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def write_jsonl(path: Path, row: dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def token_windows(tokenizer, texts, total_length: int, max_windows: int):
    ids = tokenizer("\n\n".join(texts), add_special_tokens=False, return_tensors="pt").input_ids[0]
    pos = 0
    out = []
    while pos + total_length <= ids.numel() and len(out) < max_windows:
        out.append(ids[pos : pos + total_length].clone())
        pos += total_length - 1
    return out


def _mamba_layers(model):
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        layers = getattr(getattr(model, "backbone", None), "layers", [])
    for idx, layer in enumerate(layers):
        mamba = getattr(layer, "mamba", None)
        if mamba is None:
            mamba = getattr(layer, "mixer", None)
        if mamba is not None:
            if not hasattr(mamba, "layer_idx"):
                mamba.layer_idx = idx
            yield mamba


def _get_live_mxfp8_cache(mamba, ssm_state):
    if isinstance(ssm_state, MXFP8StateCache):
        return ssm_state
    return mamba._mxfp8_fused_caches[id(ssm_state)]


def _conv1d_with_initial_state(mamba, xbc, conv_state):
    # conv_state stores the last d_conv raw xBC vectors in chronological order.
    xbc_t = rearrange(xbc, "b l d -> b d l")
    full = torch.cat([conv_state, xbc_t], dim=-1)
    y = causal_conv1d_fn(
        full,
        rearrange(mamba.conv1d.weight, "d 1 w -> d w"),
        bias=mamba.conv1d.bias,
        activation=mamba.activation,
    ).transpose(1, 2)
    final_conv = full[:, :, -mamba.d_conv :].contiguous()
    return y[:, -xbc.shape[1] :, :].contiguous(), final_conv


def _chunk_refresh_one_layer(mamba, hidden_states, conv_start, state_start):
    dtype = hidden_states.dtype
    zxbcdt = mamba.in_proj(hidden_states)
    d_mlp = (zxbcdt.shape[-1] - 2 * mamba.d_ssm - 2 * mamba.ngroups * mamba.d_state - mamba.nheads) // 2
    z0, x0, z, xbc, dt = torch.split(
        zxbcdt,
        [d_mlp, d_mlp, mamba.d_ssm, mamba.d_ssm + 2 * mamba.ngroups * mamba.d_state, mamba.nheads],
        dim=-1,
    )
    del z0, x0
    xbc, final_conv = _conv1d_with_initial_state(mamba, xbc, conv_start)
    x, b_mat, c_mat = torch.split(
        xbc,
        [mamba.d_ssm, mamba.ngroups * mamba.d_state, mamba.ngroups * mamba.d_state],
        dim=-1,
    )
    a_mat = -torch.exp(mamba.A_log.float())
    dt_limit_kwargs = {} if mamba.dt_limit == (0.0, float("inf")) else dict(dt_limit=mamba.dt_limit)
    y, final_state = mamba_chunk_scan_combined(
        rearrange(x, "b l (h p) -> b l h p", p=mamba.headdim),
        dt,
        a_mat,
        rearrange(b_mat, "b l (g n) -> b l g n", g=mamba.ngroups),
        rearrange(c_mat, "b l (g n) -> b l g n", g=mamba.ngroups),
        chunk_size=mamba.chunk_size,
        D=rearrange(mamba.D, "(h p) -> h p", p=mamba.headdim) if mamba.D_has_hdim else mamba.D,
        z=rearrange(z, "b l (h p) -> b l h p", p=mamba.headdim) if not mamba.rmsnorm else None,
        dt_bias=mamba.dt_bias,
        dt_softplus=True,
        initial_states=state_start,
        return_final_states=True,
        **dt_limit_kwargs,
    )
    del y
    return final_conv.to(dtype), final_state.to(dtype)


class MambaChunkRefresh:
    def __init__(self, model, inference_params, interval: int):
        self.model = model
        self.inference_params = inference_params
        self.interval = interval
        self.records: dict[int, list[torch.Tensor]] = {}
        self.snapshots: dict[int, tuple[torch.Tensor, MXFP8StateCache]] = {}

    def start(self):
        self.records = {}
        self.snapshots = {}
        for mamba in _mamba_layers(self.model):
            states = self.inference_params.key_value_memory_dict.get(mamba.layer_idx)
            if states is None:
                continue
            conv_state, ssm_state = states[:2]
            cache = _get_live_mxfp8_cache(mamba, ssm_state)
            self.snapshots[mamba.layer_idx] = (conv_state.clone(), clone_state_cache(cache))
            mamba._mxfp8_chunk_recorder = self.records

    def finish(self):
        for mamba in _mamba_layers(self.model):
            if hasattr(mamba, "_mxfp8_chunk_recorder"):
                mamba._mxfp8_chunk_recorder = None

    def clear(self):
        self.records = {}
        self.snapshots = {}

    def refresh(self):
        try:
            for mamba in _mamba_layers(self.model):
                hidden_list = self.records.get(mamba.layer_idx)
                if not hidden_list:
                    continue
                conv_start, state_cache = self.snapshots[mamba.layer_idx]
                hidden = torch.cat(hidden_list, dim=1)
                state_start = dequantize_state_cache(state_cache, hidden.dtype)
                final_conv, final_state = _chunk_refresh_one_layer(mamba, hidden, conv_start, state_start)
                live_conv, live_ssm = self.inference_params.key_value_memory_dict[mamba.layer_idx][:2]
                live_conv.copy_(final_conv)
                live_cache = _get_live_mxfp8_cache(mamba, live_ssm)
                quantize_state_into_cache(final_state, live_cache)
        finally:
            self.finish()
            self.clear()


def setup_caches(model, batch_size: int, max_seqlen: int, mode: str):
    reset_model_sequence_state(model)
    prepare_attention_kv_caches(model, batch_size=batch_size, max_seqlen=max_seqlen)
    caches = new_runtime_caches(model, batch_size=batch_size, max_seqlen=max_seqlen, kv_mode="normal")
    if mode != "normal":
        initialize_mxfp8_fused_state_cache(model)
    return caches


def run_prefill(model, caches, input_ids, context_length: int, chunk_size: int):
    mem = int(getattr(model.config, "num_memory_tokens", 0) or 0)
    out = None
    for start in range(0, context_length, chunk_size):
        end = min(start + chunk_size, context_length)
        caches.mamba_inference_params.seqlen_offset = start
        pos = None
        if start > 0:
            pos_values = torch.arange(mem + start, mem + end, device=model.device, dtype=torch.long)
            pos = pos_values.unsqueeze(0).expand(input_ids.shape[0], -1)
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


def decode_ppl(model, windows, mode: str, context_length: int, decode_length: int, prefill_chunk_size: int, refresh_interval: int):
    total_nll = 0.0
    total_tokens = 0
    start_time = time.perf_counter()
    mem = int(getattr(model.config, "num_memory_tokens", 0) or 0)
    with torch.inference_mode():
        for ids in windows:
            ids = ids.to(model.device).unsqueeze(0)
            caches = setup_caches(model, 1, context_length + decode_length + 1, mode)
            run_prefill(model, caches, ids[:, :context_length], context_length, prefill_chunk_size)
            refresher = None
            if mode != "normal":
                quantize_current_mamba_states(model, caches.mamba_inference_params)
            if mode == "mxfp8_refresh256":
                refresher = MambaChunkRefresh(model, caches.mamba_inference_params, refresh_interval)
            for i in range(decode_length):
                if refresher is not None and i % refresh_interval == 0:
                    refresher.start()
                caches.mamba_inference_params.seqlen_offset = context_length + i
                pos = torch.full((1, 1), mem + context_length + i, device=model.device, dtype=torch.long)
                out = model(
                    input_ids=ids[:, context_length + i : context_length + i + 1],
                    position_ids=pos,
                    past_key_values=caches.past_key_values,
                    fla_past_key_values=caches.fla_past_key_values,
                    mamba_inference_params=caches.mamba_inference_params,
                    use_cache=True,
                    calc_logits_for_entire_prompt=False,
                )
                loss = F.cross_entropy(out.logits[:, -1, :], ids[:, context_length + i + 1], reduction="sum")
                total_nll += float(loss.item())
                total_tokens += 1
                if refresher is not None and (i + 1) % refresh_interval == 0:
                    refresher.refresh()
            if refresher is not None:
                refresher.finish()
            del caches
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    cuda_sync()
    elapsed = time.perf_counter() - start_time
    return {
        "metric": "decode_ppl",
        "ppl": math.exp(total_nll / max(total_tokens, 1)),
        "nll": total_nll,
        "tokens": total_tokens,
        "elapsed_s": elapsed,
        "tokens_per_s": total_tokens / elapsed if elapsed > 0 else None,
    }


def latency_once(model, mode: str, batch_size: int, context_length: int, decode_length: int, prefill_chunk_size: int, refresh_interval: int):
    vocab = model.config.vocab_size
    mem = int(getattr(model.config, "num_memory_tokens", 0) or 0)
    input_ids = torch.randint(0, vocab, (batch_size, context_length), device=model.device)
    caches = setup_caches(model, batch_size, context_length + decode_length + 1, mode)
    with torch.inference_mode():
        out = run_prefill(model, caches, input_ids, context_length, prefill_chunk_size)
        if mode != "normal":
            quantize_current_mamba_states(model, caches.mamba_inference_params)
        next_ids = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
        refresher = MambaChunkRefresh(model, caches.mamba_inference_params, refresh_interval) if mode == "mxfp8_refresh256" else None
        cuda_sync()
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
    del caches
    return elapsed


def latency_benchmark(model, mode: str, batch_size: int, context_length: int, decode_length: int, warmup: int, repeats: int, prefill_chunk_size: int, refresh_interval: int):
    samples = []
    for i in range(warmup + repeats):
        elapsed = latency_once(model, mode, batch_size, context_length, decode_length, prefill_chunk_size, refresh_interval)
        if i >= warmup:
            samples.append(elapsed)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    decode_s = sum(samples) / len(samples)
    return {
        "metric": "latency",
        "decode_latency_s": decode_s,
        "decode_latency_per_token_s": decode_s / decode_length,
        "tokens_per_s": batch_size * decode_length / decode_s,
        "decode_length": decode_length,
        "repeats": repeats,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/home/vrintern/tmp/models/Nemotron-Flash-1B")
    parser.add_argument("--dataset-path", default="/home/vrintern/tmp/datasets/wikitext")
    parser.add_argument("--output", default="results/decode_ppl_latency_mamba_refresh.jsonl")
    parser.add_argument("--mode", required=True, choices=sorted(MODES))
    parser.add_argument("--metrics", default="ppl,latency")
    parser.add_argument("--context-lengths", default="2048,8192")
    parser.add_argument("--batch-sizes", default="1,32")
    parser.add_argument("--decode-length", type=int, default=1000)
    parser.add_argument("--refresh-interval", type=int, default=256)
    parser.add_argument("--ppl-windows", type=int, default=2)
    parser.add_argument("--prefill-chunk-size", type=int, default=256)
    parser.add_argument("--latency-warmup", type=int, default=1)
    parser.add_argument("--latency-repeats", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    model, tokenizer = load_nemotron(
        args.model_path,
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )
    if args.mode != "normal":
        enable_mxfp8_fused_state_cache(model, group_size=32)

    output = Path(args.output)
    metrics = {x.strip() for x in args.metrics.split(",") if x.strip()}
    texts = None
    if "ppl" in metrics:
        texts = load_wikitext_texts(args.dataset_path, split="test")

    base = {
        "mode": args.mode,
        "kv_mode": "normal",
        "decode_length": args.decode_length,
        "refresh_interval": args.refresh_interval if args.mode == "mxfp8_refresh256" else None,
        "device": args.device,
        "dtype": args.dtype,
    }
    for ctx in parse_ints(args.context_lengths):
        if "ppl" in metrics:
            windows = token_windows(tokenizer, texts, ctx + args.decode_length + 1, args.ppl_windows)
            row = decode_ppl(model, windows, args.mode, ctx, args.decode_length, args.prefill_chunk_size, args.refresh_interval)
            write_jsonl(output, {**base, "context_length": ctx, "ppl_windows": len(windows), **row})
        if "latency" in metrics:
            for bs in parse_ints(args.batch_sizes):
                row = latency_benchmark(
                    model,
                    args.mode,
                    bs,
                    ctx,
                    args.decode_length,
                    args.latency_warmup,
                    args.latency_repeats,
                    args.prefill_chunk_size,
                    args.refresh_interval,
                )
                write_jsonl(output, {**base, "context_length": ctx, "batch_size": bs, **row})


if __name__ == "__main__":
    main()
