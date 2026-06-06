from __future__ import annotations

import math
import time
from contextlib import contextmanager

import torch
import torch.nn.functional as F

from .cache import dequantize_recurrent_caches, quantize_recurrent_caches
from .modeling import new_runtime_caches, prepare_attention_kv_caches, reset_model_sequence_state
from .mxfp8_fused import (
    enable_mxfp8_fused_state_cache,
    initialize_mxfp8_fused_state_cache,
    quantize_current_mamba_states,
)


def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@contextmanager
def inference_mode():
    with torch.inference_mode():
        yield


def cached_teacher_forcing_ppl(
    model,
    token_segments,
    kv_mode: str,
    mamba_mode: str,
    context_length: int,
    mamba_group_size: int = 32,
):
    fused_mxfp8 = mamba_mode == "mxfp8_fused"
    if fused_mxfp8:
        enable_mxfp8_fused_state_cache(model, group_size=mamba_group_size)
    total_nll = 0.0
    total_tokens = 0
    start = time.perf_counter()
    with inference_mode():
        for seg in token_segments:
            ids = seg.to(model.device).unsqueeze(0)
            reset_model_sequence_state(model)
            if fused_mxfp8:
                initialize_mxfp8_fused_state_cache(model)
            caches = new_runtime_caches(model, batch_size=1, max_seqlen=context_length + 1, kv_mode=kv_mode)
            prepare_attention_kv_caches(model, batch_size=1, max_seqlen=context_length + 1)
            mem = int(getattr(model.config, "num_memory_tokens", 0) or 0)
            for t in range(ids.shape[1] - 1):
                caches.mamba_inference_params.seqlen_offset = t
                if not fused_mxfp8:
                    dequantize_recurrent_caches(caches.mamba_inference_params, caches.fla_past_key_values)
                pos = None if t == 0 else torch.tensor([[mem + t]], device=model.device, dtype=torch.long)
                out = model(
                    input_ids=ids[:, t : t + 1],
                    position_ids=pos,
                    past_key_values=caches.past_key_values,
                    fla_past_key_values=caches.fla_past_key_values,
                    mamba_inference_params=caches.mamba_inference_params,
                    use_cache=True,
                    calc_logits_for_entire_prompt=False,
                )
                logits = out.logits[:, -1, :]
                target = ids[:, t + 1]
                loss = F.cross_entropy(logits, target, reduction="sum")
                total_nll += float(loss.item())
                total_tokens += 1
                if fused_mxfp8:
                    if t == 0:
                        quantize_current_mamba_states(model, caches.mamba_inference_params)
                else:
                    quantize_recurrent_caches(
                        caches.mamba_inference_params,
                        caches.fla_past_key_values,
                        mode=mamba_mode,
                        group_size=mamba_group_size,
                    )
    cuda_sync()
    elapsed = time.perf_counter() - start
    return {
        "ppl": math.exp(total_nll / max(total_tokens, 1)),
        "nll": total_nll,
        "tokens": total_tokens,
        "elapsed_s": elapsed,
        "tokens_per_s": total_tokens / elapsed if elapsed > 0 else None,
    }


def latency_benchmark(
    model,
    kv_mode: str,
    mamba_mode: str,
    batch_size: int,
    prefill_length: int,
    decode_length: int,
    warmup: int = 1,
    repeats: int = 3,
    mamba_group_size: int = 32,
    prefill_chunk_size: int = 0,
):
    fused_mxfp8 = mamba_mode == "mxfp8_fused"
    if fused_mxfp8:
        enable_mxfp8_fused_state_cache(model, group_size=mamba_group_size)
    vocab = model.config.vocab_size
    mem = int(getattr(model.config, "num_memory_tokens", 0) or 0)

    def run_once(measure: bool):
        reset_model_sequence_state(model)
        if fused_mxfp8:
            initialize_mxfp8_fused_state_cache(model)
        prepare_attention_kv_caches(
            model,
            batch_size=batch_size,
            max_seqlen=prefill_length + decode_length + 1,
        )
        caches = new_runtime_caches(
            model,
            batch_size=batch_size,
            max_seqlen=prefill_length + decode_length + 1,
            kv_mode=kv_mode,
        )
        input_ids = torch.randint(0, vocab, (batch_size, prefill_length), device=model.device)
        cuda_sync()
        prefill_start = time.perf_counter()
        with inference_mode():
            chunk_size = prefill_chunk_size if prefill_chunk_size and prefill_chunk_size > 0 else prefill_length
            out = None
            for start in range(0, prefill_length, chunk_size):
                end = min(start + chunk_size, prefill_length)
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
            if fused_mxfp8:
                quantize_current_mamba_states(model, caches.mamba_inference_params)
            else:
                quantize_recurrent_caches(
                    caches.mamba_inference_params,
                    caches.fla_past_key_values,
                    mode=mamba_mode,
                    group_size=mamba_group_size,
                )
        cuda_sync()
        prefill_s = time.perf_counter() - prefill_start

        next_ids = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
        cuda_sync()
        decode_start = time.perf_counter()
        with inference_mode():
            for i in range(decode_length):
                caches.mamba_inference_params.seqlen_offset = prefill_length + i
                if not fused_mxfp8:
                    dequantize_recurrent_caches(caches.mamba_inference_params, caches.fla_past_key_values)
                pos = torch.full((batch_size, 1), mem + prefill_length + i, device=model.device, dtype=torch.long)
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
                if not fused_mxfp8:
                    quantize_recurrent_caches(
                        caches.mamba_inference_params,
                        caches.fla_past_key_values,
                        mode=mamba_mode,
                        group_size=mamba_group_size,
                    )
        cuda_sync()
        decode_s = time.perf_counter() - decode_start
        return prefill_s, decode_s

    samples = []
    for i in range(warmup + repeats):
        prefill_s, decode_s = run_once(measure=i >= warmup)
        if i >= warmup:
            samples.append((prefill_s, decode_s))
    prefill = sum(x[0] for x in samples) / len(samples)
    decode = sum(x[1] for x in samples) / len(samples)
    return {
        "prefill_latency_s": prefill,
        "decode_latency_s": decode,
        "decode_latency_per_token_s": decode / decode_length,
        "tokens_per_s": batch_size * decode_length / decode,
        "decode_length": decode_length,
        "repeats": repeats,
    }
