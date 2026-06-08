from __future__ import annotations

import argparse
import gc
import json
import math
import os
import time
from pathlib import Path
import sys
from typing import Any

_CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_CODE_ROOT / "shared"))

os.environ["HF_HOME"] = "/home/vrintern/tmp/.hf-cache"
os.environ["HF_XET_CACHE"] = "/home/vrintern/tmp/.hf-xet-cache"
os.environ["HF_MODULES_CACHE"] = "/home/vrintern/tmp/.hf-modules-cache"
os.environ["HF_DATASETS_CACHE"] = "/home/vrintern/tmp/.hf-cache/datasets"
os.environ["XDG_CACHE_HOME"] = "/home/vrintern/tmp/.cache"

import torch
import torch.nn.functional as F

from hybrid_quant.cache import dequantize_obj, quantize_tensor_tree
from hybrid_quant.data import load_wikitext_texts
from hybrid_quant.modeling import (
    load_nemotron,
    new_runtime_caches,
    prepare_attention_kv_caches,
    reset_model_sequence_state,
)
from hybrid_quant.mxfp8_fused import (
    allocate_mxfp8_state_cache,
    enable_mxfp8_fused_state_cache,
    quantize_state_into_cache,
)
from hybrid_quant.quant import TensorQuantizer


BASELINE_LAYER = -1


def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def iter_mamba_layers(model):
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        layers = getattr(getattr(model, "backbone", None), "layers", [])
    for idx, layer in enumerate(layers):
        mamba = getattr(layer, "mamba", None)
        if mamba is None:
            mamba = getattr(layer, "mixer", None)
        if mamba is not None and hasattr(mamba, "step"):
            if not hasattr(mamba, "layer_idx"):
                mamba.layer_idx = idx
            yield idx, mamba


def token_windows(tokenizer, texts, total_length: int, max_windows: int):
    ids = tokenizer("\n\n".join(texts), add_special_tokens=False, return_tensors="pt").input_ids[0]
    pos = 0
    out = []
    while pos + total_length <= ids.numel() and len(out) < max_windows:
        out.append(ids[pos : pos + total_length].clone())
        # Overlap one token so every window still has a next-token label.
        pos += total_length - 1
    return out


def write_json(path: Path, payload: dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


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


def configure_single_quantized_layer(model, layer_index: int, group_size: int, quant_backend: str):
    if layer_index == BASELINE_LAYER or quant_backend == "mxfp8_sr":
        return
    enable_mxfp8_fused_state_cache(model, group_size=group_size)
    found = False
    for idx, mamba in iter_mamba_layers(model):
        enabled = idx == layer_index
        mamba._mxfp8_fused_enabled = enabled
        if enabled:
            found = True
            mamba._mxfp8_fused_group_size = group_size
            mamba._mxfp8_fused_caches.clear()
    if not found:
        raise ValueError(f"Layer index {layer_index} is not a Mamba layer in this model.")


def quantize_target_layer_state_fused(model, inference_params, layer_index: int, group_size: int):
    if layer_index == BASELINE_LAYER:
        return
    target = dict(iter_mamba_layers(model))[layer_index]
    states = inference_params.key_value_memory_dict.get(target.layer_idx)
    if states is None:
        raise RuntimeError(f"No Mamba state found for layer {layer_index} after prefill.")
    _, ssm_state = states[:2]
    cache = allocate_mxfp8_state_cache(ssm_state, group_size=group_size)
    quantize_state_into_cache(ssm_state, cache)
    target._mxfp8_fused_caches[id(ssm_state)] = cache


def dequantize_target_layer_state_sr(inference_params, layer_index: int):
    if layer_index == BASELINE_LAYER:
        return
    states = inference_params.key_value_memory_dict.get(layer_index)
    if states is not None:
        inference_params.key_value_memory_dict[layer_index] = dequantize_obj(states)


def quantize_target_layer_state_sr(inference_params, layer_index: int, quantizer: TensorQuantizer):
    if layer_index == BASELINE_LAYER:
        return
    states = inference_params.key_value_memory_dict.get(layer_index)
    if states is None:
        raise RuntimeError(f"No Mamba state found for layer {layer_index}.")
    inference_params.key_value_memory_dict[layer_index] = quantize_tensor_tree(states, quantizer)


def decode_ppl(
    model,
    windows,
    layer_index: int,
    quant_backend: str,
    context_length: int,
    decode_length: int,
    prefill_chunk_size: int,
    group_size: int,
):
    total_nll = 0.0
    total_tokens = 0
    start_time = time.perf_counter()
    mem = int(getattr(model.config, "num_memory_tokens", 0) or 0)
    sr_quantizer = TensorQuantizer("mxfp8", group_size=group_size, stochastic=True)
    with torch.inference_mode():
        for ids in windows:
            ids = ids.to(model.device).unsqueeze(0)
            reset_model_sequence_state(model)
            prepare_attention_kv_caches(model, batch_size=1, max_seqlen=context_length + decode_length + 1)
            caches = new_runtime_caches(
                model,
                batch_size=1,
                max_seqlen=context_length + decode_length + 1,
                kv_mode="normal",
            )
            run_prefill(model, caches, ids[:, :context_length], context_length, prefill_chunk_size)
            if quant_backend == "mxfp8_fused":
                quantize_target_layer_state_fused(model, caches.mamba_inference_params, layer_index, group_size)
            elif quant_backend == "mxfp8_sr":
                quantize_target_layer_state_sr(caches.mamba_inference_params, layer_index, sr_quantizer)
            for i in range(decode_length):
                caches.mamba_inference_params.seqlen_offset = context_length + i
                if quant_backend == "mxfp8_sr":
                    dequantize_target_layer_state_sr(caches.mamba_inference_params, layer_index)
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
                if quant_backend == "mxfp8_sr":
                    quantize_target_layer_state_sr(caches.mamba_inference_params, layer_index, sr_quantizer)
            del caches
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    cuda_sync()
    elapsed = time.perf_counter() - start_time
    return {
        "ppl": math.exp(total_nll / max(total_tokens, 1)),
        "nll": total_nll,
        "tokens": total_tokens,
        "elapsed_s": elapsed,
        "tokens_per_s": total_tokens / elapsed if elapsed > 0 else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/home/vrintern/tmp/models/Nemotron-Flash-1B")
    parser.add_argument("--dataset-path", default="/home/vrintern/tmp/datasets/wikitext")
    parser.add_argument("--output", required=True)
    parser.add_argument("--layer-index", type=int, default=BASELINE_LAYER)
    parser.add_argument("--quant-backend", choices=["mxfp8_sr", "mxfp8_fused"], default="mxfp8_sr")
    parser.add_argument("--list-layers", action="store_true")
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--decode-length", type=int, default=256)
    parser.add_argument("--ppl-windows", type=int, default=2)
    parser.add_argument("--prefill-chunk-size", type=int, default=256)
    parser.add_argument("--mamba-group-size", type=int, default=32)
    parser.add_argument("--ppl-split", default="test")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ready-file", default=None)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    seed = args.seed + max(args.layer_index, 0)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    base = {
        "worker_pid": os.getpid(),
        "device": args.device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "model_path": args.model_path,
        "dataset_path": args.dataset_path,
        "dtype": args.dtype,
        "attn_implementation": args.attn_implementation,
        "context_length": args.context_length,
        "decode_length": args.decode_length,
        "ppl_split": args.ppl_split,
        "mamba_group_size": args.mamba_group_size,
        "seed": seed,
    }

    model, tokenizer = load_nemotron(
        args.model_path,
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )
    layer_indices = [idx for idx, _ in iter_mamba_layers(model)]
    if args.list_layers:
        payload = {**base, "mamba_layers": layer_indices, "mamba_layer_count": len(layer_indices)}
        write_json(Path(args.output), payload)
        print(json.dumps(payload, indent=2), flush=True)
        return

    configure_single_quantized_layer(model, args.layer_index, args.mamba_group_size, args.quant_backend)
    if args.ready_file:
        write_json(Path(args.ready_file), {**base, "status": "ready", "layer_index": args.layer_index})

    texts = load_wikitext_texts(args.dataset_path, split=args.ppl_split)
    windows = token_windows(tokenizer, texts, args.context_length + args.decode_length + 1, args.ppl_windows)
    result = decode_ppl(
        model,
        windows,
        args.layer_index,
        args.quant_backend,
        args.context_length,
        args.decode_length,
        args.prefill_chunk_size,
        args.mamba_group_size,
    )
    row = {
        **base,
        "metric": "decode_ppl",
        "layer_index": args.layer_index,
        "quantized": args.layer_index != BASELINE_LAYER,
        "quantization": "none" if args.layer_index == BASELINE_LAYER else f"{args.quant_backend}_state_only",
        "ppl_windows": len(windows),
        **result,
    }
    write_json(Path(args.output), row)
    print(json.dumps(row, indent=2), flush=True)


if __name__ == "__main__":
    main()
