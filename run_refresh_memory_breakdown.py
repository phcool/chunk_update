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

import torch

from hybrid_quant.modeling import load_nemotron
from hybrid_quant.mxfp8_fused import (
    enable_mxfp8_fused_state_cache,
    quantize_current_mamba_states,
    quantize_state_into_cache,
    dequantize_state_cache,
)
from run_decode_ppl_latency_mamba_refresh import (
    MambaChunkRefresh,
    _chunk_refresh_one_layer,
    _get_live_mxfp8_cache,
    _mamba_layers,
    cuda_sync,
    run_prefill,
    setup_caches,
)


def mib(x: int | float) -> float:
    return float(x) / 1024.0 / 1024.0


def tensor_bytes(x) -> int:
    if isinstance(x, torch.Tensor):
        return x.numel() * x.element_size()
    return 0


def cache_bytes(cache) -> int:
    return tensor_bytes(cache.q_state) + tensor_bytes(cache.scale_e8m0)


def snapshot_bytes(refresher: MambaChunkRefresh) -> dict:
    conv = 0
    state = 0
    for conv_state, state_cache in refresher.snapshots.values():
        conv += tensor_bytes(conv_state)
        state += cache_bytes(state_cache)
    return {"snapshot_conv_mib": mib(conv), "snapshot_mxfp8_state_mib": mib(state), "snapshot_total_mib": mib(conv + state)}


def record_bytes(refresher: MambaChunkRefresh) -> dict:
    total = 0
    by_layer = {}
    for layer_idx, hidden_list in refresher.records.items():
        b = sum(tensor_bytes(x) for x in hidden_list)
        by_layer[str(layer_idx)] = mib(b)
        total += b
    return {"recorded_hidden_mib": mib(total), "recorded_layers": len(by_layer), "recorded_hidden_by_layer_mib": by_layer}


def measure(args):
    torch.cuda.set_device(torch.device(args.device))
    model, _ = load_nemotron(args.model_path, device=args.device, dtype=args.dtype, attn_implementation=args.attn_implementation)
    enable_mxfp8_fused_state_cache(model, group_size=32)
    gc.collect()
    torch.cuda.empty_cache()

    vocab = model.config.vocab_size
    mem = int(getattr(model.config, "num_memory_tokens", 0) or 0)
    input_ids = torch.randint(0, vocab, (args.batch_size, args.context_length), device=model.device)
    caches = setup_caches(model, args.batch_size, args.context_length + args.decode_tokens + 1, "mxfp8_refresh256")
    with torch.inference_mode():
        out = run_prefill(model, caches, input_ids, args.context_length, args.prefill_chunk_size)
        quantize_current_mamba_states(model, caches.mamba_inference_params)
        cuda_sync()
        baseline = torch.cuda.memory_allocated(model.device)

        refresher = MambaChunkRefresh(model, caches.mamba_inference_params, args.decode_tokens)
        refresher.start()
        cuda_sync()
        after_start = torch.cuda.memory_allocated(model.device)

        next_ids = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
        for i in range(args.decode_tokens):
            caches.mamba_inference_params.seqlen_offset = args.context_length + i
            pos = torch.full((args.batch_size, 1), mem + args.context_length + i, device=model.device, dtype=torch.long)
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
        cuda_sync()
        before_refresh = torch.cuda.memory_allocated(model.device)
        before_refresh_peak = torch.cuda.max_memory_allocated(model.device)
        snapshot_stats = snapshot_bytes(refresher)
        record_stats = record_bytes(refresher)

        layer_rows = []
        torch.cuda.reset_peak_memory_stats(model.device)
        refresh_start = torch.cuda.memory_allocated(model.device)
        for mamba in _mamba_layers(model):
            hidden_list = refresher.records.get(mamba.layer_idx)
            if not hidden_list:
                continue
            live_conv, live_ssm = caches.mamba_inference_params.key_value_memory_dict[mamba.layer_idx][:2]
            conv_start, state_cache = refresher.snapshots[mamba.layer_idx]
            layer_base = torch.cuda.memory_allocated(model.device)
            hidden = torch.cat(hidden_list, dim=1)
            cuda_sync()
            after_cat = torch.cuda.memory_allocated(model.device)
            state_start = dequantize_state_cache(state_cache, hidden.dtype)
            cuda_sync()
            after_dequant = torch.cuda.memory_allocated(model.device)
            final_conv, final_state = _chunk_refresh_one_layer(mamba, hidden, conv_start, state_start)
            cuda_sync()
            after_chunk = torch.cuda.memory_allocated(model.device)
            live_conv.copy_(final_conv)
            live_cache = _get_live_mxfp8_cache(mamba, live_ssm)
            quantize_state_into_cache(final_state, live_cache)
            cuda_sync()
            after_quant = torch.cuda.memory_allocated(model.device)
            layer_peak = torch.cuda.max_memory_allocated(model.device)
            layer_rows.append(
                {
                    "layer_idx": mamba.layer_idx,
                    "base_delta_mib": mib(layer_base - refresh_start),
                    "cat_delta_mib": mib(after_cat - layer_base),
                    "dequant_delta_mib": mib(after_dequant - after_cat),
                    "chunk_delta_mib": mib(after_chunk - after_dequant),
                    "quant_delta_mib": mib(after_quant - after_chunk),
                    "layer_peak_delta_from_refresh_start_mib": mib(layer_peak - refresh_start),
                    "hidden_chunk_mib": mib(tensor_bytes(hidden)),
                    "bf16_state_mib": mib(tensor_bytes(state_start)),
                    "final_state_mib": mib(tensor_bytes(final_state)),
                    "final_conv_mib": mib(tensor_bytes(final_conv)),
                }
            )
            del hidden, state_start, final_conv, final_state

        cuda_sync()
        after_refresh = torch.cuda.memory_allocated(model.device)
        refresh_peak = torch.cuda.max_memory_allocated(model.device)
        refresher.finish()
        refresher.clear()
        gc.collect()
        torch.cuda.empty_cache()
        cuda_sync()
        after_cleanup = torch.cuda.memory_allocated(model.device)

    payload = {
        "context_length": args.context_length,
        "batch_size": args.batch_size,
        "decode_tokens_before_refresh": args.decode_tokens,
        "baseline_after_prefill_mib": mib(baseline),
        "after_refresh_start_mib": mib(after_start),
        "after_refresh_start_delta_mib": mib(after_start - baseline),
        "before_refresh_mib": mib(before_refresh),
        "before_refresh_delta_mib": mib(before_refresh - baseline),
        "before_refresh_peak_mib": mib(before_refresh_peak),
        "before_refresh_peak_delta_mib": mib(before_refresh_peak - baseline),
        "after_refresh_mib": mib(after_refresh),
        "after_refresh_delta_mib": mib(after_refresh - baseline),
        "after_cleanup_mib": mib(after_cleanup),
        "after_cleanup_delta_mib": mib(after_cleanup - baseline),
        "refresh_peak_mib": mib(refresh_peak),
        "refresh_peak_delta_mib": mib(refresh_peak - baseline),
        **snapshot_stats,
        **record_stats,
        "layers": layer_rows,
    }
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/home/vrintern/tmp/models/Nemotron-Flash-1B")
    parser.add_argument("--output", default="results/refresh_memory_breakdown_ctx2048_bs32.json")
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--decode-tokens", type=int, default=256)
    parser.add_argument("--prefill-chunk-size", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    args = parser.parse_args()
    payload = measure(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items() if k != "layers"}, indent=2))
    print("layers", len(payload["layers"]))


if __name__ == "__main__":
    main()
