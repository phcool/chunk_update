# Hybrid KV Cache / Mamba State Quantization Handoff

This folder collects the files needed to continue the Nemotron-Flash-1B hybrid cache quantization work.

## Scope

- Model: `/home/vrintern/tmp/models/Nemotron-Flash-1B`
- Dataset: `/home/vrintern/tmp/datasets/wikitext`
- Architecture: hybrid attention + Mamba2
- KV cache modes: `normal`, `fp8`, `int4`
- Mamba state modes: `normal`, `mxfp8_fused`
- PPL: WikiText-103 teacher forcing, contexts `2048`, `8192`
- Latency: decode-focused, batch sizes `1`, `32`, decode length `256`

## Code

- `code/run_worker.py`: single worker entrypoint.
- `code/run_experiments.py`: 6-config multi-GPU launcher.
- `code/run_mamba_state_only.py`: Mamba state only launcher.
- `code/run_mxfp8_profile.py`: model-level MXFP8 fused decode profile.
- `code/run_state_kernel_microprofile.py`: state update kernel microprofile/ablation.
- `code/hybrid_quant/quant.py`: FP8, INT4, MXFP8 tensor quantizers.
- `code/hybrid_quant/cache.py`: attention cache and recurrent cache quant helpers.
- `code/hybrid_quant/mxfp8_fused.py`: fused MXFP8 E4M3 + E8M0 Mamba state update kernel.
- `code/hybrid_quant/metrics.py`: PPL and latency implementations.
- `code/hybrid_quant/modeling.py`: Nemotron model loading and runtime cache setup.

## Results

- `results/full_kv_mamba_results.json`: full 6-config PPL + latency run, 36/36 tasks completed.
- `results/mamba_state_only_results.json`: KV normal, Mamba normal vs MXFP8 fused.
- `results/mamba_state_decode_summary.json`: decode-focused summary for Mamba state only.
- `results/latency_ctx2048_bs32_64_128_256_normal_vs_mxfp8_fused_summary.json`: batch sweep for Mamba normal vs fused MXFP8.
- `results/profile_mxfp8_fused_ctx2048_bs32_64_summary.json`: model-level synced profile for fused MXFP8 decode.
- `results/state_kernel_microprofile_bs32_64_summary.json`: compact kernel ablation summary.
- `results/state_kernel_microprofile_bs32_64_with_normal_breakdown.json`: full kernel ablation result.

## Key Notes

- `mxfp8_fused` stores Mamba SSM state as FP8 E4M3 payload plus uint8 E8M0 scale metadata, group size 32.
- The fused kernel performs dequant/read, state update/output, and requant/writeback in one Triton kernel.
- Nsight Compute hardware counters were unavailable on this machine due to `ERR_NVGPUCTRPERM`.
- Kernel internal percentages are ablation/difference estimates, not NCU source-line timings.
- KV FP8/INT4 cache quantization is currently simulated in cache wrappers, not a fused attention kernel.

## Latest Full Run PPL Highlights

```text
kv=normal mamba=normal       ctx=2048 ppl=16.2411
kv=normal mamba=normal       ctx=8192 ppl=20.4408
kv=normal mamba=mxfp8_fused  ctx=2048 ppl=17.7273
kv=normal mamba=mxfp8_fused  ctx=8192 ppl=23.9227

kv=fp8    mamba=normal       ctx=2048 ppl=16.2655
kv=fp8    mamba=normal       ctx=8192 ppl=20.5185
kv=fp8    mamba=mxfp8_fused  ctx=2048 ppl=17.7267
kv=fp8    mamba=mxfp8_fused  ctx=8192 ppl=24.0073

kv=int4   mamba=normal       ctx=2048 ppl=16.2186
kv=int4   mamba=normal       ctx=8192 ppl=20.9434
kv=int4   mamba=mxfp8_fused  ctx=2048 ppl=17.7534
kv=int4   mamba=mxfp8_fused  ctx=8192 ppl=24.8241
```

## Latest Decode Latency Highlights

```text
kv=normal mamba=normal       ctx=2048 bs=32 decode=9.286 ms/token
kv=normal mamba=mxfp8_fused  ctx=2048 bs=32 decode=9.314 ms/token

kv=fp8    mamba=normal       ctx=2048 bs=32 decode=9.499 ms/token
kv=fp8    mamba=mxfp8_fused  ctx=2048 bs=32 decode=13.699 ms/token

kv=int4   mamba=normal       ctx=2048 bs=32 decode=19.110 ms/token
kv=int4   mamba=mxfp8_fused  ctx=2048 bs=32 decode=20.110 ms/token
```
