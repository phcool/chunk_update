# Nemotron Hybrid Cache Quantization Experiments

This repo benchmarks cache/state quantization for `nvidia/Nemotron-Flash-1B`.

The six experiment configurations are:

- KV cache: `normal`, `fp8`, `int4`
- Mamba/recurrent state: `normal`, `mxfp8`

The launcher runs one worker per configuration and assigns workers across the
available GPUs.

## Run

```bash
cd /home/vrintern/tmp/chunk_update
python run_experiments.py \
  --model-path /home/vrintern/tmp/models/Nemotron-Flash-1B \
  --dataset-path /home/vrintern/tmp/datasets/wikitext \
  --gpus 0,1,2,3,4,5,6,7 \
  --context-lengths 2048,8192 \
  --batch-sizes 1,32 \
  --decode-length 256 \
  --result-json results/all_results.json
```

For a quick smoke test:

```bash
python run_experiments.py \
  --metrics latency \
  --context-lengths 2048 \
  --batch-sizes 1 \
  --decode-length 8 \
  --latency-warmup 0 \
  --latency-repeats 1 \
  --dry-run
```

## Notes

- The model is loaded with `attn_implementation_new="flash_attention_2"` so
  attention K/V goes through a replaceable `past_key_values` cache.
- PPL uses cached teacher forcing: each next token is scored while feeding the
  ground-truth previous token. This is slower than full-sequence teacher
  forcing, but it is the path where KV cache and recurrent state quantization
  actually affect outputs.
- MXFP8 state quantization is store-only: recurrent state math runs in the model
  dtype, states are quantized after the forward call, then dequantized before
  the next token. The implementation uses block scaling with stochastic
  rounding and a default group size of 32.
- The final merged result is a JSON file with every PPL and latency row.
