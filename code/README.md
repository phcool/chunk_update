Experiment code is grouped by model family.

- `shared/`: quantization utilities, cache wrappers, and fused MXFP8 kernels reused by both experiment families.
- `hybrid/`: Nemotron-Flash-1B hybrid attention + Mamba/Mamba2 experiment runners.
- `mamba2/`: pure Mamba2-1.3B experiment runners.

The original root-level scripts are intentionally kept in place so existing command lines still work.
