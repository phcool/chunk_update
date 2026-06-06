Pure Mamba2-1.3B experiment scripts.

Main entry points:

- `run_mamba2_state_quant_refresh.py`: PPL, decode latency, and memory trace for normal / MXFP8 / refresh256 modes.
- `run_mamba2_state_quant_refresh_launcher.py`: multi-GPU launcher for the pure Mamba2 experiment.

The decode-native memory mode uses `mxfp8_fused_native` and `mxfp8_refresh256_native`, replacing BF16 state cache entries with MXFP8 state caches after prefill.
