Hybrid chunk-update experiment scripts.

Main entry points:

- `run_experiments.py`: full KV cache x Mamba state PPL/latency sweep.
- `run_decode_ppl_latency_mamba_refresh.py`: normal / MXFP8 / MXFP8 refresh256 decode experiments.
- `run_decode_memory_trace_plot.py`: memory trace PNG generation for hybrid decode.
- `run_prefill_decode_memory_sweep.py`: prefill -> decode memory sweep including KV cache.
- `run_refresh_memory_breakdown.py`: refresh256 memory component breakdown.
