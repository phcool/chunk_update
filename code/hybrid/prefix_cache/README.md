Hybrid prefix-cache experiments.

Current implementation:

- `eviction_aware_cache.py`: real radix prefix cache storing actual `mamba_states`
  and `kv_cache` tensor trees.
- On memory pressure, oldest full-precision leaves are quantized first with
  MXFP8 stochastic rounding.
- Branch states are automatically quantized when the branch has at most one
  full-precision leaf below it.
- If no more state can be quantized, node deletion falls back to Marconi-style
  utility over leaves and single-child intermediate nodes.
- `demo_real_policy.py`: small real-tensor smoke test for the policy.

Integration point: pass runtime state objects into `HybridPrefixState`.
The cache does not simulate memory savings; it stores compressed tensor objects
and reports bytes from the actual backing tensors.
