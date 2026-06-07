from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

PREFIX_DIR = Path(__file__).resolve().parent
SHARED_DIR = PREFIX_DIR.parents[1] / "shared"
for path in (PREFIX_DIR, SHARED_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from eviction_aware_cache import EvictionAwarePrefixCache, HybridPrefixState


def _make_state(token_ids: list[int], device: torch.device, dtype: torch.dtype) -> HybridPrefixState:
    mamba_states = {
        "layer0": (
            torch.randn(2, 4, 32, 128, device=device, dtype=dtype),
            torch.randn(2, 4, 32, 128, device=device, dtype=dtype),
        )
    }
    kv_cache = {
        "layer0": (
            torch.randn(2, 8, len(token_ids), 64, device=device, dtype=dtype),
            torch.randn(2, 8, len(token_ids), 64, device=device, dtype=dtype),
        )
    }
    return HybridPrefixState(input_ids=list(token_ids), mamba_states=mamba_states, kv_cache=kv_cache)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capacity-mib", type=float, default=8.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=Path("results/hybrid/prefix_cache/demo_real_policy.json"))
    args = parser.parse_args()

    device = torch.device(args.device)
    cache = EvictionAwarePrefixCache(capacity_bytes=int(args.capacity_mib * 1024 * 1024), group_size=32)
    requests = [
        [1, 2, 3, 4, 5, 6],
        [1, 2, 3, 4, 8, 9],
        [1, 2, 3, 7, 10, 11],
        [20, 21, 22, 23],
        [1, 2, 3, 4, 5, 30],
    ]
    for tokens in requests:
        _, branchoff_required, prefix_len = cache.match_prefix(tokens, update_access=False)
        branch_state = _make_state(tokens[:prefix_len], device, torch.float16) if branchoff_required else None
        cache.insert(tokens, _make_state(tokens, device, torch.float16), state_at_branchoff=branch_state)

    cache.free_bytes(1)
    summary = {"stats": cache.stats(), "events": cache.events}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
