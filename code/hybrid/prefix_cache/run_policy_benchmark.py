from __future__ import annotations

import argparse
from collections import Counter
import gc
import json
from pathlib import Path
import random
import sys
import time
from typing import Any

import torch
from transformers import AutoTokenizer

PREFIX_DIR = Path(__file__).resolve().parent
SHARED_DIR = PREFIX_DIR.parents[1] / "shared"
for path in (PREFIX_DIR, SHARED_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from eviction_aware_cache import EvictionAwarePrefixCache, HybridPrefixState


POLICIES = {
    "marconi": {"quantize_on_pressure": False, "branch_follow_quantization": False, "quantize_on_insert": False},
    "leaf_quant": {"quantize_on_pressure": True, "branch_follow_quantization": False, "quantize_on_insert": False},
    "leaf_branch_quant": {"quantize_on_pressure": True, "branch_follow_quantization": True, "quantize_on_insert": False},
    "all_mxfp8": {"quantize_on_pressure": False, "branch_follow_quantization": False, "quantize_on_insert": True},
}


class StateFactory:
    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        batch_size: int,
        num_mamba_layers: int,
        state_heads: int,
        state_dim: int,
        dstate: int,
        num_attn_layers: int,
        kv_heads: int,
        kv_head_dim: int,
    ):
        self.device = device
        self.dtype = dtype
        self.batch_size = batch_size
        self.num_mamba_layers = num_mamba_layers
        self.state_heads = state_heads
        self.state_dim = state_dim
        self.dstate = dstate
        self.num_attn_layers = num_attn_layers
        self.kv_heads = kv_heads
        self.kv_head_dim = kv_head_dim

    def make(self, input_ids: list[int], kv_tokens: int, *, quantized: bool = False, group_size: int = 32) -> HybridPrefixState:
        mamba_states = {
            layer: torch.zeros(
                self.batch_size,
                self.state_heads,
                self.state_dim,
                self.dstate,
                device=self.device,
                dtype=self.dtype,
            )
            for layer in range(self.num_mamba_layers)
        }
        kv_len = max(0, kv_tokens)
        kv_cache = {
            layer: (
                torch.zeros(
                    self.batch_size,
                    self.kv_heads,
                    kv_len,
                    self.kv_head_dim,
                    device=self.device,
                    dtype=self.dtype,
                ),
                torch.zeros(
                    self.batch_size,
                    self.kv_heads,
                    kv_len,
                    self.kv_head_dim,
                    device=self.device,
                    dtype=self.dtype,
                ),
            )
            for layer in range(self.num_attn_layers)
        }
        state = HybridPrefixState(input_ids=list(input_ids), mamba_states=mamba_states, kv_cache=kv_cache)
        if quantized:
            state.quantize_mamba_state(group_size)
        return state


def generate_branching_trace(
    *,
    num_requests: int,
    num_families: int,
    branches_per_family: int,
    system_len: int,
    family_len: int,
    branch_len: int,
    suffix_len: int,
    repeat_prob: float,
    seed: int,
) -> list[list[int]]:
    rng = random.Random(seed)
    system = list(range(1, system_len + 1))
    families = []
    token_base = 10_000
    for family_idx in range(num_families):
        family = list(range(token_base, token_base + family_len))
        token_base += family_len
        branches = []
        for _ in range(branches_per_family):
            branch = list(range(token_base, token_base + branch_len))
            token_base += branch_len
            branches.append(branch)
        families.append((family, branches))

    previous: list[list[int]] = []
    trace = []
    for req_idx in range(num_requests):
        if previous and rng.random() < repeat_prob:
            base = rng.choice(previous)
            if rng.random() < 0.35:
                trace.append(list(base))
            else:
                extension = list(range(token_base, token_base + suffix_len))
                token_base += suffix_len
                trace.append(list(base) + extension)
            continue

        family_idx = int(rng.random() ** 1.7 * num_families)
        family, branches = families[family_idx]
        branch = branches[rng.randrange(branches_per_family)]
        suffix = list(range(token_base, token_base + suffix_len))
        token_base += suffix_len
        request = system + family + branch + suffix
        previous.append(request)
        trace.append(request)
    return trace


def load_ultrachat_trace(
    *,
    dataset_path: Path,
    tokenizer_path: Path,
    num_requests: int,
    max_tokens: int,
    min_tokens: int,
) -> list[list[int]]:
    from datasets import load_from_disk

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), trust_remote_code=True)
    ds = load_from_disk(str(dataset_path))
    trace: list[list[int]] = []

    for row in ds:
        messages = row.get("messages") or row.get("conversations")
        if not messages:
            continue
        prefix_text = ""
        for message in messages:
            role = message.get("role") or message.get("from") or "unknown"
            content = message.get("content") or message.get("value") or ""
            if not content:
                continue
            prefix_text += f"{role}: {content}\n"
            if role in {"user", "human"}:
                token_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
                if len(token_ids) < min_tokens:
                    continue
                trace.append(token_ids[:max_tokens])
                if num_requests > 0 and len(trace) >= num_requests:
                    return trace
    if not trace:
        raise RuntimeError(f"No usable conversation prefixes found in {dataset_path}.")
    return trace


def iter_ultrachat_trace(
    *,
    dataset_path: Path,
    tokenizer_path: Path,
    num_requests: int,
    max_tokens: int,
    min_tokens: int,
):
    from datasets import load_from_disk

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), trust_remote_code=True)
    ds = load_from_disk(str(dataset_path))
    emitted = 0
    for row_idx, row in enumerate(ds, start=1):
        messages = row.get("messages") or row.get("conversations")
        if not messages:
            continue
        prefix_text = ""
        for message in messages:
            role = message.get("role") or message.get("from") or "unknown"
            content = message.get("content") or message.get("value") or ""
            if not content:
                continue
            prefix_text += f"{role}: {content}\n"
            if role in {"user", "human"}:
                token_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
                if len(token_ids) < min_tokens:
                    continue
                yield token_ids[:max_tokens]
                emitted += 1
                if num_requests > 0 and emitted >= num_requests:
                    return
        if row_idx % 10000 == 0:
            print(f"[trace] scanned_rows={row_idx} emitted_requests={emitted}", flush=True)


def run_policy(
    policy_name: str,
    trace,
    capacity_bytes: int,
    factory: StateFactory,
    args: argparse.Namespace,
) -> dict[str, Any]:
    cfg = POLICIES[policy_name]
    cache = EvictionAwarePrefixCache(
        capacity_bytes=capacity_bytes,
        group_size=args.group_size,
        num_ssm_layers=args.num_mamba_layers,
        num_attn_layers=args.num_attn_layers,
        num_mlp_layers=args.num_mlp_layers,
        hidden_size=args.hidden_size,
        state_size=args.model_state_size,
        eff_weight=args.eff_weight,
        quantize_on_pressure=cfg["quantize_on_pressure"],
        branch_follow_quantization=cfg["branch_follow_quantization"],
    )

    total_tokens = 0
    reused_tokens = 0
    request_hits = 0
    insertions = 0
    lookup_time_s = 0.0
    insert_time_s = 0.0
    max_cache_bytes = 0
    request_count = 0
    trace_len = len(trace) if hasattr(trace, "__len__") else None
    start = time.perf_counter()

    for req_idx, tokens in enumerate(trace, start=1):
        request_count = req_idx
        lookup_start = time.perf_counter()
        reusable, branchoff_required, prefix_len = cache.match_prefix(tokens, update_access=True)
        lookup_time_s += time.perf_counter() - lookup_start

        total_tokens += len(tokens)
        reused_tokens += len(reusable)
        request_hits += int(len(reusable) > 0)

        if prefix_len == len(tokens) and len(reusable) == len(tokens):
            max_cache_bytes = max(max_cache_bytes, cache.total_nbytes())
            continue

        suffix_len = max(0, len(tokens) - prefix_len)
        quantized = cfg["quantize_on_insert"]
        branch_state = None
        if branchoff_required:
            branch_state = factory.make(tokens[:prefix_len], prefix_len, quantized=quantized, group_size=args.group_size)
        leaf_state = factory.make(tokens, suffix_len if prefix_len else len(tokens), quantized=quantized, group_size=args.group_size)

        insert_start = time.perf_counter()
        cache.insert(tokens, leaf_state, state_at_branchoff=branch_state)
        insert_time_s += time.perf_counter() - insert_start
        insertions += 1
        max_cache_bytes = max(max_cache_bytes, cache.total_nbytes())
        if args.progress_every > 0 and req_idx % args.progress_every == 0:
            total_repr = str(trace_len) if trace_len is not None else "?"
            print(
                f"[{policy_name}] {req_idx}/{total_repr} "
                f"hit={reused_tokens / max(total_tokens, 1):.4f} "
                f"nodes={cache.stats()['nodes']} "
                f"cache_mib={cache.total_nbytes() / 2**20:.1f}",
                flush=True,
            )

    elapsed_s = time.perf_counter() - start
    event_counts = Counter(event["type"] for event in cache.events)
    stats = cache.stats()
    recompute_tokens = total_tokens - reused_tokens
    return {
        "policy": policy_name,
        "capacity_bytes": capacity_bytes,
        "capacity_mib": capacity_bytes / 2**20,
        "requests": request_count,
        "insertions": insertions,
        "request_hit_rate": request_hits / max(request_count, 1),
        "token_hit_rate": reused_tokens / max(total_tokens, 1),
        "reused_tokens": reused_tokens,
        "recompute_tokens": recompute_tokens,
        "avg_recompute_tokens": recompute_tokens / max(request_count, 1),
        "max_cache_bytes": max_cache_bytes,
        "elapsed_s": elapsed_s,
        "lookup_time_s": lookup_time_s,
        "insert_time_s": insert_time_s,
        "events": dict(event_counts),
        **stats,
    }


def run_once(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    stream_ultrachat = args.stream_trace and args.trace_source == "ultrachat"
    if args.trace_source == "synthetic":
        trace = generate_branching_trace(
            num_requests=args.num_requests,
            num_families=args.num_families,
            branches_per_family=args.branches_per_family,
            system_len=args.system_len,
            family_len=args.family_len,
            branch_len=args.branch_len,
            suffix_len=args.suffix_len,
            repeat_prob=args.repeat_prob,
            seed=args.seed,
        )
    elif args.trace_source == "ultrachat" and not stream_ultrachat:
        trace = load_ultrachat_trace(
            dataset_path=args.ultrachat_path,
            tokenizer_path=args.tokenizer_path,
            num_requests=args.num_requests,
            max_tokens=args.max_tokens,
            min_tokens=args.min_tokens,
        )
    elif stream_ultrachat:
        trace = None
    else:
        raise ValueError(f"Unknown trace source: {args.trace_source}")
    factory = StateFactory(
        device=device,
        dtype=dtype,
        batch_size=args.batch_size,
        num_mamba_layers=args.num_mamba_layers,
        state_heads=args.state_heads,
        state_dim=args.state_dim,
        dstate=args.dstate,
        num_attn_layers=args.num_attn_layers,
        kv_heads=args.kv_heads,
        kv_head_dim=args.kv_head_dim,
    )

    full = None
    full_bytes = None
    if not args.skip_unlimited:
        if stream_ultrachat:
            raise ValueError("--stream-trace requires --skip-unlimited; use --capacity-mib for large runs.")
        full = run_policy("marconi", trace, 10**15, factory, args)
        full_bytes = full["total_bytes"]
    results = []
    capacity_jobs: list[tuple[float | None, int]] = []
    if args.no_budget_ratios:
        args.budget_ratios = []
    if args.capacity_mib:
        capacity_jobs.extend((None, int(mib * 2**20)) for mib in args.capacity_mib)
    if args.budget_ratios:
        if full_bytes is None:
            raise ValueError("--budget-ratios requires unlimited baseline; remove --skip-unlimited or use --capacity-mib.")
        capacity_jobs.extend((ratio, max(1, int(full_bytes * ratio))) for ratio in args.budget_ratios)
    for ratio, capacity in capacity_jobs:
        for policy in args.policies:
            policy_trace = (
                iter_ultrachat_trace(
                    dataset_path=args.ultrachat_path,
                    tokenizer_path=args.tokenizer_path,
                    num_requests=args.num_requests,
                    max_tokens=args.max_tokens,
                    min_tokens=args.min_tokens,
                )
                if stream_ultrachat
                else trace
            )
            result = run_policy(policy, policy_trace, capacity, factory, args)
            result["budget_ratio"] = ratio
            result["capacity_mib_requested"] = capacity / 2**20
            result["full_cache_bytes"] = full_bytes
            results.append(result)
            if args.incremental_output:
                partial_trace_summary = {
                    "requests": results[0]["requests"],
                    "total_tokens": results[0]["reused_tokens"] + results[0]["recompute_tokens"],
                    "avg_tokens": (results[0]["reused_tokens"] + results[0]["recompute_tokens"])
                    / max(results[0]["requests"], 1),
                    "streamed": stream_ultrachat,
                    "partial": True,
                }
                partial = {
                    "config": {
                        key: str(value) if isinstance(value, Path) else value
                        for key, value in vars(args).items()
                    },
                    "trace": partial_trace_summary,
                    "unlimited_marconi": full,
                    "results": results,
                    "partial": True,
                }
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(partial, indent=2), encoding="utf-8")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if stream_ultrachat and results:
        trace_summary = {
            "requests": results[0]["requests"],
            "total_tokens": results[0]["reused_tokens"] + results[0]["recompute_tokens"],
            "avg_tokens": (results[0]["reused_tokens"] + results[0]["recompute_tokens"]) / max(results[0]["requests"], 1),
            "streamed": True,
        }
    else:
        trace_summary = {
            "requests": len(trace),
            "total_tokens": sum(len(x) for x in trace),
            "avg_tokens": sum(len(x) for x in trace) / max(len(trace), 1),
            "streamed": False,
        }
    return {
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "trace": trace_summary,
        "unlimited_marconi": full,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/hybrid/prefix_cache/policy_benchmark.json"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    parser.add_argument("--trace-source", choices=["synthetic", "ultrachat"], default="synthetic")
    parser.add_argument("--stream-trace", action="store_true")
    parser.add_argument("--ultrachat-path", type=Path, default=Path("datasets/ultrachat_200k/train_sft"))
    parser.add_argument("--tokenizer-path", type=Path, default=Path("/home/vrintern/tmp/models/Nemotron-Flash-1B"))
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--min-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-requests", type=int, default=160, help="Use <=0 to consume every usable request.")
    parser.add_argument("--num-families", type=int, default=16)
    parser.add_argument("--branches-per-family", type=int, default=4)
    parser.add_argument("--system-len", type=int, default=128)
    parser.add_argument("--family-len", type=int, default=96)
    parser.add_argument("--branch-len", type=int, default=64)
    parser.add_argument("--suffix-len", type=int, default=64)
    parser.add_argument("--repeat-prob", type=float, default=0.35)
    parser.add_argument("--budget-ratios", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    parser.add_argument("--no-budget-ratios", action="store_true")
    parser.add_argument("--capacity-mib", type=float, nargs="+", default=None)
    parser.add_argument("--skip-unlimited", action="store_true")
    parser.add_argument("--progress-every", type=int, default=0)
    parser.add_argument("--incremental-output", action="store_true")
    parser.add_argument("--policies", nargs="+", default=list(POLICIES))
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-mamba-layers", type=int, default=6)
    parser.add_argument("--state-heads", type=int, default=8)
    parser.add_argument("--state-dim", type=int, default=64)
    parser.add_argument("--dstate", type=int, default=64)
    parser.add_argument("--num-attn-layers", type=int, default=2)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--kv-head-dim", type=int, default=64)
    parser.add_argument("--num-mlp-layers", type=int, default=24)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--model-state-size", type=int, default=128)
    parser.add_argument("--eff-weight", type=float, default=0.0)
    args = parser.parse_args()

    for policy in args.policies:
        if policy not in POLICIES:
            raise ValueError(f"Unknown policy {policy}; choices: {sorted(POLICIES)}")

    summary = run_once(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
