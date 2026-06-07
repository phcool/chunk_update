from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import itertools
from pathlib import Path
import sys
import time
from typing import Any, Iterable

import torch
import torch.nn.functional as F

SHARED_DIR = Path(__file__).resolve().parents[1] / "shared"
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))

from hybrid_quant.mxfp8_fused import MXFP8StateCache, dequantize_state_cache
from hybrid_quant.quant import QuantizedTensor


FP8_MAX = 448.0


class StatePrecision(str, Enum):
    NONE = "none"
    MXFP8 = "mxfp8"
    FULL = "full"


@dataclass
class MXFP8SRTensor:
    q: torch.Tensor
    scale_e8m0: torch.Tensor
    orig_shape: tuple[int, ...]
    orig_dtype: torch.dtype
    group_size: int
    pad: int = 0

    def dequantize(self, dtype: torch.dtype | None = None) -> torch.Tensor:
        out_dtype = dtype or self.orig_dtype
        q = self.q.to(torch.float32)
        scale = torch.exp2(self.scale_e8m0.to(torch.float32) - 127.0)
        x = (q * scale.unsqueeze(-1)).reshape(*self.orig_shape[:-1], -1)
        if self.pad:
            x = x[..., :-self.pad]
        return x.reshape(self.orig_shape).to(out_dtype)

    def nbytes(self) -> int:
        return self.q.numel() * self.q.element_size() + self.scale_e8m0.numel() * self.scale_e8m0.element_size()


@dataclass
class HybridPrefixState:
    """Real cached hybrid-model state for one radix node."""

    input_ids: list[int]
    mamba_states: Any = None
    kv_cache: Any = None
    precision: StatePrecision = StatePrecision.FULL

    def has_mamba_state(self) -> bool:
        return self.precision is not StatePrecision.NONE and self.mamba_states is not None

    def has_full_mamba_state(self) -> bool:
        return self.precision is StatePrecision.FULL and self.mamba_states is not None

    def has_quantized_mamba_state(self) -> bool:
        return self.precision is StatePrecision.MXFP8 and self.mamba_states is not None

    def quantize_mamba_state(self, group_size: int = 32) -> int:
        if not self.has_full_mamba_state():
            return 0
        before = tensor_tree_nbytes(self.mamba_states)
        self.mamba_states = quantize_tensor_tree_mxfp8_sr(self.mamba_states, group_size)
        self.precision = StatePrecision.MXFP8
        after = tensor_tree_nbytes(self.mamba_states)
        return max(0, before - after)

    def dequantize_mamba_state(self, dtype: torch.dtype | None = None) -> Any:
        if self.precision is not StatePrecision.MXFP8:
            return self.mamba_states
        return dequantize_tensor_tree(self.mamba_states, dtype=dtype)

    def drop_mamba_state(self) -> int:
        before = tensor_tree_nbytes(self.mamba_states)
        self.mamba_states = None
        self.precision = StatePrecision.NONE
        return before

    def drop_all(self) -> int:
        before = self.nbytes()
        self.mamba_states = None
        self.kv_cache = None
        self.precision = StatePrecision.NONE
        return before

    def nbytes(self) -> int:
        return tensor_tree_nbytes(self.mamba_states) + tensor_tree_nbytes(self.kv_cache)

    def mamba_nbytes(self) -> int:
        return tensor_tree_nbytes(self.mamba_states)

    def kv_nbytes(self) -> int:
        return tensor_tree_nbytes(self.kv_cache)


@dataclass
class PrefixCacheNode:
    key: tuple[int, ...] = field(default_factory=tuple)
    value: list[int] = field(default_factory=list)
    parent: "PrefixCacheNode | None" = None
    children: dict[int, "PrefixCacheNode"] = field(default_factory=dict)
    hybrid_state: HybridPrefixState | None = None
    last_access_time: float = field(default_factory=time.time)

    def __lt__(self, other: "PrefixCacheNode") -> bool:
        return self.last_access_time < other.last_access_time

    @property
    def is_root(self) -> bool:
        return self.parent is None

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0 and not self.is_root

    @property
    def is_single_child_intermediate(self) -> bool:
        return len(self.children) == 1 and not self.is_root

    @property
    def has_state(self) -> bool:
        return self.hybrid_state is not None and self.hybrid_state.has_mamba_state()

    def token_path(self) -> list[int]:
        node = self
        chunks: list[list[int]] = []
        while node.parent is not None:
            chunks.append(node.value)
            node = node.parent
        chunks.reverse()
        return list(itertools.chain.from_iterable(chunks))


class EvictionAwarePrefixCache:
    """Radix prefix cache with real MXFP8-SR state compression.

    Under memory pressure the cache first compresses the oldest full-precision
    leaf states. When a branch has at most one full-precision leaf below it, the
    branch state is compressed as well. Only after all available states are
    compressed does it evict nodes using Marconi's utility objective.
    """

    def __init__(
        self,
        capacity_bytes: int,
        *,
        group_size: int = 32,
        num_ssm_layers: int = 6,
        num_attn_layers: int = 18,
        num_mlp_layers: int = 24,
        hidden_size: int = 2048,
        state_size: int = 128,
        eff_weight: float = 0.0,
        use_logical_ts: bool = True,
    ):
        self.root = PrefixCacheNode()
        self.capacity_bytes = int(capacity_bytes)
        self.group_size = group_size
        self.num_ssm_layers = num_ssm_layers
        self.num_attn_layers = num_attn_layers
        self.num_mlp_layers = num_mlp_layers
        self.hidden_size = hidden_size
        self.state_size = state_size
        self.eff_weight = eff_weight
        self.use_logical_ts = use_logical_ts
        self.logical_ts = 0
        self.num_nodes = 0
        self.events: list[dict[str, Any]] = []

    def insert(
        self,
        token_ids: list[int],
        state_at_leaf: HybridPrefixState,
        *,
        state_at_branchoff: HybridPrefixState | None = None,
    ) -> PrefixCacheNode:
        self._tick()
        token_ids = list(token_ids)
        _, branchoff_required, _ = self.match_prefix(token_ids, update_access=False)
        if branchoff_required and state_at_branchoff is None:
            raise ValueError("Insertion creates a branch node, but state_at_branchoff was not provided.")

        bytes_needed = state_at_leaf.nbytes()
        if branchoff_required:
            bytes_needed += state_at_branchoff.nbytes()
        self.ensure_capacity(bytes_needed)
        return self._insert_helper(self.root, tuple(token_ids), token_ids, state_at_leaf, state_at_branchoff)

    def match_prefix(self, token_ids: list[int], *, update_access: bool = True) -> tuple[list[int], bool, int]:
        self._tick()
        matched_chunks: list[list[int]] = []
        prefix_len, last_reusable = self._match_prefix_helper(self.root, tuple(token_ids), matched_chunks)
        reusable = list(itertools.chain.from_iterable(matched_chunks))
        branchoff_required = prefix_len > len(reusable)
        if update_access and last_reusable is not None and not last_reusable.is_root:
            last_reusable.last_access_time = self._now()
        return reusable, branchoff_required, prefix_len

    def ensure_capacity(self, bytes_needed: int) -> int:
        overflow = self.total_nbytes() + bytes_needed - self.capacity_bytes
        if overflow <= 0:
            return 0
        return self.free_bytes(overflow)

    def free_bytes(self, target_bytes: int) -> int:
        freed = self.quantize_for_pressure(target_bytes)
        while freed < target_bytes and self.num_nodes > 0:
            node = self._select_marconi_victim()
            if node is None:
                break
            freed += self._evict_node(node)
        return freed

    def quantize_for_pressure(self, target_bytes: int) -> int:
        freed = 0
        for leaf in sorted(self._collect_leaves(), key=lambda node: node.last_access_time):
            if freed >= target_bytes:
                break
            if not (leaf.hybrid_state and leaf.hybrid_state.has_full_mamba_state()):
                continue
            delta = leaf.hybrid_state.quantize_mamba_state(self.group_size)
            freed += delta
            self.events.append({"type": "quantize_leaf", "tokens": len(leaf.token_path()), "freed_bytes": delta})
            freed += self._quantize_following_branches(leaf, target_bytes - freed)
        return freed

    def total_nbytes(self) -> int:
        return sum(node.hybrid_state.nbytes() for node in self._iter_nodes() if node.hybrid_state is not None)

    def mamba_nbytes(self) -> int:
        return sum(node.hybrid_state.mamba_nbytes() for node in self._iter_nodes() if node.hybrid_state is not None)

    def kv_nbytes(self) -> int:
        return sum(node.hybrid_state.kv_nbytes() for node in self._iter_nodes() if node.hybrid_state is not None)

    def stats(self) -> dict[str, int]:
        nodes = list(self._iter_nodes())
        full = sum(1 for node in nodes if node.hybrid_state and node.hybrid_state.precision is StatePrecision.FULL)
        quant = sum(1 for node in nodes if node.hybrid_state and node.hybrid_state.precision is StatePrecision.MXFP8)
        none = sum(1 for node in nodes if node.hybrid_state and node.hybrid_state.precision is StatePrecision.NONE)
        return {
            "nodes": len(nodes),
            "full_state_nodes": full,
            "mxfp8_state_nodes": quant,
            "empty_state_nodes": none,
            "total_bytes": self.total_nbytes(),
            "mamba_bytes": self.mamba_nbytes(),
            "kv_bytes": self.kv_nbytes(),
        }

    def _insert_helper(
        self,
        node: PrefixCacheNode,
        key: tuple[int, ...],
        value: list[int],
        state_at_leaf: HybridPrefixState,
        state_at_branchoff: HybridPrefixState | None,
    ) -> PrefixCacheNode:
        if not key:
            return node
        child = node.children.get(key[0])
        if child is None:
            new_node = PrefixCacheNode(
                key=key,
                value=value,
                parent=node,
                hybrid_state=state_at_leaf,
                last_access_time=self._now(),
            )
            node.children[key[0]] = new_node
            self.num_nodes += 1
            return new_node

        prefix_len = _key_match(child.key, key)
        if prefix_len == len(child.key):
            if prefix_len == len(key):
                child.hybrid_state = state_at_leaf
                child.last_access_time = self._now()
                return child
            return self._insert_helper(child, key[prefix_len:], value[prefix_len:], state_at_leaf, state_at_branchoff)

        if state_at_branchoff is None:
            raise ValueError("Split requires state_at_branchoff.")
        branch = self._split_node(child, prefix_len, state_at_branchoff)
        return self._insert_helper(branch, key[prefix_len:], value[prefix_len:], state_at_leaf, None)

    def _split_node(
        self,
        child: PrefixCacheNode,
        split_len: int,
        state_at_branchoff: HybridPrefixState,
    ) -> PrefixCacheNode:
        old_parent = child.parent
        if old_parent is None:
            raise RuntimeError("Cannot split root.")
        branch = PrefixCacheNode(
            key=child.key[:split_len],
            value=child.value[:split_len],
            parent=old_parent,
            hybrid_state=state_at_branchoff,
            last_access_time=self._now(),
        )
        child.key = child.key[split_len:]
        child.value = child.value[split_len:]
        child.parent = branch
        branch.children[child.key[0]] = child
        old_parent.children[branch.key[0]] = branch
        self.num_nodes += 1
        return branch

    def _match_prefix_helper(
        self,
        node: PrefixCacheNode,
        key: tuple[int, ...],
        matched_chunks: list[list[int]],
    ) -> tuple[int, PrefixCacheNode | None]:
        if not key:
            return 0, node
        child = node.children.get(key[0])
        if child is None:
            return 0, node
        prefix_len = _key_match(child.key, key)
        if prefix_len < len(child.key):
            return prefix_len, node
        if child.has_state:
            matched_chunks.append(child.value)
            last_reusable = child
        else:
            last_reusable = node
        suffix_len, suffix_node = self._match_prefix_helper(child, key[prefix_len:], matched_chunks)
        return prefix_len + suffix_len, suffix_node or last_reusable

    def _quantize_following_branches(self, leaf: PrefixCacheNode, remaining_target: int) -> int:
        freed = 0
        node = leaf.parent
        while node is not None and not node.is_root:
            if len(node.children) >= 2 and node.hybrid_state and node.hybrid_state.has_full_mamba_state():
                if self._count_full_precision_leaves(node) <= 1:
                    delta = node.hybrid_state.quantize_mamba_state(self.group_size)
                    freed += delta
                    self.events.append({"type": "quantize_branch", "tokens": len(node.token_path()), "freed_bytes": delta})
                    if freed >= remaining_target:
                        break
            node = node.parent
        return freed

    def _select_marconi_victim(self) -> PrefixCacheNode | None:
        candidates = [node for node in self._iter_nodes() if node.is_leaf or node.is_single_child_intermediate]
        if not candidates:
            return None
        now = self._now()
        efficiencies = [self._flops_efficiency(node) for node in candidates]
        recency = [1.0 / max(now - node.last_access_time, 1.0) for node in candidates]
        eff_scores = _normalize(efficiencies)
        recency_scores = _normalize(recency)
        utilities = [
            self.eff_weight * eff_score + recency_score
            for eff_score, recency_score in zip(eff_scores, recency_scores)
        ]
        return candidates[utilities.index(min(utilities))]

    def _evict_node(self, node: PrefixCacheNode) -> int:
        if node.is_leaf:
            freed = node.hybrid_state.drop_all() if node.hybrid_state is not None else 0
            self._delete_leaf(node)
            self.events.append({"type": "delete_leaf", "tokens": len(node.token_path()), "freed_bytes": freed})
            return freed
        if node.is_single_child_intermediate:
            freed = node.hybrid_state.drop_mamba_state() if node.hybrid_state is not None else 0
            self._evict_intermediate_node(node)
            self.events.append({"type": "delete_intermediate_state", "tokens": len(node.token_path()), "freed_bytes": freed})
            return freed
        return 0

    def _delete_leaf(self, node: PrefixCacheNode) -> None:
        if node.parent is None:
            return
        del node.parent.children[node.key[0]]
        self.num_nodes -= 1

    def _evict_intermediate_node(self, node: PrefixCacheNode) -> None:
        if not node.is_single_child_intermediate or node.parent is None:
            return
        child = next(iter(node.children.values()))
        child.key = tuple(node.value + child.value)
        child.value = node.value + child.value
        child.parent = node.parent
        node.parent.children[node.key[0]] = child
        self.num_nodes -= 1

    def _flops_efficiency(self, node: PrefixCacheNode) -> float:
        seqlen_total = len(node.token_path())
        seqlen_child = len(node.value)
        seqlen_parent = max(0, seqlen_total - seqlen_child)
        mamba = self.num_ssm_layers * _mamba_flops(seqlen_child, self.hidden_size, self.state_size)
        attn = self.num_attn_layers * (
            _attn_flops(seqlen_total, self.hidden_size) - _attn_flops(seqlen_parent, self.hidden_size)
        )
        mlp = self.num_mlp_layers * (
            _mlp_flops(seqlen_total, self.hidden_size) - _mlp_flops(seqlen_parent, self.hidden_size)
        )
        memory = node.hybrid_state.nbytes() if node.hybrid_state is not None else 1
        return (mamba + attn + mlp) / max(memory, 1)

    def _count_full_precision_leaves(self, node: PrefixCacheNode) -> int:
        total = 0
        for leaf in self._collect_leaves(node):
            if leaf.hybrid_state is not None and leaf.hybrid_state.has_full_mamba_state():
                total += 1
        return total

    def _collect_leaves(self, start: PrefixCacheNode | None = None) -> list[PrefixCacheNode]:
        leaves: list[PrefixCacheNode] = []
        for node in self._iter_nodes(start or self.root):
            if node.is_leaf:
                leaves.append(node)
        return leaves

    def _iter_nodes(self, start: PrefixCacheNode | None = None) -> Iterable[PrefixCacheNode]:
        root = start or self.root
        stack = list(root.children.values()) if root.is_root else [root]
        while stack:
            node = stack.pop()
            yield node
            stack.extend(node.children.values())

    def _tick(self) -> None:
        if self.use_logical_ts:
            self.logical_ts += 1

    def _now(self) -> float:
        return float(self.logical_ts) if self.use_logical_ts else time.time()


def quantize_tensor_tree_mxfp8_sr(obj: Any, group_size: int = 32) -> Any:
    if isinstance(obj, (MXFP8StateCache, MXFP8SRTensor, QuantizedTensor)):
        return obj
    if isinstance(obj, torch.Tensor) and obj.is_floating_point():
        return quantize_tensor_mxfp8_sr(obj, group_size=group_size)
    if isinstance(obj, tuple):
        return tuple(quantize_tensor_tree_mxfp8_sr(x, group_size) for x in obj)
    if isinstance(obj, list):
        return [quantize_tensor_tree_mxfp8_sr(x, group_size) for x in obj]
    if isinstance(obj, dict):
        return {k: quantize_tensor_tree_mxfp8_sr(v, group_size) for k, v in obj.items()}
    return obj


def quantize_tensor_mxfp8_sr(x: torch.Tensor, group_size: int = 32) -> MXFP8SRTensor:
    if not hasattr(torch, "float8_e4m3fn"):
        raise RuntimeError("This PyTorch build does not expose torch.float8_e4m3fn.")
    orig_shape = tuple(x.shape)
    orig_dtype = x.dtype
    x_fp32 = x.detach().to(torch.float32)
    pad = (-x_fp32.shape[-1]) % group_size
    if pad:
        x_fp32 = F.pad(x_fp32, (0, pad))
    grouped = x_fp32.reshape(*x_fp32.shape[:-1], -1, group_size)
    amax = grouped.abs().amax(dim=-1, keepdim=True)
    raw_exp = torch.ceil(torch.log2(torch.clamp(amax / FP8_MAX, min=1.0e-30)))
    biased_exp = torch.clamp(raw_exp + 127.0, min=1.0, max=254.0)
    scale = torch.where(amax == 0.0, torch.zeros_like(amax), torch.exp2(biased_exp - 127.0))
    y = torch.where(amax == 0.0, torch.zeros_like(grouped), grouped / scale)
    y = torch.clamp(y, -FP8_MAX, FP8_MAX)
    y_floor = torch.floor(y * 16.0)
    y = (y_floor + torch.bernoulli(torch.clamp(y * 16.0 - y_floor, 0.0, 1.0))) / 16.0
    return MXFP8SRTensor(
        q=y.to(torch.float8_e4m3fn).contiguous(),
        scale_e8m0=biased_exp.squeeze(-1).to(torch.uint8).contiguous(),
        orig_shape=orig_shape,
        orig_dtype=orig_dtype,
        group_size=group_size,
        pad=pad,
    )


def dequantize_tensor_tree(obj: Any, dtype: torch.dtype | None = None) -> Any:
    if isinstance(obj, MXFP8SRTensor):
        return obj.dequantize(dtype=dtype)
    if isinstance(obj, QuantizedTensor):
        return obj.dequantize(dtype=dtype)
    if isinstance(obj, MXFP8StateCache):
        return dequantize_state_cache(obj, dtype=dtype or torch.float16)
    if isinstance(obj, tuple):
        return tuple(dequantize_tensor_tree(x, dtype=dtype) for x in obj)
    if isinstance(obj, list):
        return [dequantize_tensor_tree(x, dtype=dtype) for x in obj]
    if isinstance(obj, dict):
        return {k: dequantize_tensor_tree(v, dtype=dtype) for k, v in obj.items()}
    return obj


def tensor_tree_nbytes(obj: Any) -> int:
    if obj is None:
        return 0
    if isinstance(obj, torch.Tensor):
        return obj.numel() * obj.element_size()
    if isinstance(obj, MXFP8SRTensor):
        return obj.nbytes()
    if isinstance(obj, QuantizedTensor):
        return tensor_tree_nbytes(obj.q) + tensor_tree_nbytes(obj.scale) + tensor_tree_nbytes(obj.zero)
    if isinstance(obj, MXFP8StateCache):
        return tensor_tree_nbytes(obj.q_state) + tensor_tree_nbytes(obj.scale_e8m0)
    if isinstance(obj, dict):
        return sum(tensor_tree_nbytes(v) for v in obj.values())
    if isinstance(obj, (tuple, list)):
        return sum(tensor_tree_nbytes(v) for v in obj)
    return 0


def _key_match(key0: tuple[int, ...], key1: tuple[int, ...]) -> int:
    matched = 0
    for x, y in zip(key0, key1):
        if x != y:
            break
        matched += 1
    return matched


def _normalize(values: list[float]) -> list[float]:
    if len(values) <= 1:
        return [1.0] * len(values)
    min_val = min(values)
    max_val = max(values)
    if min_val == max_val:
        return [1.0] * len(values)
    return [(value - min_val) / (max_val - min_val) for value in values]


def _attn_flops(length: int, hidden_size: int) -> int:
    return 8 * length * hidden_size**2 + 4 * length**2 * hidden_size


def _mlp_flops(length: int, hidden_size: int) -> int:
    return 16 * length * hidden_size**2


def _mamba_flops(length: int, hidden_size: int, state_size: int) -> int:
    return 12 * length * hidden_size**2 + 16 * length * hidden_size * state_size + 10 * length * hidden_size
