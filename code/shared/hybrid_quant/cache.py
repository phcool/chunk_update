from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
from transformers.cache_utils import DynamicCache

from .quant import QuantizedTensor, TensorQuantizer


def dequantize_obj(obj: Any, dtype: Optional[torch.dtype] = None) -> Any:
    if isinstance(obj, QuantizedTensor):
        return obj.dequantize(dtype=dtype)
    if isinstance(obj, tuple):
        return tuple(dequantize_obj(x, dtype=dtype) for x in obj)
    if isinstance(obj, list):
        return [dequantize_obj(x, dtype=dtype) for x in obj]
    if isinstance(obj, dict):
        return {k: dequantize_obj(v, dtype=dtype) for k, v in obj.items()}
    return obj


def quantize_tensor_tree(obj: Any, quantizer: TensorQuantizer, include_conv: bool = True) -> Any:
    if isinstance(obj, torch.Tensor) and obj.is_floating_point():
        return quantizer.quantize(obj)
    if isinstance(obj, tuple):
        return tuple(quantize_tensor_tree(x, quantizer, include_conv=include_conv) for x in obj)
    if isinstance(obj, list):
        return [quantize_tensor_tree(x, quantizer, include_conv=include_conv) for x in obj]
    if isinstance(obj, dict):
        return {k: quantize_tensor_tree(v, quantizer, include_conv=include_conv) for k, v in obj.items()}
    return obj


class QuantizedAttentionCache(DynamicCache):
    """Drop-in replacement for Nemotron's AttentionDynamicCache.

    The model receives dequantized tensors for math, while stored K/V entries are
    kept in the selected compressed representation between calls.
    """

    def __init__(self, config, batch_size: int, quantizer: TensorQuantizer, device=None):
        super().__init__()
        self.quantizer = quantizer
        self.key_cache = [torch.tensor([[]] * batch_size, device=device) for _ in range(config.num_hidden_layers)]
        self.value_cache = [torch.tensor([[]] * batch_size, device=device) for _ in range(config.num_hidden_layers)]

    def __getitem__(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            self.quantizer.dequantize(self.key_cache[layer_idx]),
            self.quantizer.dequantize(self.value_cache[layer_idx]),
        )

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        old_k = self.key_cache[layer_idx]
        old_v = self.value_cache[layer_idx]
        if isinstance(old_k, torch.Tensor) and old_k.shape[-1] == 0:
            full_k = key_states
            full_v = value_states
        else:
            full_k = torch.cat([self.quantizer.dequantize(old_k, key_states.dtype), key_states], dim=2)
            full_v = torch.cat([self.quantizer.dequantize(old_v, value_states.dtype), value_states], dim=2)
        self.key_cache[layer_idx] = self.quantizer.quantize(full_k)
        self.value_cache[layer_idx] = self.quantizer.quantize(full_v)
        return full_k, full_v

    def get_seq_length(self, layer_idx=None) -> int:
        if layer_idx is None:
            lengths = [self._seq_len(x) for x in self.key_cache]
            return max(lengths) if lengths else 0
        return self._seq_len(self.key_cache[layer_idx])

    @staticmethod
    def _seq_len(cache_entry: torch.Tensor | QuantizedTensor) -> int:
        if isinstance(cache_entry, QuantizedTensor):
            return cache_entry.shape[-2]
        if cache_entry.shape[-1] == 0:
            return 0
        return cache_entry.shape[-2]


class NormalAttentionCache(DynamicCache):
    def __init__(self, config, batch_size: int, device=None):
        super().__init__()
        self.key_cache = [torch.tensor([[]] * batch_size, device=device) for _ in range(config.num_hidden_layers)]
        self.value_cache = [torch.tensor([[]] * batch_size, device=device) for _ in range(config.num_hidden_layers)]

    def __getitem__(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if self.key_cache[layer_idx].shape[-1] == 0:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=2)
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx=None) -> int:
        if layer_idx is None:
            return max(self._seq_len(x) for x in self.key_cache)
        return self._seq_len(self.key_cache[layer_idx])

    @staticmethod
    def _seq_len(cache_entry):
        if cache_entry.shape[-1] == 0:
            return 0
        return cache_entry.shape[-2]


def make_attention_cache(config, batch_size: int, kv_mode: str, device, group_size: int = 32):
    if kv_mode == "normal":
        return NormalAttentionCache(config, batch_size=batch_size, device=device)
    quantizer = TensorQuantizer(kv_mode, group_size=group_size)
    return QuantizedAttentionCache(config, batch_size=batch_size, quantizer=quantizer, device=device)


def dequantize_recurrent_caches(mamba_inference_params=None, fla_past_key_values=None):
    if mamba_inference_params is not None:
        mamba_inference_params.key_value_memory_dict = dequantize_obj(mamba_inference_params.key_value_memory_dict)
    if fla_past_key_values is not None and hasattr(fla_past_key_values, "states"):
        fla_past_key_values.states = dequantize_obj(fla_past_key_values.states)


def quantize_recurrent_caches(mamba_inference_params=None, fla_past_key_values=None, mode: str = "normal", group_size: int = 32):
    if mode == "normal":
        return
    quantizer = TensorQuantizer("mxfp8", group_size=group_size, stochastic=True)
    if mamba_inference_params is not None:
        mamba_inference_params.key_value_memory_dict = quantize_tensor_tree(
            mamba_inference_params.key_value_memory_dict, quantizer
        )
    if fla_past_key_values is not None and hasattr(fla_past_key_values, "states"):
        fla_past_key_values.states = quantize_tensor_tree(fla_past_key_values.states, quantizer)
