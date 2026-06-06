from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch


FP8_MAX = 448.0
INT4_MAX = 7.0


def _pad_last_dim(x: torch.Tensor, group_size: int) -> Tuple[torch.Tensor, int]:
    pad = (-x.shape[-1]) % group_size
    if pad:
        x = torch.nn.functional.pad(x, (0, pad))
    return x, pad


def _unpad_last_dim(x: torch.Tensor, pad: int) -> torch.Tensor:
    if pad:
        return x[..., :-pad]
    return x


def _group_view(x: torch.Tensor, group_size: int) -> Tuple[torch.Tensor, int]:
    x_pad, pad = _pad_last_dim(x, group_size)
    return x_pad.reshape(*x_pad.shape[:-1], -1, group_size), pad


def _stochastic_round(x: torch.Tensor) -> torch.Tensor:
    floor = torch.floor(x)
    prob = torch.clamp(x - floor, 0.0, 1.0)
    return floor + torch.bernoulli(prob)


@dataclass
class QuantizedTensor:
    mode: str
    q: torch.Tensor
    scale: Optional[torch.Tensor]
    orig_shape: Tuple[int, ...]
    orig_dtype: torch.dtype
    group_size: int
    pad: int = 0
    zero: Optional[torch.Tensor] = None

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.orig_shape

    @property
    def device(self) -> torch.device:
        return self.q.device

    def dequantize(self, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        out_dtype = dtype or self.orig_dtype
        if self.mode == "normal":
            return self.q.to(out_dtype)
        if self.mode in {"fp8", "mxfp8"}:
            x = self.q.to(torch.float32) * self.scale
            x = _unpad_last_dim(x.reshape(*self.orig_shape[:-1], -1), self.pad)
            return x.reshape(self.orig_shape).to(out_dtype)
        if self.mode == "int4":
            packed = self.q
            low = packed & 0x0F
            high = (packed >> 4) & 0x0F
            vals = torch.stack((low, high), dim=-1).reshape(*packed.shape[:-1], -1)
            vals = vals.to(torch.int16)
            vals = torch.where(vals >= 8, vals - 16, vals).to(torch.float32)
            x = vals.reshape(*self.orig_shape[:-1], -1, self.group_size) * self.scale
            x = _unpad_last_dim(x.reshape(*self.orig_shape[:-1], -1), self.pad)
            return x.reshape(self.orig_shape).to(out_dtype)
        raise ValueError(f"Unknown quantized tensor mode: {self.mode}")


class TensorQuantizer:
    def __init__(self, mode: str, group_size: int = 32, stochastic: bool = False):
        self.mode = mode.lower()
        self.group_size = group_size
        self.stochastic = stochastic
        if self.mode not in {"normal", "fp8", "int4", "mxfp8"}:
            raise ValueError(f"Unsupported quantization mode: {mode}")

    def quantize(self, x: torch.Tensor) -> torch.Tensor | QuantizedTensor:
        if self.mode == "normal":
            return x
        if self.mode == "fp8":
            return self._quantize_fp8(x, mxfp8=False)
        if self.mode == "mxfp8":
            return self._quantize_fp8(x, mxfp8=True)
        if self.mode == "int4":
            return self._quantize_int4(x)
        raise AssertionError("unreachable")

    def dequantize(self, x: torch.Tensor | QuantizedTensor, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        if isinstance(x, QuantizedTensor):
            return x.dequantize(dtype=dtype)
        return x.to(dtype) if dtype is not None else x

    def _quantize_fp8(self, x: torch.Tensor, mxfp8: bool) -> QuantizedTensor:
        if not hasattr(torch, "float8_e4m3fn"):
            raise RuntimeError("This PyTorch build does not expose torch.float8_e4m3fn.")
        orig_dtype = x.dtype
        grouped, pad = _group_view(x.detach().to(torch.float32), self.group_size)
        amax = grouped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        scale = amax / FP8_MAX
        if mxfp8:
            scale = torch.pow(2.0, torch.ceil(torch.log2(scale)))
        y = grouped / scale
        if self.stochastic:
            y = _stochastic_round(y * 16.0) / 16.0
        q = y.clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
        return QuantizedTensor(
            mode="mxfp8" if mxfp8 else "fp8",
            q=q,
            scale=scale,
            orig_shape=tuple(x.shape),
            orig_dtype=orig_dtype,
            group_size=self.group_size,
            pad=pad,
        )

    def _quantize_int4(self, x: torch.Tensor) -> QuantizedTensor:
        orig_dtype = x.dtype
        grouped, pad = _group_view(x.detach().to(torch.float32), self.group_size)
        amax = grouped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        scale = amax / INT4_MAX
        y = grouped / scale
        if self.stochastic:
            y_floor = torch.floor(y)
            y = y_floor + torch.bernoulli(torch.clamp(y - y_floor, 0.0, 1.0))
        q = y.round().clamp(-8, 7).to(torch.int16)
        q_unsigned = torch.where(q < 0, q + 16, q).to(torch.uint8)
        if q_unsigned.shape[-1] % 2:
            q_unsigned = torch.nn.functional.pad(q_unsigned, (0, 1))
        low = q_unsigned[..., 0::2]
        high = q_unsigned[..., 1::2]
        packed = low | (high << 4)
        return QuantizedTensor(
            mode="int4",
            q=packed.contiguous(),
            scale=scale,
            orig_shape=tuple(x.shape),
            orig_dtype=orig_dtype,
            group_size=self.group_size,
            pad=pad,
        )
