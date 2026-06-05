from __future__ import annotations

from dataclasses import dataclass
from types import MethodType
import time

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from einops import rearrange, repeat

try:
    from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as _RMSNormGated  # noqa: F401
except Exception:
    pass

from mamba_ssm.ops.triton.softplus import softplus


FP8_MAX = 448.0


def _profile_enabled(module) -> bool:
    return getattr(module, "_mxfp8_profile", None) is not None


def _cuda_sync_if_needed(device):
    if torch.cuda.is_available() and torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def _profile_section(module, name: str, fn):
    profile = getattr(module, "_mxfp8_profile", None)
    if profile is None:
        return fn()
    device = next(module.parameters()).device
    _cuda_sync_if_needed(device)
    start = time.perf_counter()
    out = fn()
    _cuda_sync_if_needed(device)
    profile[name] = profile.get(name, 0.0) + time.perf_counter() - start
    profile[name + "_calls"] = profile.get(name + "_calls", 0) + 1
    return out


@dataclass
class MXFP8StateCache:
    q_state: torch.Tensor
    scale_e8m0: torch.Tensor
    group_size: int
    initialized: bool = False


@triton.jit
def _quantize_state_kernel(
    state_ptr,
    q_ptr,
    scale_e8m0_ptr,
    total_rows: tl.constexpr,
    dstate: tl.constexpr,
    stride_state_row: tl.constexpr,
    stride_state_n: tl.constexpr,
    stride_q_row: tl.constexpr,
    stride_q_n: tl.constexpr,
    stride_scale_e8m0_row: tl.constexpr,
    stride_scale_e8m0_g: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    offs_n = tl.arange(0, BLOCK_N)
    state = tl.load(
        state_ptr + row * stride_state_row + offs_n * stride_state_n,
        mask=(row < total_rows) & (offs_n < dstate),
        other=0.0,
    ).to(tl.float32)
    for g in tl.static_range(0, NUM_GROUPS):
        g_mask = (offs_n >= g * GROUP_SIZE) & (offs_n < (g + 1) * GROUP_SIZE) & (offs_n < dstate)
        amax = tl.max(tl.where(g_mask, tl.abs(state), 0.0), axis=0)
        raw_exp = tl.ceil(tl.log(tl.maximum(amax / 448.0, 1.0e-30)) * 1.4426950408889634)
        biased_exp = tl.minimum(tl.maximum(raw_exp + 127.0, 1.0), 254.0)
        scale = tl.where(amax == 0.0, 0.0, tl.exp2(biased_exp - 127.0))
        q = tl.maximum(tl.minimum(state / scale, 448.0), -448.0)
        q = tl.where(amax == 0.0, 0.0, q)
        tl.store(
            q_ptr + row * stride_q_row + offs_n * stride_q_n,
            q,
            mask=(row < total_rows) & g_mask,
        )
        tl.store(
            scale_e8m0_ptr + row * stride_scale_e8m0_row + g * stride_scale_e8m0_g,
            biased_exp.to(tl.uint8),
            mask=row < total_rows,
        )


@triton.heuristics({"HAS_DT_BIAS": lambda args: args["dt_bias_ptr"] is not None})
@triton.heuristics({"HAS_D": lambda args: args["D_ptr"] is not None})
@triton.heuristics({"HAS_Z": lambda args: args["z_ptr"] is not None})
@triton.jit
def _mxfp8_selective_state_update_kernel(
    q_state_ptr,
    scale_e8m0_ptr,
    x_ptr,
    dt_ptr,
    dt_bias_ptr,
    A_ptr,
    B_ptr,
    C_ptr,
    D_ptr,
    z_ptr,
    out_ptr,
    batch: tl.constexpr,
    nheads: tl.constexpr,
    dim: tl.constexpr,
    dstate: tl.constexpr,
    nheads_ngroups_ratio: tl.constexpr,
    stride_q_batch: tl.constexpr,
    stride_q_head: tl.constexpr,
    stride_q_dim: tl.constexpr,
    stride_q_dstate: tl.constexpr,
    stride_scale_e8m0_batch: tl.constexpr,
    stride_scale_e8m0_head: tl.constexpr,
    stride_scale_e8m0_dim: tl.constexpr,
    stride_scale_e8m0_group: tl.constexpr,
    stride_x_batch: tl.constexpr,
    stride_x_head: tl.constexpr,
    stride_x_dim: tl.constexpr,
    stride_dt_batch: tl.constexpr,
    stride_dt_head: tl.constexpr,
    stride_dt_dim: tl.constexpr,
    stride_dt_bias_head: tl.constexpr,
    stride_dt_bias_dim: tl.constexpr,
    stride_A_head: tl.constexpr,
    stride_A_dim: tl.constexpr,
    stride_A_dstate: tl.constexpr,
    stride_B_batch: tl.constexpr,
    stride_B_group: tl.constexpr,
    stride_B_dstate: tl.constexpr,
    stride_C_batch: tl.constexpr,
    stride_C_group: tl.constexpr,
    stride_C_dstate: tl.constexpr,
    stride_D_head: tl.constexpr,
    stride_D_dim: tl.constexpr,
    stride_z_batch: tl.constexpr,
    stride_z_head: tl.constexpr,
    stride_z_dim: tl.constexpr,
    stride_out_batch: tl.constexpr,
    stride_out_head: tl.constexpr,
    stride_out_dim: tl.constexpr,
    DT_SOFTPLUS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_DSTATE: tl.constexpr,
    HAS_DT_BIAS: tl.constexpr,
    HAS_D: tl.constexpr,
    HAS_Z: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_DSTATE)

    q_base = q_state_ptr + pid_b * stride_q_batch + pid_h * stride_q_head
    scale_e8m0_base = scale_e8m0_ptr + pid_b * stride_scale_e8m0_batch + pid_h * stride_scale_e8m0_head
    x_base = x_ptr + pid_b * stride_x_batch + pid_h * stride_x_head
    dt_base = dt_ptr + pid_b * stride_dt_batch + pid_h * stride_dt_head
    A_base = A_ptr + pid_h * stride_A_head
    B_base = B_ptr + pid_b * stride_B_batch + (pid_h // nheads_ngroups_ratio) * stride_B_group
    C_base = C_ptr + pid_b * stride_C_batch + (pid_h // nheads_ngroups_ratio) * stride_C_group
    out_base = out_ptr + pid_b * stride_out_batch + pid_h * stride_out_head

    q_ptrs = q_base + offs_m[:, None] * stride_q_dim + offs_n[None, :] * stride_q_dstate
    group_ids = offs_n // GROUP_SIZE
    scale_e8m0_ptrs = (
        scale_e8m0_base
        + offs_m[:, None] * stride_scale_e8m0_dim
        + group_ids[None, :] * stride_scale_e8m0_group
    )
    mask = (offs_m[:, None] < dim) & (offs_n[None, :] < dstate)
    state = tl.load(q_ptrs, mask=mask, other=0.0).to(tl.float32)
    old_exp = tl.load(scale_e8m0_ptrs, mask=mask, other=127).to(tl.float32)
    old_scale = tl.exp2(old_exp - 127.0)
    state = state * old_scale

    x = tl.load(x_base + offs_m * stride_x_dim, mask=offs_m < dim, other=0.0).to(tl.float32)
    dt = tl.load(dt_base + offs_m * stride_dt_dim, mask=offs_m < dim, other=0.0).to(tl.float32)
    if HAS_DT_BIAS:
        dt_bias_base = dt_bias_ptr + pid_h * stride_dt_bias_head
        dt += tl.load(dt_bias_base + offs_m * stride_dt_bias_dim, mask=offs_m < dim, other=0.0).to(tl.float32)
    if DT_SOFTPLUS:
        dt = tl.where(dt <= 20.0, softplus(dt), dt)

    A = tl.load(
        A_base + offs_m[:, None] * stride_A_dim + offs_n[None, :] * stride_A_dstate,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    B = tl.load(B_base + offs_n * stride_B_dstate, mask=offs_n < dstate, other=0.0).to(tl.float32)
    C = tl.load(C_base + offs_n * stride_C_dstate, mask=offs_n < dstate, other=0.0).to(tl.float32)

    dA = tl.exp(A * dt[:, None])
    dB = B[None, :] * dt[:, None]
    state = state * dA + dB * x[:, None]

    out = tl.sum(state * C[None, :], axis=1)
    if HAS_D:
        D_base = D_ptr + pid_h * stride_D_head
        D = tl.load(D_base + offs_m * stride_D_dim, mask=offs_m < dim, other=0.0).to(tl.float32)
        out += x * D
    if HAS_Z:
        z_base = z_ptr + pid_b * stride_z_batch + pid_h * stride_z_head
        z = tl.load(z_base + offs_m * stride_z_dim, mask=offs_m < dim, other=0.0).to(tl.float32)
        out *= z * tl.sigmoid(z)
    tl.store(out_base + offs_m * stride_out_dim, out, mask=offs_m < dim)

    for g in tl.static_range(0, NUM_GROUPS):
        g_mask = (offs_n >= g * GROUP_SIZE) & (offs_n < (g + 1) * GROUP_SIZE) & (offs_n < dstate)
        amax = tl.max(tl.where(g_mask[None, :], tl.abs(state), 0.0), axis=1)
        raw_exp = tl.ceil(tl.log(tl.maximum(amax / 448.0, 1.0e-30)) * 1.4426950408889634)
        biased_exp = tl.minimum(tl.maximum(raw_exp + 127.0, 1.0), 254.0)
        new_scale = tl.where(amax == 0.0, 0.0, tl.exp2(biased_exp - 127.0))
        q = tl.maximum(tl.minimum(state / new_scale[:, None], 448.0), -448.0)
        q = tl.where(amax[:, None] == 0.0, 0.0, q)
        tl.store(q_ptrs, q, mask=mask & g_mask[None, :])
        tl.store(
            scale_e8m0_base + offs_m * stride_scale_e8m0_dim + g * stride_scale_e8m0_group,
            biased_exp.to(tl.uint8),
            mask=offs_m < dim,
        )


def allocate_mxfp8_state_cache(ssm_state: torch.Tensor, group_size: int = 32) -> MXFP8StateCache:
    if not hasattr(torch, "float8_e4m3fn"):
        raise RuntimeError("This PyTorch build does not expose torch.float8_e4m3fn.")
    if ssm_state.dim() != 4:
        raise ValueError(f"Expected SSM state [B,H,P,N], got {tuple(ssm_state.shape)}")
    num_groups = triton.cdiv(ssm_state.shape[-1], group_size)
    return MXFP8StateCache(
        q_state=torch.empty_like(ssm_state, dtype=torch.float8_e4m3fn),
        scale_e8m0=torch.empty(*ssm_state.shape[:-1], num_groups, device=ssm_state.device, dtype=torch.uint8),
        group_size=group_size,
        initialized=False,
    )


def quantize_state_into_cache(ssm_state: torch.Tensor, cache: MXFP8StateCache):
    b, h, p, n = ssm_state.shape
    block_n = triton.next_power_of_2(n)
    total_rows = b * h * p
    state_flat = ssm_state.reshape(total_rows, n)
    q_flat = cache.q_state.reshape(total_rows, n)
    scale_flat = cache.scale_e8m0.reshape(total_rows, cache.scale_e8m0.shape[-1])
    _quantize_state_kernel[(total_rows,)](
        state_flat,
        q_flat,
        scale_flat,
        total_rows,
        n,
        state_flat.stride(0),
        state_flat.stride(1),
        q_flat.stride(0),
        q_flat.stride(1),
        scale_flat.stride(0),
        scale_flat.stride(1),
        cache.group_size,
        cache.scale_e8m0.shape[-1],
        block_n,
        num_warps=4,
    )
    cache.initialized = True


def mxfp8_selective_state_update(
    cache: MXFP8StateCache,
    x,
    dt,
    A,
    B,
    C,
    D=None,
    z=None,
    dt_bias=None,
    dt_softplus=False,
):
    q_state = cache.q_state
    batch, nheads, dim, dstate = q_state.shape
    out = torch.empty_like(x)
    block_m = 32 if dstate <= 16 else 16 if dstate <= 32 else 8 if dstate <= 64 else 4
    block_n = triton.next_power_of_2(dstate)
    grid = (triton.cdiv(dim, block_m), batch, nheads)
    z_strides = z.stride() if z is not None else (0, 0, 0)
    d_strides = D.stride() if D is not None else (0, 0)
    dt_bias_strides = dt_bias.stride() if dt_bias is not None else (0, 0)
    with torch.cuda.device(x.device.index):
        _mxfp8_selective_state_update_kernel[grid](
            q_state,
            cache.scale_e8m0,
            x,
            dt,
            dt_bias,
            A,
            B,
            C,
            D,
            z,
            out,
            batch,
            nheads,
            dim,
            dstate,
            nheads // B.shape[1],
            q_state.stride(0),
            q_state.stride(1),
            q_state.stride(2),
            q_state.stride(3),
            cache.scale_e8m0.stride(0),
            cache.scale_e8m0.stride(1),
            cache.scale_e8m0.stride(2),
            cache.scale_e8m0.stride(3),
            x.stride(0),
            x.stride(1),
            x.stride(2),
            dt.stride(0),
            dt.stride(1),
            dt.stride(2),
            dt_bias_strides[0],
            dt_bias_strides[1],
            A.stride(0),
            A.stride(1),
            A.stride(2),
            B.stride(0),
            B.stride(1),
            B.stride(2),
            C.stride(0),
            C.stride(1),
            C.stride(2),
            d_strides[0],
            d_strides[1],
            z_strides[0],
            z_strides[1],
            z_strides[2],
            out.stride(0),
            out.stride(1),
            out.stride(2),
            dt_softplus,
            cache.group_size,
            cache.scale_e8m0.shape[-1],
            block_m,
            block_n,
            num_warps=4,
        )
    return out


@triton.heuristics({"HAS_D": lambda args: args["D_ptr"] is not None})
@triton.jit
def _mxfp8_selective_state_update_ablation_kernel(
    q_state_ptr,
    scale_e8m0_ptr,
    x_ptr,
    dt_ptr,
    A_ptr,
    B_ptr,
    C_ptr,
    D_ptr,
    out_ptr,
    batch: tl.constexpr,
    nheads: tl.constexpr,
    dim: tl.constexpr,
    dstate: tl.constexpr,
    nheads_ngroups_ratio: tl.constexpr,
    stride_q_batch: tl.constexpr,
    stride_q_head: tl.constexpr,
    stride_q_dim: tl.constexpr,
    stride_q_dstate: tl.constexpr,
    stride_scale_e8m0_batch: tl.constexpr,
    stride_scale_e8m0_head: tl.constexpr,
    stride_scale_e8m0_dim: tl.constexpr,
    stride_scale_e8m0_group: tl.constexpr,
    stride_x_batch: tl.constexpr,
    stride_x_head: tl.constexpr,
    stride_x_dim: tl.constexpr,
    stride_dt_batch: tl.constexpr,
    stride_dt_head: tl.constexpr,
    stride_dt_dim: tl.constexpr,
    stride_A_head: tl.constexpr,
    stride_A_dim: tl.constexpr,
    stride_A_dstate: tl.constexpr,
    stride_B_batch: tl.constexpr,
    stride_B_group: tl.constexpr,
    stride_B_dstate: tl.constexpr,
    stride_C_batch: tl.constexpr,
    stride_C_group: tl.constexpr,
    stride_C_dstate: tl.constexpr,
    stride_D_head: tl.constexpr,
    stride_D_dim: tl.constexpr,
    stride_out_batch: tl.constexpr,
    stride_out_head: tl.constexpr,
    stride_out_dim: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_DSTATE: tl.constexpr,
    MODE: tl.constexpr,
    HAS_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_DSTATE)

    q_base = q_state_ptr + pid_b * stride_q_batch + pid_h * stride_q_head
    scale_e8m0_base = scale_e8m0_ptr + pid_b * stride_scale_e8m0_batch + pid_h * stride_scale_e8m0_head
    x_base = x_ptr + pid_b * stride_x_batch + pid_h * stride_x_head
    dt_base = dt_ptr + pid_b * stride_dt_batch + pid_h * stride_dt_head
    A_base = A_ptr + pid_h * stride_A_head
    B_base = B_ptr + pid_b * stride_B_batch + (pid_h // nheads_ngroups_ratio) * stride_B_group
    C_base = C_ptr + pid_b * stride_C_batch + (pid_h // nheads_ngroups_ratio) * stride_C_group
    out_base = out_ptr + pid_b * stride_out_batch + pid_h * stride_out_head

    q_ptrs = q_base + offs_m[:, None] * stride_q_dim + offs_n[None, :] * stride_q_dstate
    group_ids = offs_n // GROUP_SIZE
    scale_e8m0_ptrs = (
        scale_e8m0_base
        + offs_m[:, None] * stride_scale_e8m0_dim
        + group_ids[None, :] * stride_scale_e8m0_group
    )
    mask = (offs_m[:, None] < dim) & (offs_n[None, :] < dstate)
    state = tl.load(q_ptrs, mask=mask, other=0.0).to(tl.float32)
    old_exp = tl.load(scale_e8m0_ptrs, mask=mask, other=127).to(tl.float32)
    state = state * tl.exp2(old_exp - 127.0)

    x = tl.load(x_base + offs_m * stride_x_dim, mask=offs_m < dim, other=0.0).to(tl.float32)
    if MODE == 0:
        out = tl.sum(state, axis=1) + x * 0.0
        tl.store(out_base + offs_m * stride_out_dim, out, mask=offs_m < dim)
        return

    dt = tl.load(dt_base + offs_m * stride_dt_dim, mask=offs_m < dim, other=0.0).to(tl.float32)
    A = tl.load(
        A_base + offs_m[:, None] * stride_A_dim + offs_n[None, :] * stride_A_dstate,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    B = tl.load(B_base + offs_n * stride_B_dstate, mask=offs_n < dstate, other=0.0).to(tl.float32)
    C = tl.load(C_base + offs_n * stride_C_dstate, mask=offs_n < dstate, other=0.0).to(tl.float32)

    state = state * tl.exp(A * dt[:, None]) + B[None, :] * dt[:, None] * x[:, None]
    out = tl.sum(state * C[None, :], axis=1)
    if HAS_D:
        D_base = D_ptr + pid_h * stride_D_head
        D = tl.load(D_base + offs_m * stride_D_dim, mask=offs_m < dim, other=0.0).to(tl.float32)
        out += x * D
    tl.store(out_base + offs_m * stride_out_dim, out, mask=offs_m < dim)

    if MODE == 1:
        return

    for g in tl.static_range(0, NUM_GROUPS):
        g_mask = (offs_n >= g * GROUP_SIZE) & (offs_n < (g + 1) * GROUP_SIZE) & (offs_n < dstate)
        amax = tl.max(tl.where(g_mask[None, :], tl.abs(state), 0.0), axis=1)
        raw_exp = tl.ceil(tl.log(tl.maximum(amax / 448.0, 1.0e-30)) * 1.4426950408889634)
        biased_exp = tl.minimum(tl.maximum(raw_exp + 127.0, 1.0), 254.0)
        new_scale = tl.where(amax == 0.0, 0.0, tl.exp2(biased_exp - 127.0))
        q = tl.maximum(tl.minimum(state / new_scale[:, None], 448.0), -448.0)
        q = tl.where(amax[:, None] == 0.0, 0.0, q)
        tl.store(q_ptrs, q, mask=mask & g_mask[None, :])
        tl.store(
            scale_e8m0_base + offs_m * stride_scale_e8m0_dim + g * stride_scale_e8m0_group,
            biased_exp.to(tl.uint8),
            mask=offs_m < dim,
        )


def mxfp8_selective_state_update_ablation(cache, x, dt, A, B, C, D=None, mode: str = "full"):
    q_state = cache.q_state
    batch, nheads, dim, dstate = q_state.shape
    out = torch.empty_like(x)
    block_m = 32 if dstate <= 16 else 16 if dstate <= 32 else 8 if dstate <= 64 else 4
    block_n = triton.next_power_of_2(dstate)
    grid = (triton.cdiv(dim, block_m), batch, nheads)
    d_strides = D.stride() if D is not None else (0, 0)
    mode_id = {"dequant_only": 0, "update_no_requant": 1, "full": 2}[mode]
    with torch.cuda.device(x.device.index):
        _mxfp8_selective_state_update_ablation_kernel[grid](
            q_state,
            cache.scale_e8m0,
            x,
            dt,
            A,
            B,
            C,
            D,
            out,
            batch,
            nheads,
            dim,
            dstate,
            nheads // B.shape[1],
            q_state.stride(0),
            q_state.stride(1),
            q_state.stride(2),
            q_state.stride(3),
            cache.scale_e8m0.stride(0),
            cache.scale_e8m0.stride(1),
            cache.scale_e8m0.stride(2),
            cache.scale_e8m0.stride(3),
            x.stride(0),
            x.stride(1),
            x.stride(2),
            dt.stride(0),
            dt.stride(1),
            dt.stride(2),
            A.stride(0),
            A.stride(1),
            A.stride(2),
            B.stride(0),
            B.stride(1),
            B.stride(2),
            C.stride(0),
            C.stride(1),
            C.stride(2),
            d_strides[0],
            d_strides[1],
            out.stride(0),
            out.stride(1),
            out.stride(2),
            cache.group_size,
            cache.scale_e8m0.shape[-1],
            block_m,
            block_n,
            mode_id,
            num_warps=4,
        )
    return out


def enable_mxfp8_fused_state_cache(model, group_size: int = 32):
    for layer in getattr(model.model, "layers", []):
        mamba = getattr(layer, "mamba", None)
        if mamba is None or getattr(mamba, "_mxfp8_fused_patched", False):
            continue
        mamba._mxfp8_fused_group_size = group_size
        mamba._mxfp8_fused_caches = {}
        original_step = mamba.step

        def make_step_with_mxfp8(base_step):
            def step_with_mxfp8(self, hidden_states, conv_state, ssm_state):
                if not getattr(self, "_mxfp8_fused_enabled", False):
                    return base_step(hidden_states, conv_state, ssm_state)
                if hidden_states.shape[1] != 1:
                    return base_step(hidden_states, conv_state, ssm_state)
                return _mxfp8_step(self, hidden_states, conv_state, ssm_state)

            return step_with_mxfp8

        mamba.step = MethodType(make_step_with_mxfp8(original_step), mamba)
        mamba._mxfp8_fused_patched = True


def initialize_mxfp8_fused_state_cache(model):
    for layer in getattr(model.model, "layers", []):
        mamba = getattr(layer, "mamba", None)
        if mamba is None or not getattr(mamba, "_mxfp8_fused_patched", False):
            continue
        mamba._mxfp8_fused_enabled = True
        mamba._mxfp8_fused_caches.clear()


def quantize_current_mamba_states(model, inference_params):
    for layer in getattr(model.model, "layers", []):
        mamba = getattr(layer, "mamba", None)
        if mamba is None or not getattr(mamba, "_mxfp8_fused_enabled", False):
            continue
        states = inference_params.key_value_memory_dict.get(mamba.layer_idx)
        if states is None:
            continue
        _, ssm_state = states
        cache = allocate_mxfp8_state_cache(ssm_state, group_size=mamba._mxfp8_fused_group_size)
        quantize_state_into_cache(ssm_state, cache)
        mamba._mxfp8_fused_caches[id(ssm_state)] = cache


def _mxfp8_step(self, hidden_states, conv_state, ssm_state):
    dtype = hidden_states.dtype
    zxbcdt = _profile_section(self, "mamba_in_proj_s", lambda: self.in_proj(hidden_states.squeeze(1)))
    d_mlp = (zxbcdt.shape[-1] - 2 * self.d_ssm - 2 * self.ngroups * self.d_state - self.nheads) // 2
    z0, x0, z, xBC, dt = _profile_section(
        self,
        "mamba_split_s",
        lambda: torch.split(
            zxbcdt,
            [d_mlp, d_mlp, self.d_ssm, self.d_ssm + 2 * self.ngroups * self.d_state, self.nheads],
            dim=-1,
        ),
    )
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update as _unused  # noqa: F401

    if self.conv1d is None:
        raise RuntimeError("Mamba2 conv1d module is required for fused MXFP8 decode.")
    from causal_conv1d import causal_conv1d_update

    xBC = _profile_section(
        self,
        "mamba_conv_update_s",
        lambda: causal_conv1d_update(
            xBC,
            conv_state,
            rearrange(self.conv1d.weight, "d 1 w -> d w"),
            self.conv1d.bias,
            self.activation,
        ),
    )

    def prep_update_args():
        x, b_mat, c_mat = torch.split(xBC, [self.d_ssm, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
        a_mat = -torch.exp(self.A_log.float())
        a_mat = repeat(a_mat, "h -> h p n", p=self.headdim, n=self.d_state).to(dtype=torch.float32)
        dt_mat = repeat(dt, "b h -> b h p", p=self.headdim)
        dt_bias_mat = repeat(self.dt_bias, "h -> h p", p=self.headdim)
        d_mat = repeat(self.D, "h -> h p", p=self.headdim)
        b_mat = rearrange(b_mat, "b (g n) -> b g n", g=self.ngroups)
        c_mat = rearrange(c_mat, "b (g n) -> b g n", g=self.ngroups)
        x_mat = rearrange(x, "b (h p) -> b h p", p=self.headdim)
        z_mat = None if self.rmsnorm else rearrange(z, "b (h p) -> b h p", p=self.headdim)
        return x, x_mat, dt_mat, a_mat, b_mat, c_mat, d_mat, z_mat, dt_bias_mat

    x, x_reshaped, dt, A, B, C, D, z_for_update, dt_bias = _profile_section(
        self, "mamba_update_prep_s", prep_update_args
    )

    def get_or_init_cache():
        cache = self._mxfp8_fused_caches.get(id(ssm_state))
        if cache is None or not cache.initialized:
            cache = allocate_mxfp8_state_cache(ssm_state, group_size=self._mxfp8_fused_group_size)
            quantize_state_into_cache(ssm_state, cache)
            self._mxfp8_fused_caches[id(ssm_state)] = cache
        return cache

    cache = _profile_section(self, "mamba_cache_lookup_s", get_or_init_cache)
    y = _profile_section(
        self,
        "mamba_mxfp8_state_update_kernel_s",
        lambda: mxfp8_selective_state_update(
            cache,
            x_reshaped,
            dt,
            A,
            B,
            C,
            D,
            z=z_for_update,
            dt_bias=dt_bias,
            dt_softplus=True,
        ),
    )
    y = rearrange(y, "b h p -> b (h p)")
    if self.rmsnorm:
        y = _profile_section(self, "mamba_rmsnorm_s", lambda: self.norm(y, z))
    if d_mlp > 0:
        y = _profile_section(self, "mamba_mlp_gate_cat_s", lambda: torch.cat([F.silu(z0) * x0, y], dim=-1))
    out = _profile_section(self, "mamba_out_proj_s", lambda: self.out_proj(y).unsqueeze(1))
    return out, conv_state, ssm_state
