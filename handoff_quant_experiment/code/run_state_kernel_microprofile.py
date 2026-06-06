from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import triton
import triton.language as tl
from einops import repeat
from mamba_ssm.ops.triton.selective_state_update import selective_state_update

from hybrid_quant.mxfp8_fused import (
    allocate_mxfp8_state_cache,
    mxfp8_selective_state_update,
    mxfp8_selective_state_update_ablation,
    quantize_state_into_cache,
)


@triton.heuristics({"HAS_D": lambda args: args["D_ptr"] is not None})
@triton.jit
def _normal_state_update_ablation_kernel(
    state_ptr,
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
    stride_state_batch: tl.constexpr,
    stride_state_head: tl.constexpr,
    stride_state_dim: tl.constexpr,
    stride_state_dstate: tl.constexpr,
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

    state_base = state_ptr + pid_b * stride_state_batch + pid_h * stride_state_head
    x_base = x_ptr + pid_b * stride_x_batch + pid_h * stride_x_head
    dt_base = dt_ptr + pid_b * stride_dt_batch + pid_h * stride_dt_head
    A_base = A_ptr + pid_h * stride_A_head
    B_base = B_ptr + pid_b * stride_B_batch + (pid_h // nheads_ngroups_ratio) * stride_B_group
    C_base = C_ptr + pid_b * stride_C_batch + (pid_h // nheads_ngroups_ratio) * stride_C_group
    out_base = out_ptr + pid_b * stride_out_batch + pid_h * stride_out_head

    state_ptrs = state_base + offs_m[:, None] * stride_state_dim + offs_n[None, :] * stride_state_dstate
    mask = (offs_m[:, None] < dim) & (offs_n[None, :] < dstate)
    state = tl.load(state_ptrs, mask=mask, other=0.0).to(tl.float32)
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
    if MODE == 2:
        tl.store(state_ptrs, state, mask=mask)


def normal_state_update_ablation(state, x, dt, A, B, C, D=None, mode: str = "full"):
    batch, nheads, dim, dstate = state.shape
    out = torch.empty_like(x)
    block_m = 32 if dstate <= 16 else 16 if dstate <= 32 else 8 if dstate <= 64 else 4
    block_n = triton.next_power_of_2(dstate)
    grid = (triton.cdiv(dim, block_m), batch, nheads)
    d_strides = D.stride() if D is not None else (0, 0)
    mode_id = {"read_only": 0, "update_no_writeback": 1, "full": 2}[mode]
    with torch.cuda.device(x.device.index):
        _normal_state_update_ablation_kernel[grid](
            state,
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
            state.stride(0),
            state.stride(1),
            state.stride(2),
            state.stride(3),
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
            block_m,
            block_n,
            mode_id,
            num_warps=4,
        )
    return out


def event_time_ms(fn, warmup: int, repeats: int):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(repeats):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return {
        "mean_ms": sum(times) / len(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "repeats": repeats,
    }


def make_inputs(batch: int, device: str, dtype: torch.dtype):
    # Nemotron-Flash-1B observed Mamba2 decode shape:
    # ssm_state: [B, nheads, headdim, d_state], nheads=64, headdim=64, d_state=128.
    nheads = 64
    headdim = 64
    dstate = 128
    ngroups = 1
    state = torch.randn(batch, nheads, headdim, dstate, device=device, dtype=dtype) * 0.01
    x = torch.randn(batch, nheads, headdim, device=device, dtype=dtype)
    dt = torch.rand(batch, nheads, headdim, device=device, dtype=dtype) * 0.01
    a = -torch.rand(nheads, headdim, dstate, device=device, dtype=torch.float32)
    b = torch.randn(batch, ngroups, dstate, device=device, dtype=dtype) * 0.01
    c = torch.randn(batch, ngroups, dstate, device=device, dtype=dtype) * 0.01
    d = torch.ones(nheads, headdim, device=device, dtype=dtype)
    dt_bias = torch.zeros(nheads, headdim, device=device, dtype=dtype)
    return state, x, dt, a, b, c, d, dt_bias


def profile_batch(batch: int, device: str, warmup: int, repeats: int, group_size: int):
    dtype = torch.bfloat16
    state, x, dt, a, b, c, d, dt_bias = make_inputs(batch, device, dtype)
    cache = allocate_mxfp8_state_cache(state, group_size=group_size)
    quantize_state_into_cache(state, cache)

    normal = event_time_ms(
        lambda: selective_state_update(
            state,
            x,
            dt,
            a,
            b,
            c,
            d,
            z=None,
            dt_bias=dt_bias,
            dt_softplus=True,
        ),
        warmup,
        repeats,
    )
    normal_read = event_time_ms(
        lambda: normal_state_update_ablation(state, x, dt, a, b, c, d, mode="read_only"),
        warmup,
        repeats,
    )
    normal_update_no_writeback = event_time_ms(
        lambda: normal_state_update_ablation(state, x, dt, a, b, c, d, mode="update_no_writeback"),
        warmup,
        repeats,
    )
    normal_full_ablation = event_time_ms(
        lambda: normal_state_update_ablation(state, x, dt, a, b, c, d, mode="full"),
        warmup,
        repeats,
    )
    fused_dequant = event_time_ms(
        lambda: mxfp8_selective_state_update_ablation(cache, x, dt, a, b, c, d, mode="dequant_only"),
        warmup,
        repeats,
    )
    fused_update_no_requant = event_time_ms(
        lambda: mxfp8_selective_state_update_ablation(cache, x, dt, a, b, c, d, mode="update_no_requant"),
        warmup,
        repeats,
    )
    fused_full_ablation = event_time_ms(
        lambda: mxfp8_selective_state_update_ablation(cache, x, dt, a, b, c, d, mode="full"),
        warmup,
        repeats,
    )
    fused_full_prod = event_time_ms(
        lambda: mxfp8_selective_state_update(cache, x, dt, a, b, c, d, z=None, dt_bias=dt_bias, dt_softplus=True),
        warmup,
        repeats,
    )

    dequant_ms = fused_dequant["mean_ms"]
    update_compute_ms = max(fused_update_no_requant["mean_ms"] - fused_dequant["mean_ms"], 0.0)
    requant_ms = max(fused_full_ablation["mean_ms"] - fused_update_no_requant["mean_ms"], 0.0)
    total = fused_full_ablation["mean_ms"]
    normal_read_ms = normal_read["mean_ms"]
    normal_update_ms = max(normal_update_no_writeback["mean_ms"] - normal_read["mean_ms"], 0.0)
    normal_writeback_ms = max(normal_full_ablation["mean_ms"] - normal_update_no_writeback["mean_ms"], 0.0)
    normal_ablation_total = normal_full_ablation["mean_ms"]
    return {
        "batch_size": batch,
        "shape": {
            "state": list(state.shape),
            "x": list(x.shape),
            "A": list(a.shape),
            "B": list(b.shape),
            "C": list(c.shape),
        },
        "normal_selective_state_update": normal,
        "normal_read_only": normal_read,
        "normal_update_no_writeback": normal_update_no_writeback,
        "normal_full_ablation": normal_full_ablation,
        "normal_estimated_breakdown": {
            "state_read_ms": normal_read_ms,
            "update_and_output_ms": normal_update_ms,
            "state_writeback_ms": normal_writeback_ms,
            "state_read_pct": normal_read_ms / normal_ablation_total * 100.0 if normal_ablation_total else None,
            "update_and_output_pct": normal_update_ms / normal_ablation_total * 100.0 if normal_ablation_total else None,
            "state_writeback_pct": normal_writeback_ms / normal_ablation_total * 100.0 if normal_ablation_total else None,
        },
        "mxfp8_fused_dequant_only": fused_dequant,
        "mxfp8_fused_update_no_requant": fused_update_no_requant,
        "mxfp8_fused_full_ablation": fused_full_ablation,
        "mxfp8_fused_full_production": fused_full_prod,
        "mxfp8_fused_estimated_breakdown": {
            "dequant_read_ms": dequant_ms,
            "update_and_output_ms": update_compute_ms,
            "requant_writeback_ms": requant_ms,
            "dequant_read_pct": dequant_ms / total * 100.0 if total else None,
            "update_and_output_pct": update_compute_ms / total * 100.0 if total else None,
            "requant_writeback_pct": requant_ms / total * 100.0 if total else None,
        },
        "fused_vs_normal_pct": (
            (fused_full_prod["mean_ms"] - normal["mean_ms"]) / normal["mean_ms"] * 100.0
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-sizes", default="32,64")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    batches = [int(x) for x in args.batch_sizes.split(",") if x.strip()]
    results = [profile_batch(bs, args.device, args.warmup, args.repeats, args.group_size) for bs in batches]
    out = {
        "note": "Kernel microprofile using CUDA events. MXFP8 internal percentages are ablation/difference estimates, not source-line hardware counters.",
        "group_size": args.group_size,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "results": results,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
