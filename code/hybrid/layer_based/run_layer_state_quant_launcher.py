from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


BASELINE_LAYER = -1


def parse_csv(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def parse_layers(value: str) -> list[int] | None:
    if value.strip().lower() == "auto":
        return None
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_summary(summary_json: Path, summary_md: Path, rows: list[dict[str, Any]], failures: list[dict[str, Any]]):
    baseline = next((r for r in rows if r.get("layer_index") == BASELINE_LAYER), None)
    baseline_ppl = baseline.get("ppl") if baseline else None
    sorted_rows = sorted(rows, key=lambda r: (r.get("layer_index") != BASELINE_LAYER, r.get("layer_index", 10**9)))
    for row in sorted_rows:
        if baseline_ppl is not None and row.get("ppl") is not None:
            row["delta_ppl_vs_baseline"] = row["ppl"] - baseline_ppl
            row["relative_ppl_vs_baseline_pct"] = (row["ppl"] / baseline_ppl - 1.0) * 100.0

    payload = {
        "created_unix": time.time(),
        "baseline_ppl": baseline_ppl,
        "failures": failures,
        "results": sorted_rows,
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "| layer_index | quantization | ppl | delta_ppl | relative_delta_% | tokens | elapsed_s | device |",
        "|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in sorted_rows:
        layer = "baseline" if row.get("layer_index") == BASELINE_LAYER else str(row.get("layer_index"))
        delta = row.get("delta_ppl_vs_baseline")
        rel = row.get("relative_ppl_vs_baseline_pct")
        lines.append(
            "| {layer} | {quant} | {ppl:.6f} | {delta} | {rel} | {tokens} | {elapsed:.2f} | {device} |".format(
                layer=layer,
                quant=row.get("quantization"),
                ppl=row.get("ppl", float("nan")),
                delta="" if delta is None else f"{delta:.6f}",
                rel="" if rel is None else f"{rel:.4f}",
                tokens=row.get("tokens", ""),
                elapsed=row.get("elapsed_s", 0.0),
                device=row.get("device", ""),
            )
        )
    if failures:
        lines.extend(["", "Failures:", ""])
        for failure in failures:
            lines.append(f"- layer={failure.get('layer_index')} returncode={failure.get('returncode')}")
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_probe(args, worker: Path, out_dir: Path, gpu: str):
    probe_out = (out_dir / "probe_layers.json").resolve()
    cmd = [
        args.python,
        str(worker),
        "--model-path",
        args.model_path,
        "--dataset-path",
        args.dataset_path,
        "--output",
        str(probe_out),
        "--list-layers",
        "--device",
        f"cuda:{gpu}" if args.use_cuda_devices else gpu,
        "--dtype",
        args.dtype,
        "--attn-implementation",
        args.attn_implementation,
    ]
    print(f"[probe] {' '.join(cmd)}", flush=True)
    if args.dry_run:
        return []
    subprocess.run(cmd, check=True, cwd=worker.parent)
    return read_json(probe_out)["mamba_layers"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/home/vrintern/tmp/models/Nemotron-Flash-1B")
    parser.add_argument("--dataset-path", default="/home/vrintern/tmp/datasets/wikitext")
    parser.add_argument("--out-dir", default="results/layer_based/nemotron_1b_mxfp8_state")
    parser.add_argument("--summary-json", default="results/layer_based/nemotron_1b_mxfp8_state_summary.json")
    parser.add_argument("--summary-md", default="results/layer_based/nemotron_1b_mxfp8_state_summary.md")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--layers", default="auto")
    parser.add_argument("--quant-backend", choices=["mxfp8_sr", "mxfp8_fused"], default="mxfp8_sr")
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--decode-length", type=int, default=256)
    parser.add_argument("--ppl-windows", type=int, default=2)
    parser.add_argument("--prefill-chunk-size", type=int, default=256)
    parser.add_argument("--mamba-group-size", type=int, default=32)
    parser.add_argument("--ppl-split", default="test")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--include-baseline", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-cuda-devices", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    python_path = Path(args.python)
    if not python_path.is_absolute():
        python_path = Path.cwd() / python_path
    args.python = str(python_path)

    worker = Path(__file__).resolve().parent / "run_layer_state_quant_worker.py"
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    gpus = parse_csv(args.gpus)
    if not gpus:
        raise ValueError("--gpus must contain at least one device id.")

    layers = parse_layers(args.layers)
    if layers is None:
        layers = run_probe(args, worker, out_dir, gpus[0])
    tasks = []
    if args.include_baseline:
        tasks.append(BASELINE_LAYER)
    tasks.extend(layers)

    specs = []
    for idx, layer_index in enumerate(tasks):
        gpu = gpus[idx % len(gpus)]
        name = "baseline" if layer_index == BASELINE_LAYER else f"layer_{layer_index}"
        output = (out_dir / f"{idx:03d}_{name}.json").resolve()
        ready = (out_dir / f"{idx:03d}_{name}.ready.json").resolve()
        if output.exists():
            output.unlink()
        if ready.exists():
            ready.unlink()
        device = f"cuda:{gpu}" if args.use_cuda_devices else gpu
        cmd = [
            args.python,
            str(worker),
            "--model-path",
            args.model_path,
            "--dataset-path",
            args.dataset_path,
            "--output",
            str(output),
            "--ready-file",
            str(ready),
            "--layer-index",
            str(layer_index),
            "--quant-backend",
            args.quant_backend,
            "--context-length",
            str(args.context_length),
            "--decode-length",
            str(args.decode_length),
            "--ppl-windows",
            str(args.ppl_windows),
            "--prefill-chunk-size",
            str(args.prefill_chunk_size),
            "--mamba-group-size",
            str(args.mamba_group_size),
            "--ppl-split",
            args.ppl_split,
            "--dtype",
            args.dtype,
            "--attn-implementation",
            args.attn_implementation,
            "--seed",
            str(args.seed),
            "--device",
            device,
        ]
        env = os.environ.copy()
        env.pop("CUDA_VISIBLE_DEVICES", None)
        env["HF_HOME"] = "/home/vrintern/tmp/.hf-cache"
        env["HF_XET_CACHE"] = "/home/vrintern/tmp/.hf-xet-cache"
        env["HF_MODULES_CACHE"] = "/home/vrintern/tmp/.hf-modules-cache"
        env["HF_DATASETS_CACHE"] = "/home/vrintern/tmp/.hf-cache/datasets"
        env["XDG_CACHE_HOME"] = "/home/vrintern/tmp/.cache"
        specs.append({"layer_index": layer_index, "cmd": cmd, "env": env, "output": output, "ready": ready, "gpu": gpu})
        print(f"[task] gpu={gpu} layer={layer_index} output={output}", flush=True)
        if args.dry_run:
            print(" ".join(cmd), flush=True)

    if args.dry_run:
        return

    failures = []
    running = []
    pending = list(specs)
    max_concurrent = min(len(gpus), len(pending)) if pending else 0
    while pending or running:
        while pending and len(running) < max_concurrent:
            spec = pending.pop(0)
            proc = subprocess.Popen(spec["cmd"], env=spec["env"], cwd=worker.parent)
            running.append((spec, proc))
            print(f"[launch] gpu={spec['gpu']} layer={spec['layer_index']} pid={proc.pid}", flush=True)
        still_running = []
        for spec, proc in running:
            code = proc.poll()
            if code is None:
                still_running.append((spec, proc))
            elif code != 0:
                failures.append({"layer_index": spec["layer_index"], "returncode": code, "output": str(spec["output"])})
                print(f"[fail] layer={spec['layer_index']} returncode={code}", flush=True)
            else:
                print(f"[done-task] layer={spec['layer_index']} output={spec['output']}", flush=True)
        running = still_running
        if running:
            time.sleep(2.0)

    rows = []
    for spec in specs:
        if spec["output"].exists():
            rows.append(read_json(spec["output"]))
    write_summary(Path(args.summary_json).resolve(), Path(args.summary_md).resolve(), rows, failures)
    print(f"[done] wrote {args.summary_json} and {args.summary_md}", flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
