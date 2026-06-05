from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from itertools import product
from pathlib import Path


CONFIGS = list(product(["normal", "fp8", "int4"], ["normal", "mxfp8"]))


def parse_gpu_list(value: str):
    return [x.strip() for x in value.split(",") if x.strip()]


def read_jsonl(path: Path):
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/home/vrintern/tmp/models/Nemotron-Flash-1B")
    parser.add_argument("--dataset-path", default="/home/vrintern/tmp/datasets/wikitext")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--result-json", default="results/all_results.json")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--metrics", default="ppl,latency")
    parser.add_argument("--context-lengths", default="2048,8192")
    parser.add_argument("--batch-sizes", default="1,32")
    parser.add_argument("--decode-length", type=int, default=256)
    parser.add_argument("--ppl-max-eval-tokens", type=int, default=8192)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--latency-warmup", type=int, default=1)
    parser.add_argument("--latency-repeats", type=int, default=3)
    parser.add_argument("--mamba-group-size", type=int, default=32)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    gpus = parse_gpu_list(args.gpus)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    procs = []
    worker_outputs = []
    for idx, (kv_mode, mamba_mode) in enumerate(CONFIGS):
        gpu = gpus[idx % len(gpus)]
        worker_out = out_dir / f"worker_{idx}_{kv_mode}_{mamba_mode}.jsonl"
        if worker_out.exists():
            worker_out.unlink()
        worker_outputs.append(worker_out)
        worker_args = [
            "--model-path",
            args.model_path,
            "--dataset-path",
            args.dataset_path,
            "--output",
            str(worker_out),
            "--kv-mode",
            kv_mode,
            "--mamba-mode",
            mamba_mode,
            "--metrics",
            args.metrics,
            "--context-lengths",
            args.context_lengths,
            "--batch-sizes",
            args.batch_sizes,
            "--decode-length",
            str(args.decode_length),
            "--ppl-max-eval-tokens",
            str(args.ppl_max_eval_tokens),
            "--dtype",
            args.dtype,
            "--device",
            f"cuda:{gpu}",
            "--latency-warmup",
            str(args.latency_warmup),
            "--latency-repeats",
            str(args.latency_repeats),
            "--mamba-group-size",
            str(args.mamba_group_size),
        ]
        cmd = [
            args.python,
            "-c",
            "import sys, run_worker; sys.argv=['run_worker.py']+sys.argv[1:]; run_worker.main()",
            *worker_args,
        ]
        env = os.environ.copy()
        env.pop("CUDA_VISIBLE_DEVICES", None)
        env["HF_HOME"] = "/home/vrintern/tmp/.hf-cache"
        env["HF_XET_CACHE"] = "/home/vrintern/tmp/.hf-xet-cache"
        env["HF_MODULES_CACHE"] = "/home/vrintern/tmp/.hf-modules-cache"
        env["HF_DATASETS_CACHE"] = "/home/vrintern/tmp/.hf-cache/datasets"
        env["XDG_CACHE_HOME"] = "/home/vrintern/tmp/.cache"
        print(f"[launch] gpu={gpu} kv={kv_mode} mamba={mamba_mode} output={worker_out}", flush=True)
        if args.dry_run:
            print(" ".join(cmd))
            continue
        procs.append((kv_mode, mamba_mode, subprocess.Popen(cmd, env=env, cwd=Path(__file__).parent)))

    failures = []
    for kv_mode, mamba_mode, proc in procs:
        code = proc.wait()
        if code != 0:
            failures.append({"kv_mode": kv_mode, "mamba_mode": mamba_mode, "returncode": code})

    rows = []
    for path in worker_outputs:
        rows.extend(read_jsonl(path))

    result_path = Path(args.result_json)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_unix": time.time(),
        "configs": [{"kv_mode": kv, "mamba_state_mode": mm} for kv, mm in CONFIGS],
        "failures": failures,
        "results": rows,
    }
    result_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] wrote {result_path} with {len(rows)} result rows", flush=True)
    if failures:
        print(f"[warn] failures: {failures}", flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
