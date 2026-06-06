from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
import sys

_CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_CODE_ROOT / "shared"))


def parse_csv(value: str):
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
    parser.add_argument("--out-dir", default="results/mamba_state_only")
    parser.add_argument("--result-json", default="results/mamba_state_only_results.json")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
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

    gpus = parse_csv(args.gpus)
    contexts = parse_csv(args.context_lengths)
    batches = parse_csv(args.batch_sizes)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    for mamba_mode in ("normal", "mxfp8"):
        for ctx in contexts:
            tasks.append(
                {
                    "name": f"ppl_{mamba_mode}_ctx{ctx}",
                    "mamba_mode": mamba_mode,
                    "metrics": "ppl",
                    "context_lengths": ctx,
                    "batch_sizes": "1",
                }
            )
        for ctx in contexts:
            for batch in batches:
                tasks.append(
                    {
                        "name": f"latency_{mamba_mode}_ctx{ctx}_bs{batch}",
                        "mamba_mode": mamba_mode,
                        "metrics": "latency",
                        "context_lengths": ctx,
                        "batch_sizes": batch,
                    }
                )

    worker_outputs = []
    launch_specs = []
    for idx, task in enumerate(tasks):
        gpu = gpus[idx % len(gpus)]
        worker_out = out_dir / f"{idx:02d}_{task['name']}.jsonl"
        ready_file = out_dir / f"{idx:02d}_{task['name']}.ready.json"
        if worker_out.exists():
            worker_out.unlink()
        if ready_file.exists():
            ready_file.unlink()
        worker_outputs.append(worker_out)
        worker_args = [
            "--model-path",
            args.model_path,
            "--dataset-path",
            args.dataset_path,
            "--output",
            str(worker_out),
            "--ready-file",
            str(ready_file),
            "--kv-mode",
            "normal",
            "--mamba-mode",
            task["mamba_mode"],
            "--metrics",
            task["metrics"],
            "--context-lengths",
            task["context_lengths"],
            "--batch-sizes",
            task["batch_sizes"],
            "--decode-length",
            str(args.decode_length),
            "--ppl-max-eval-tokens",
            str(args.ppl_max_eval_tokens),
            "--dtype",
            args.dtype,
            "--device",
            f"cuda:{gpu}",
            "--attn-implementation",
            "sdpa",
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
            (
                "import sys; sys.path.insert(0, %r); "
                "import run_worker; sys.argv=['run_worker.py']+sys.argv[1:]; run_worker.main()"
            )
            % str(Path(__file__).resolve().parent),
            *worker_args,
        ]
        env = os.environ.copy()
        env.pop("CUDA_VISIBLE_DEVICES", None)
        env["HF_HOME"] = "/home/vrintern/tmp/.hf-cache"
        env["HF_XET_CACHE"] = "/home/vrintern/tmp/.hf-xet-cache"
        env["HF_MODULES_CACHE"] = "/home/vrintern/tmp/.hf-modules-cache"
        env["HF_DATASETS_CACHE"] = "/home/vrintern/tmp/.hf-cache/datasets"
        env["XDG_CACHE_HOME"] = "/home/vrintern/tmp/.cache"
        print(f"[launch] gpu={gpu} {task['name']} output={worker_out}", flush=True)
        if args.dry_run:
            print(" ".join(cmd), flush=True)
            continue
        launch_specs.append((task, cmd, env, ready_file))

    failures = []
    running = []
    pending = list(launch_specs)
    max_concurrent = min(len(gpus), len(pending)) if pending else 0
    while pending or running:
        while pending and len(running) < max_concurrent:
            task, cmd, env, ready_file = pending.pop(0)
            proc = subprocess.Popen(cmd, env=env, cwd=Path(__file__).parent)
            running.append((task, proc))
            start_wait = time.time()
            while not ready_file.exists() and proc.poll() is None and time.time() - start_wait < 120:
                time.sleep(1.0)
        still_running = []
        for task, proc in running:
            code = proc.poll()
            if code is None:
                still_running.append((task, proc))
            elif code != 0:
                failures.append({"task": task["name"], "returncode": code})
        running = still_running
        if running:
            time.sleep(2.0)

    rows = []
    for path in worker_outputs:
        rows.extend(read_jsonl(path))

    result_path = Path(args.result_json)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_unix": time.time(),
        "scope": "kv_normal_mamba_state_normal_vs_mxfp8",
        "tasks": tasks,
        "failures": failures,
        "results": rows,
    }
    result_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] wrote {result_path} with {len(rows)} rows", flush=True)
    if failures:
        print(f"[warn] failures: {failures}", flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
