from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
import sys

_CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_CODE_ROOT / "shared"))


def parse_ints(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def build_tasks(args):
    tasks = []
    modes = ["normal", "mxfp8_fused", "mxfp8_refresh256"]
    task_id = 0
    for mode in modes:
        for ctx in parse_ints(args.context_lengths):
            tasks.append(
                {
                    "id": task_id,
                    "mode": mode,
                    "metric": "ppl",
                    "context_length": ctx,
                    "batch_size": None,
                }
            )
            task_id += 1
            for bs in parse_ints(args.batch_sizes):
                tasks.append(
                    {
                        "id": task_id,
                        "mode": mode,
                        "metric": "latency",
                        "context_length": ctx,
                        "batch_size": bs,
                    }
                )
                task_id += 1
    return tasks


def task_command(args, task, gpu: int, output: Path):
    argv = [
        "run_decode_ppl_latency_mamba_refresh.py",
        "--mode",
        task["mode"],
        "--metrics",
        task["metric"],
        "--context-lengths",
        str(task["context_length"]),
        "--decode-length",
        str(args.decode_length),
        "--refresh-interval",
        str(args.refresh_interval),
        "--ppl-windows",
        str(args.ppl_windows),
        "--prefill-chunk-size",
        str(args.prefill_chunk_size),
        "--latency-warmup",
        str(args.latency_warmup),
        "--latency-repeats",
        str(args.latency_repeats),
        "--output",
        str(output),
        "--device",
        f"cuda:{gpu}",
        "--dtype",
        args.dtype,
        "--attn-implementation",
        args.attn_implementation,
    ]
    if task["metric"] == "latency":
        argv.extend(["--batch-sizes", str(task["batch_size"])])
    module_dir = Path(__file__).resolve().parent
    code = (
        "import sys; sys.path.insert(0, %r); "
        "import run_decode_ppl_latency_mamba_refresh as r; sys.argv=%r; r.main()"
    ) % (str(module_dir), argv)
    return [sys.executable, "-c", code]


def aggregate(task_dir: Path, summary_path: Path):
    rows = []
    for path in sorted(task_dir.glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    payload = {"results": rows}
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results/decode_refresh_3mode")
    parser.add_argument("--summary", default="results/decode_refresh_3mode_summary.json")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--context-lengths", default="2048,8192")
    parser.add_argument("--batch-sizes", default="1,32")
    parser.add_argument("--decode-length", type=int, default=1000)
    parser.add_argument("--refresh-interval", type=int, default=256)
    parser.add_argument("--ppl-windows", type=int, default=2)
    parser.add_argument("--prefill-chunk-size", type=int, default=256)
    parser.add_argument("--latency-warmup", type=int, default=1)
    parser.add_argument("--latency-repeats", type=int, default=2)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = build_tasks(args)
    gpus = parse_ints(args.gpus)
    pending = list(tasks)
    running = {}
    completed = []
    failed = []

    while pending or running:
        for gpu in gpus:
            if gpu in running or not pending:
                continue
            task = pending.pop(0)
            output = output_dir / f"{task['id']:02d}_{task['metric']}_{task['mode']}_ctx{task['context_length']}"
            if task["batch_size"] is not None:
                output = Path(str(output) + f"_bs{task['batch_size']}")
            output = output.with_suffix(".jsonl")
            log = output.with_suffix(".log")
            cmd = task_command(args, task, gpu, output)
            with log.open("w", encoding="utf-8") as log_f:
                proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT)
            running[gpu] = (proc, task, output, log, time.time())
            print(json.dumps({"event": "start", "gpu": gpu, **task, "output": str(output)}), flush=True)

        time.sleep(5)
        for gpu, (proc, task, output, log, start) in list(running.items()):
            rc = proc.poll()
            if rc is None:
                continue
            elapsed = time.time() - start
            row = {"event": "done", "gpu": gpu, "returncode": rc, "elapsed_s": elapsed, **task, "output": str(output)}
            print(json.dumps(row), flush=True)
            if rc == 0:
                completed.append(task)
            else:
                failed.append({**task, "returncode": rc, "log": str(log)})
            del running[gpu]
            aggregate(output_dir, Path(args.summary))

    rows = aggregate(output_dir, Path(args.summary))
    final = {"completed": len(completed), "failed": failed, "rows": len(rows), "summary": args.summary}
    (output_dir / "launcher_status.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    print(json.dumps({"event": "finished", **final}, ensure_ascii=False), flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
