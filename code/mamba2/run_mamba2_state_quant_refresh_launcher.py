from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
import sys

_CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_CODE_ROOT / "shared"))
sys.path.insert(0, str(_CODE_ROOT / "hybrid" / "chunk_update"))


def parse_csv(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def parse_ints(value: str) -> list[int]:
    return [int(x) for x in parse_csv(value)]


def build_tasks(args):
    tasks = []
    tid = 0
    for ctx in parse_ints(args.context_lengths):
        for mode in parse_csv(args.modes):
            tasks.append({"id": tid, "metric": "ppl", "ctx": ctx, "mode": mode, "bs": None})
            tid += 1
        for bs in parse_ints(args.batch_sizes):
            for mode in parse_csv(args.modes):
                tasks.append({"id": tid, "metric": "latency", "ctx": ctx, "mode": mode, "bs": bs})
                tid += 1
        tasks.append({"id": tid, "metric": "memory", "ctx": ctx, "mode": ",".join(parse_csv(args.modes)), "bs": args.memory_batch_size})
        tid += 1
    return tasks


def command(args, task, gpu: int, output: Path, output_dir: Path):
    metrics = task["metric"]
    argv = [
        "run_mamba2_state_quant_refresh.py",
        "--metrics",
        metrics,
        "--modes",
        task["mode"],
        "--context-lengths",
        str(task["ctx"]),
        "--decode-length",
        str(args.decode_length),
        "--refresh-interval",
        str(args.refresh_interval),
        "--ppl-windows",
        str(args.ppl_windows),
        "--latency-warmup",
        str(args.latency_warmup),
        "--latency-repeats",
        str(args.latency_repeats),
        "--memory-batch-size",
        str(task["bs"] if metrics == "memory" else args.memory_batch_size),
        "--output",
        str(output),
        "--output-dir",
        str(output_dir),
        "--device",
        f"cuda:{gpu}",
    ]
    if metrics == "latency":
        argv.extend(["--batch-sizes", str(task["bs"])])
    module_dir = Path(__file__).resolve().parent
    code = (
        "import sys; sys.path.insert(0, %r); "
        "import run_mamba2_state_quant_refresh as r; sys.argv=%r; r.main()"
    ) % (str(module_dir), argv)
    return [sys.executable, "-c", code]


def read_rows(task_dir: Path):
    rows = []
    for path in sorted(task_dir.glob("*.json")):
        if path.name in {"summary.json", "launcher_status.json"}:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        rows.extend(payload.get("results", []))
    return rows


def aggregate(task_dir: Path, summary: Path):
    rows = read_rows(task_dir)
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps({"results": rows}, indent=2), encoding="utf-8")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results/mamba2_1_3b_quant_refresh")
    parser.add_argument("--summary", default="results/mamba2_1_3b_quant_refresh_summary.json")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--modes", default="normal,mxfp8_fused,mxfp8_refresh256")
    parser.add_argument("--context-lengths", default="2048,8192")
    parser.add_argument("--batch-sizes", default="1,32")
    parser.add_argument("--memory-batch-size", type=int, default=32)
    parser.add_argument("--decode-length", type=int, default=1000)
    parser.add_argument("--refresh-interval", type=int, default=256)
    parser.add_argument("--ppl-windows", type=int, default=2)
    parser.add_argument("--latency-warmup", type=int, default=1)
    parser.add_argument("--latency-repeats", type=int, default=2)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    task_dir = output_dir / "tasks"
    plot_dir = output_dir / "plots"
    task_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
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
            stem = f"{task['id']:02d}_{task['metric']}_ctx{task['ctx']}_{task['mode'].replace(',', '-')}"
            if task["bs"] is not None:
                stem += f"_bs{task['bs']}"
            output = task_dir / f"{stem}.json"
            log = task_dir / f"{stem}.log"
            cmd = command(args, task, gpu, output, plot_dir / stem)
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
            print(json.dumps({"event": "done", "gpu": gpu, "returncode": rc, "elapsed_s": elapsed, **task}), flush=True)
            if rc == 0:
                completed.append(task)
            else:
                failed.append({**task, "returncode": rc, "log": str(log)})
            del running[gpu]
            aggregate(task_dir, Path(args.summary))
    rows = aggregate(task_dir, Path(args.summary))
    status = {"completed": len(completed), "failed": failed, "rows": len(rows), "summary": args.summary}
    (output_dir / "launcher_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps({"event": "finished", **status}), flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
