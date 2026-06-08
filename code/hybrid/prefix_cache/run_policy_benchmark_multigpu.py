from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=int, nargs="+", default=list(range(8)))
    parser.add_argument("--trace-source", choices=["synthetic", "ultrachat"], default="synthetic")
    parser.add_argument("--ultrachat-path", default="datasets/ultrachat_200k/train_sft")
    parser.add_argument("--num-requests", type=int, default=10000)
    parser.add_argument("--capacity-mib", type=float, nargs="+", required=True)
    parser.add_argument("--policies", nargs="+", default=["marconi", "leaf_branch_quant"])
    parser.add_argument("--progress-every", type=int, default=5000)
    parser.add_argument("--output-dir", type=Path, default=Path("results/hybrid/prefix_cache/multigpu_controlled"))
    parser.add_argument("--summary-output", type=Path, default=Path("results/hybrid/prefix_cache/multigpu_controlled_summary.json"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)

    tasks = []
    for capacity in args.capacity_mib:
        for policy in args.policies:
            tasks.append((capacity, policy))

    running: list[tuple[subprocess.Popen, Path, float, str, float, int]] = []
    completed = []
    next_task = 0

    while next_task < len(tasks) or running:
        while next_task < len(tasks) and len(running) < len(args.gpus):
            capacity, policy = tasks[next_task]
            gpu = args.gpus[len(running) % len(args.gpus)]
            stem = f"{args.trace_source}_req{args.num_requests}_{int(capacity)}m_{policy}_cuda{gpu}"
            output = args.output_dir / f"{stem}.json"
            log = args.output_dir / f"{stem}.log"
            cmd = [
                sys.executable,
                "-c",
                (
                    "import sys; "
                    "sys.path.insert(0, 'code/hybrid/prefix_cache'); "
                    "import run_policy_benchmark as m; "
                    f"sys.argv={repr(['run_policy_benchmark.py', '--device', f'cuda:{gpu}', '--trace-source', args.trace_source, '--num-requests', str(args.num_requests), '--skip-unlimited', '--no-budget-ratios', '--capacity-mib', str(capacity), '--policies', policy, '--progress-every', str(args.progress_every), '--incremental-output', '--output', str(output)])}; "
                    "m.main()"
                ),
            ]
            if args.trace_source == "ultrachat":
                argv = eval(cmd[2].split("sys.argv=", 1)[1].split("; m.main()", 1)[0])
                insert_at = argv.index("--num-requests")
                argv[insert_at:insert_at] = ["--stream-trace", "--ultrachat-path", args.ultrachat_path]
                cmd[2] = (
                    "import sys; "
                    "sys.path.insert(0, 'code/hybrid/prefix_cache'); "
                    "import run_policy_benchmark as m; "
                    f"sys.argv={repr(argv)}; "
                    "m.main()"
                )
            lf = log.open("w")
            proc = subprocess.Popen(cmd, cwd=Path.cwd(), stdout=lf, stderr=subprocess.STDOUT)
            lf.close()
            running.append((proc, output, time.time(), policy, capacity, gpu))
            print(f"launched pid={proc.pid} gpu={gpu} capacity={capacity} policy={policy} output={output}", flush=True)
            next_task += 1

        still_running = []
        for proc, output, start, policy, capacity, gpu in running:
            rc = proc.poll()
            if rc is None:
                still_running.append((proc, output, start, policy, capacity, gpu))
                continue
            completed.append(
                {
                    "pid": proc.pid,
                    "returncode": rc,
                    "output": str(output),
                    "policy": policy,
                    "capacity_mib": capacity,
                    "gpu": gpu,
                    "elapsed_s": time.time() - start,
                }
            )
            print(f"finished pid={proc.pid} rc={rc} gpu={gpu} capacity={capacity} policy={policy}", flush=True)
        running = still_running
        args.summary_output.write_text(json.dumps({"tasks": completed, "running": len(running)}, indent=2), encoding="utf-8")
        if running or next_task < len(tasks):
            time.sleep(5)

    args.summary_output.write_text(json.dumps({"tasks": completed, "running": 0}, indent=2), encoding="utf-8")
    print(f"summary={args.summary_output}")


if __name__ == "__main__":
    main()
