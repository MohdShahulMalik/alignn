from __future__ import annotations

import argparse
import shlex
import subprocess
import sys


REMOTE_SCRIPT = r'''
from pathlib import Path
import json
import re
import subprocess

PROJECT_ROOT = Path.home() / "projects/alignn"
LOGS_DIR = PROJECT_ROOT / "results/logs"

WATCH_LOGS = [
    "improve_scratch_vs_original_e30_sweep_resume.log",
    "fair_e60_comparison_resume.log",
    "compare_against_original_e30_resume_watcher.log",
    "improve_scratch_vs_original_e30_sweep.log",
    "fair_e60_comparison_pipeline.log",
    "compare_against_original_e30_watcher.log",
]

PROCESS_PATTERNS = [
    "run_target_specific_scratch_improvement_sweep.sh",
    "run_target_specific_scratch_e60_completion.sh",
    "run_fair_e60_resume_pipeline.sh",
    "run_original_alignn_e60_after_wait.sh",
    "compare_against_original_e30.py",
    "alignn-train-small",
]

EPOCH_RE = re.compile(r"epoch=(\d+)")
RUN_NAME_RE = re.compile(r"--run-name\s+([^\s]+)")


def _run(cmd: list[str]) -> str:
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    return (result.stdout or "") + (result.stderr or "")


def _tail(path: Path, max_lines: int = 200) -> list[str]:
    if not path.exists():
        return []
    lines = path.read_text(errors="ignore").splitlines()
    return lines[-max_lines:]


def _summarize_log(path: Path) -> None:
    print(f"[{path.name}]")
    if not path.exists():
        print("  status: missing")
        return

    lines = _tail(path)
    print(f"  bytes: {path.stat().st_size}")

    start_lines = [line for line in lines if line.startswith("[start]")]
    done_lines = [line for line in lines if line.startswith("[done]")]
    wait_lines = [line for line in lines if line.startswith("[wait]")]
    skip_lines = [line for line in lines if line.startswith("[skip]")]
    epoch_lines = [line for line in lines if "epoch=" in line]

    if start_lines:
        latest_start = start_lines[-1]
        print(f"  latest_start: {latest_start}")
        match = RUN_NAME_RE.search(latest_start)
        if match:
            print(f"  run_name: {match.group(1)}")
    if done_lines:
        print(f"  latest_done: {done_lines[-1]}")
    if wait_lines:
        print(f"  latest_wait: {wait_lines[-1]}")
    if skip_lines:
        print(f"  latest_skip: {skip_lines[-1]}")
    if epoch_lines:
        latest_epoch = epoch_lines[-1]
        print(f"  latest_epoch: {latest_epoch}")
        epochs = [int(m.group(1)) for line in epoch_lines if (m := EPOCH_RE.search(line))]
        if epochs:
            print(f"  epoch_progress: {max(epochs)} epochs logged")

    nonempty = [line for line in lines if line.strip()]
    if nonempty:
        print(f"  latest_line: {nonempty[-1]}")


print("== Active Processes ==")
ps_cmd = [
    "ps",
    "-eo",
    "pid=,ppid=,stat=,etime=,%cpu=,%mem=,args=",
]
ps_output = _run(ps_cmd).splitlines()
matched = [line for line in ps_output if any(pattern in line for pattern in PROCESS_PATTERNS)]
if matched:
    for line in matched:
        print(line)
else:
    print("(none)")

print("\n== GPU ==")
print(_run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader"]).strip() or "(unavailable)")
compute_apps = _run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv,noheader"]).strip()
print(compute_apps or "(no compute apps)")

print("\n== Pipeline Logs ==")
for name in WATCH_LOGS:
    _summarize_log(LOGS_DIR / name)

print("\n== Completed Formation Energy Metrics ==")
rows = []
for path in sorted(LOGS_DIR.glob("*formation_energy*_test_metrics.json")):
    payload = json.loads(path.read_text())
    metrics = payload.get("test_metrics")
    if not isinstance(metrics, dict) or "mae" not in metrics:
        continue
    rows.append(
        (
            path.name,
            float(metrics.get("mae")),
            float(metrics.get("rmse")),
            float(metrics.get("p95_abs_error")),
        )
    )

if rows:
    rows.sort(key=lambda item: item[1])
    for name, mae, rmse, p95 in rows:
        print(f"{name}: mae={mae:.6f} rmse={rmse:.6f} p95={p95:.6f}")
    best_name, best_mae, best_rmse, best_p95 = rows[0]
    print(f"best_completed: {best_name} mae={best_mae:.6f} rmse={best_rmse:.6f} p95={best_p95:.6f}")
else:
    print("(none)")
'''


def main() -> int:
    parser = argparse.ArgumentParser(description="Show queued ALIGNN remote run progress.")
    parser.add_argument("--host", default="alignn")
    parser.add_argument("--ssh-option", action="append", default=[])
    args = parser.parse_args()

    cmd = ["ssh"]
    if not args.ssh_option:
        cmd.extend([
            "-o",
            "ConnectTimeout=15",
            "-o",
            "ServerAliveInterval=5",
            "-o",
            "ServerAliveCountMax=2",
        ])
    for option in args.ssh_option:
        cmd.extend(["-o", option])
    cmd.extend([args.host, "python3", "-"])

    result = subprocess.run(cmd, input=REMOTE_SCRIPT, text=True)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
