#!/usr/bin/env bash
set -euo pipefail
echo "Waiting for best5 session to finish..."
while tmux has-session -t best5 2>/dev/null; do
    sleep 60
done
echo "best5 done. Starting bm_tail."
cd ~/projects/alignn
bash scripts/run_bm_tail_aware.sh 2>&1 | tee results/logs/bm_tail_aware_resume.log
