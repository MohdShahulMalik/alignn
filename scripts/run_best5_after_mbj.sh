#!/usr/bin/env bash
set -euo pipefail
echo "Waiting for mbj session to finish..."
while tmux has-session -t mbj 2>/dev/null; do
    sleep 60
done
echo "mbj done. Starting best5."
cd ~/projects/alignn
bash scripts/run_best5_targets.sh 2>&1 | tee results/logs/best5_targets_resume.log
