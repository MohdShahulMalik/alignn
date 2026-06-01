#!/usr/bin/env bash

set -euo pipefail

echo "Waiting for L1 session to finish..."
while tmux has-session -t l1 2>/dev/null; do
    sleep 60
done
echo "L1 session done. Starting SmoothL1 formation energy runs."

bash ~/projects/alignn/scripts/run_fe_smoothl1.sh 2>&1 | tee ~/projects/alignn/results/logs/fe_smoothl1.log
