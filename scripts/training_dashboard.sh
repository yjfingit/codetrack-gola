#!/usr/bin/env bash
# Read-only loss/ETA terminal UI at 20 Hz; q exits only the dashboard.
# Default: current two-GPU frame-coverage run. --once prints once and exits.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
dashboard_args=()
if [[ -n "${CODETRACK_RUN:-}" ]]; then
  dashboard_args=(--run "$CODETRACK_RUN")
fi
exec "${PYTHON:-/home/yangjuanfeng/lab/envs/gola/bin/python}" tools/training_dashboard.py "${dashboard_args[@]}" "$@"
