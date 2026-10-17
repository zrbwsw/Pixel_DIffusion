#!/usr/bin/env bash
# agent-gpu-2 physical GPUs 0-3.
set -euo pipefail
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/train_rmsnorm2_ablation_4gpu.sh" reverse "$@"
