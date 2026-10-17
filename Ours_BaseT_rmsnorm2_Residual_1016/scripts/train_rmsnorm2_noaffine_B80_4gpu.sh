#!/usr/bin/env bash
# agent-gpu physical GPUs 4-7.
set -euo pipefail
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/train_rmsnorm2_ablation_4gpu.sh" noaffine "$@"
