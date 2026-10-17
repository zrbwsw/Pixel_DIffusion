#!/usr/bin/env bash
set -euo pipefail

# Default: 20 epochs, DINOv3-B/16 alignment, and a JiT-only input RMSNorm.
# Accept a positional config or --config, and forward all CLI overrides.
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
train_config="$project_root/configs/jit_rmsnorm_basejit_irepa_dinov3_b20_gbs512.yaml"
if [[ $# -gt 0 && "$1" != -* ]]; then
    train_config="$1"
    shift
fi
exec bash "$project_root/scripts/train_basejit_irepa.sh" "$train_config" "$@"
