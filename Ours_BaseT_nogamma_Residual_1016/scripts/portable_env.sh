#!/usr/bin/env bash
# Shared path setup for the portable launchers; does not alter model settings.
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
REPOSITORY_ROOT="$(dirname -- "$PROJECT_ROOT")"
PYTHON_BIN="${PYTHON_BIN:-python}"
PYTHON_BIN="$(command -v "$PYTHON_BIN")" || { echo "Python not found; activate irepa or set PYTHON_BIN" >&2; return 1; }
PYTHON_BIN="$(realpath -- "$PYTHON_BIN")"
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/jit${PYTHONPATH:+:$PYTHONPATH}"
export PRETRAIN_MODELS_ROOT="${PRETRAIN_MODELS_ROOT:-$REPOSITORY_ROOT/pretrain_models}"
export FID_STATS_ROOT="${FID_STATS_ROOT:-$REPOSITORY_ROOT/fid_stats}"
export NANOGEN_EVALS_ROOT="${NANOGEN_EVALS_ROOT:-$REPOSITORY_ROOT/nanogen-evals}"
# Resolve relative overrides before the launchers change directory.
export PRETRAIN_MODELS_ROOT="$(realpath -m -- "$PRETRAIN_MODELS_ROOT")"
export FID_STATS_ROOT="$(realpath -m -- "$FID_STATS_ROOT")"
export NANOGEN_EVALS_ROOT="$(realpath -m -- "$NANOGEN_EVALS_ROOT")"
if [[ -n "${IMAGENET_ROOT:-}" ]]; then
    export IMAGENET_ROOT="$(realpath -m -- "$IMAGENET_ROOT")"
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
[[ "$CUDA_VISIBLE_DEVICES" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo "CUDA_VISIBLE_DEVICES must be comma-separated GPU indices" >&2; return 1; }
IFS=, read -r -a gpus <<< "$CUDA_VISIBLE_DEVICES"
declare -A selected_gpus=()
for gpu in "${gpus[@]}"; do
    [[ -z "${selected_gpus[$gpu]:-}" ]] || { echo "Duplicate GPU index: $gpu" >&2; return 1; }
    selected_gpus[$gpu]=1
done
