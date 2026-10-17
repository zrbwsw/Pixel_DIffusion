#!/usr/bin/env bash
set -euo pipefail

# First positional argument or --config selects a config. Remaining arguments
# override its defaults. This script does not launch anything until invoked.
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
train_config="$project_root/configs/basejit_irepa_dinov1_b20_gbs512.yaml"
if [[ $# -gt 0 && "$1" != -* ]]; then
    train_config="$1"
    shift
fi
train_args=()
preflight=false
show_help=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            if [[ $# -lt 2 ]]; then
                echo '--config requires a YAML path.' >&2
                exit 2
            fi
            train_config="$2"
            shift 2
            ;;
        --config=*)
            train_config="${1#--config=}"
            shift
            ;;
        --preflight)
            preflight=true
            train_args+=("$1")
            shift
            ;;
        --help|-h)
            show_help=true
            train_args+=("$1")
            shift
            ;;
        *)
            train_args+=("$1")
            shift
            ;;
    esac
done

# Prefer an explicit GPUS selection, then an existing CUDA visibility setting.
export CUDA_VISIBLE_DEVICES="${GPUS:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}"
if [[ "$CUDA_VISIBLE_DEVICES" =~ (^,|,,|,$) || "$CUDA_VISIBLE_DEVICES" =~ [[:space:]] ]]; then
    echo 'GPU selection must be a nonempty comma-separated list without spaces.' >&2
    exit 2
fi
IFS=',' read -r -a train_gpus <<< "$CUDA_VISIBLE_DEVICES"
declare -A seen_gpus=()
for gpu in "${train_gpus[@]}"; do
    if [[ -n "${seen_gpus[$gpu]+present}" ]]; then
        echo "Repeated GPU in CUDA_VISIBLE_DEVICES: $gpu" >&2
        exit 2
    fi
    seen_gpus["$gpu"]=1
done
train_world_size="${#train_gpus[@]}"
python_bin="${PYTHON_BIN:-python}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export DINOV3_REPO="${DINOV3_REPO:-$project_root/SSL_code/dinov3}"
cd -- "$project_root"

if [[ "$preflight" == true || "$show_help" == true ]]; then
    # Report the intended distributed batch size without spawning workers,
    # constructing a model, or starting a training/evaluation operation.
    exec env WORLD_SIZE="$train_world_size" "$python_bin" main_basejit_irepa.py \
        --config "$train_config" "${train_args[@]}"
fi

# main_basejit_irepa.py checks the effective global batch and protects the
# output directory against accidental reuse before calling original iREPA.
exec "$python_bin" -m torch.distributed.run \
    --standalone --nproc_per_node="$train_world_size" \
    main_basejit_irepa.py --config "$train_config" "${train_args[@]}"
