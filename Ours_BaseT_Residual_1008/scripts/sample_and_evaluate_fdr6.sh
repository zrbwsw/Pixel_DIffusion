#!/usr/bin/env bash
set -euo pipefail

[[ "$#" -ge 3 ]] || { echo "Usage: $0 CONFIG CHECKPOINT EVAL_DIR [--gpus=4,5,6,7] [--cfg=2.5] [--sampler=flow_dpm_solver] [--steps=100] [--preflight]" >&2; exit 2; }
config="$(realpath "$1")"
checkpoint="$(realpath "$2")"
eval_dir="$(realpath -m "$3")"
shift 3

export CUDA_VISIBLE_DEVICES=4,5,6,7
export PRETRAIN_MODELS_ROOT=/data0/pretrain_models
export NANOGEN_EVALS_ROOT=/data0/zhangruobei/Work/Pixel_Diffusion/nanogen-evals
export HF_HUB_OFFLINE=1

preflight=false
sampling_args=()
for arg in "$@"; do
    case "$arg" in
        --gpus=*) export CUDA_VISIBLE_DEVICES="${arg#*=}" ;;
        --preflight) preflight=true ;;
        --cfg=*|--sampler=*|--steps=*|--gen-bsz=*|--num-images=*|--interval-min=*|--interval-max=*)
            sampling_args+=("$arg") ;;
        *) echo "Unknown argument: $arg (use --name=value)" >&2; exit 2 ;;
    esac
done
[[ "$CUDA_VISIBLE_DEVICES" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo "Invalid GPU IDs" >&2; exit 2; }

cd /data0/zhangruobei/Work/Pixel_Diffusion/Ours_BaseT_Residual_1008
export PYTHONPATH="$PWD:$PWD/jit${PYTHONPATH:+:$PYTHONPATH}"
python=/data0/zhangruobei/anaconda3/envs/irepa/bin/python
torchrun=/data0/zhangruobei/anaconda3/envs/irepa/bin/torchrun
IFS=, read -r -a gpus <<< "$CUDA_VISIBLE_DEVICES"

[[ -x "$python" && -x "$torchrun" && "${#gpus[@]}" -gt 0 ]] || { echo "Missing Python, torchrun or GPUs" >&2; exit 1; }
[[ -f "$config" && -s "$checkpoint" && -d "$NANOGEN_EVALS_ROOT" ]] || { echo "Missing config, checkpoint or offline assets" >&2; exit 1; }

if rg -q 'dinov3_vitl16|dinov3_vitlarge' "$config"; then
    dino_weights="$PRETRAIN_MODELS_ROOT/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
    dino_variant="L"
else
    dino_weights="$PRETRAIN_MODELS_ROOT/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
    dino_variant="B"
fi
[[ -s "$dino_weights" ]] || { echo "Missing DINOv3-$dino_variant weights: $dino_weights" >&2; exit 1; }

mkdir -p "$eval_dir"
"$python" -m evaluation.evaluate_fdr6 --check-assets \
    --metrics=fid,inception_score,fdr6 --cache-root="$NANOGEN_EVALS_ROOT" \
    > "$eval_dir/evaluation_assets.json"

"$python" -m evaluation.prepare_sampling \
    --config="$config" --checkpoint="$checkpoint" --output-dir="$eval_dir" \
    "${sampling_args[@]}"

if [[ "$preflight" == true ]]; then
    echo "Assets and sampling configuration validated; sampling was not started."
    exit 0
fi

if [[ ! -f "$eval_dir/samples.npz" ]]; then
    "$torchrun" --standalone --nproc_per_node="${#gpus[@]}" \
        main_dino_jit.py --config="$eval_dir/sampling_config.yaml" \
        2>&1 | tee -a "$eval_dir/sampling.log"
else
    echo "Completed NPZ exists; skip generation: $eval_dir/samples.npz"
fi

CUDA_VISIBLE_DEVICES="${gpus[0]}" "$python" -m evaluation.evaluate_fdr6 \
    --npz="$eval_dir/samples.npz" --metrics=fid,inception_score,fdr6 \
    --cache-root="$NANOGEN_EVALS_ROOT" --output-json="$eval_dir/offline_metrics.json" \
    2>&1 | tee -a "$eval_dir/offline_metrics.log"
