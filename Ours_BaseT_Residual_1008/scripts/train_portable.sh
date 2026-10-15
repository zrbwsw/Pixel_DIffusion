#!/usr/bin/env bash
set -euo pipefail

[[ "$#" -ge 1 ]] || { echo "Usage: $0 CONFIG [--preflight] [training CLI overrides...]" >&2; exit 2; }
config="$(realpath -- "$1")"
shift
source "$(dirname -- "${BASH_SOURCE[0]}")/portable_env.sh"
args=(--config "$config")
[[ -z "${IMAGENET_ROOT:-}" ]] || args+=(--data_path "$IMAGENET_ROOT")
preflight=false
for arg in "$@"; do
    if [[ "$arg" == --preflight ]]; then
        preflight=true
    else
        args+=("$arg")
    fi
done
cd -- "$PROJECT_ROOT"
report="$("$PYTHON_BIN" scripts/check_runtime.py --world-size "${#gpus[@]}" "${args[@]}")"
printf '%s\n' "$report"
if [[ "$preflight" == true ]]; then
    echo "Preflight passed; no model was loaded and no training was started."
    exit 0
fi
output_dir="$(printf '%s\n' "$report" | "$PYTHON_BIN" -c 'import json, sys; print(json.load(sys.stdin)["output_dir"])')"
mkdir -p -- "$output_dir"
"$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node="${#gpus[@]}" \
    main_dino_jit.py "${args[@]}" 2>&1 | tee -a "$output_dir/train.log"
