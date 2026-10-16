#!/usr/bin/env bash
set -euo pipefail

project="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python="${PYTHON_BIN:-python}"
"$python" -c 'import sys; assert sys.version_info[:2] == (3, 10), "Activate a Python 3.10 environment first"'
# Pin the CUDA wheel family separately from the ordinary PyPI dependencies.
"$python" -m pip install torch==2.7.0 torchvision==0.22.0 \
    --index-url https://download.pytorch.org/whl/cu126
"$python" -m pip install -r "$project/require.txt"
"$python" -m pip check
