#!/usr/bin/env bash
set -euo pipefail
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/train_reverse_8gpu.sh" pmean0 "$@"
