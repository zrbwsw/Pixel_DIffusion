"""Check local launcher arguments without initializing CUDA or training."""

import json
import os
from pathlib import Path
import subprocess

import main_basejit_irepa as entry


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/train_jit_rmsnorm_irepa_8gpu_bsz512.sh"


def test_local_launcher_preserves_training_configuration(tmp_path):
    python = tmp_path / "python-probe"
    python.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "print(json.dumps({'argv': sys.argv[1:], 'gpus': os.environ['CUDA_VISIBLE_DEVICES'], "
        "'world_size': os.environ['WORLD_SIZE']}))\n",
    )
    python.chmod(0o755)
    env = dict(os.environ, PYTHON_BIN=str(python))
    env.pop("GPUS", None)
    # The local wrapper must select eight cards even in a four-card parent shell.
    env["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    result = subprocess.run(
        ["bash", str(SCRIPT), "--preflight"], cwd=tmp_path, env=env,
        check=True, capture_output=True, text=True,
    )
    report = json.loads(result.stdout)
    assert report["gpus"] == "0,1,2,3,4,5,6,7"
    assert report["world_size"] == "8"
    assert report["argv"][0] == "main_basejit_irepa.py"
    actual = entry.parse_args(report["argv"][1:])
    reference = entry.parse_args([
        "--config", str(ROOT / "configs/jit_rmsnorm_basejit_irepa_dinov3_b20_gbs512.yaml"),
    ])
    differences = {key for key in vars(reference) if getattr(reference, key) != getattr(actual, key)}
    assert differences == {"batch_size", "lr", "exp_name", "output_dir", "preflight"}
    assert actual.batch_size == 64
    assert actual.expected_global_batch_size == actual.batch_size * 8 == 512
    assert actual.lr == reference.blr * 512 / 256 == 1e-4
    assert actual.generation_input_mode == "jit_rmsnorm"
    assert actual.enc_type == "dinov3-vit-b16"
    assert actual.epochs == reference.epochs == 20
    assert actual.preflight
