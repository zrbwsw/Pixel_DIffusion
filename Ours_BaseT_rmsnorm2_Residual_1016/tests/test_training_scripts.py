"""Exercise readable launch recipes without SSH, CUDA, or starting training."""

import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest

from scripts.check_runtime import training_args


ROOT = Path(__file__).resolve().parents[1]
BASE = "BaseT-Residual(t,1-t)-NoDinoPre-PgT-blend_rmsnorm2-NoDDTHead-B-80"
REVERSE = BASE.replace("(t,1-t)", "(1-t,t)")
RECIPES = [
    ("train_rmsnorm2_B80_4gpu.sh", f"{BASE}.yaml", 4, {}),
    ("train_rmsnorm2_B80_8gpu.sh", f"{BASE}.yaml", 8, {}),
    ("train_rmsnorm2_dropres_B80_4gpu.sh", f"{BASE.replace('-NoDDTHead', '-DropRes-NoDDTHead')}.yaml", 4,
     {"dino_grounding_drop_on_label_drop": True}),
    ("train_rmsnorm2_noaffine_B80_4gpu.sh", f"{BASE.replace('-NoDDTHead', '-NoAffine-NoDDTHead')}.yaml", 4,
     {"dino_generation_final_norm_affine": False}),
    ("train_rmsnorm2_reverse_B80_4gpu.sh", f"{REVERSE}.yaml", 4,
     {"dino_time_blend_reverse": True}),
    ("train_rmsnorm2_reverse_nomls_B80_8gpu.sh", f"{REVERSE.replace('-NoDDTHead', '-NoMLS-NoDDTHead')}.yaml", 8,
     {"dino_time_blend_reverse": True, "dino_aggregation": "none"}),
    ("train_rmsnorm2_reverse_pmean0_B80_8gpu.sh", f"{REVERSE.replace('-NoDDTHead', '-PMean0-NoDDTHead')}.yaml", 8,
     {"dino_time_blend_reverse": True, "P_mean": 0.0}),
    ("train_rmsnorm2_reverse_train_final_norm_B80_8gpu.sh", "reverse_train_final_norm_B80_8gpu.yaml", 8,
     {"dino_time_blend_reverse": True, "dino_train_final_norm": True}),
    ("train_rmsnorm2_reverse_sigmoid_blend_B80_4gpu.sh", "reverse_sigmoid_blend_B80.yaml", 4,
     {"dino_time_blend_reverse": True, "dino_time_blend_transform": "sigmoid",
      "dino_time_blend_key": 100.0}),
    ("train_rmsnorm2_reverse_sigmoid_blend_B80_8gpu.sh", "reverse_sigmoid_blend_B80.yaml", 8,
     {"dino_time_blend_reverse": True, "dino_time_blend_transform": "sigmoid",
      "dino_time_blend_key": 100.0}),
]


@pytest.fixture
def launch_env(monkeypatch, tmp_path):
    # Eliminate machine-specific overrides; parsing does not read these assets.
    for key in ("CUDA_VISIBLE_DEVICES", "RESUME_DIR", "TRAIN_ON_GPU"):
        monkeypatch.delenv(key, raising=False)
    overrides = {
        "PYTHON_BIN": sys.executable,
        "GPU_HOST": "local",
        "OUTPUT_DIR": str(tmp_path / "output with spaces (new)"),
        "IMAGENET_ROOT": str(tmp_path / "imagenet"),
        "PRETRAIN_MODELS_ROOT": str(tmp_path / "weights"),
        "FID_STATS_ROOT": str(tmp_path / "stats"),
        "NANOGEN_EVALS_ROOT": str(tmp_path / "evals"),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    for key, value in overrides.items():
        monkeypatch.setenv(key, value)
    return dict(os.environ)


def dry_run(script, env, tmp_path, *options):
    result = subprocess.run(
        ["bash", str(ROOT / "scripts" / script), "--dry-run", *options],
        env=env, cwd=tmp_path, text=True, capture_output=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    command = shlex.split(result.stdout)
    assert command[0] == "torchrun"
    index = command.index("main_dino_jit.py")
    return command[:index], training_args(command[index + 1:])


@pytest.mark.parametrize("script,config,world_size,controls", RECIPES, ids=[r[0] for r in RECIPES])
def test_recipe_matches_yaml_and_has_expected_controls(script, config, world_size, controls, launch_env, tmp_path):
    launcher, args = dry_run(script, launch_env, tmp_path)
    baseline = training_args([
        "--config", str(ROOT / "configs" / config),
        "--data_path", launch_env["IMAGENET_ROOT"],
    ])
    allowed = {"batch_size", "output_dir", "resume", "exp_name", "metrics_log_name"}
    differences = {key: (value, vars(args)[key]) for key, value in vars(baseline).items()
                   if value != vars(args)[key] and key not in allowed}
    assert differences == {}
    assert f"--nproc_per_node={world_size}" in launcher
    assert args.batch_size == 1024 // world_size
    assert args.batch_size * world_size * args.accum_iter == args.global_batch_size == 1024
    assert args.blr * args.global_batch_size / 256 == pytest.approx(2e-4)
    assert args.resume == ""
    assert args.output_dir == launch_env["OUTPUT_DIR"]
    expected_controls = {
        "dino_time_blend_reverse": False,
        "dino_train_final_norm": False,
        "dino_generation_final_norm_affine": True,
        "dino_grounding_drop_on_label_drop": False,
        "dino_aggregation": "mls",
        "P_mean": -0.8,
        **controls,
    }
    for key, value in expected_controls.items():
        assert getattr(args, key) == value, key
    assert not Path(args.output_dir).exists(), "Dry run must not create experiment output"


@pytest.mark.parametrize("script,config,world_size,controls", RECIPES, ids=[r[0] for r in RECIPES])
def test_recipe_rejects_wrong_gpu_count(script, config, world_size, controls, launch_env, tmp_path):
    env = {**launch_env, "CUDA_VISIBLE_DEVICES": "0,1"}
    result = subprocess.run(
        ["bash", str(ROOT / "scripts" / script), "--dry-run"],
        env=env, cwd=tmp_path, text=True, capture_output=True, timeout=30,
    )
    assert result.returncode != 0
    assert f"需要 {world_size} 张卡" in result.stderr
    assert not Path(env["OUTPUT_DIR"]).exists()


@pytest.mark.parametrize("resume_first", [False, True])
@pytest.mark.parametrize("equals_style", [False, True])
def test_bare_resume_uses_final_output_independent_of_order(resume_first, equals_style, launch_env, tmp_path):
    output = str(tmp_path / "overridden output (continue)")
    output_options = [f"--output_dir={output}"] if equals_style else ["--output_dir", output]
    options = ["--resume", *output_options] if resume_first else [*output_options, "--resume"]
    _, args = dry_run("train_rmsnorm2_B80_8gpu.sh", launch_env, tmp_path, *options)
    assert args.output_dir == output
    assert args.resume == output


@pytest.mark.parametrize("equals_style", [False, True])
def test_explicit_resume_survives_output_override(equals_style, launch_env, tmp_path):
    resume = str(tmp_path / "source checkpoint (old)")
    output = str(tmp_path / "new continuation")
    resume_options = [f"--resume={resume}"] if equals_style else ["--resume", resume]
    _, args = dry_run("train_rmsnorm2_B80_8gpu.sh", launch_env, tmp_path,
                      *resume_options, "--output_dir", output)
    assert args.resume == resume
    assert args.output_dir == output


@pytest.fixture
def fake_training(launch_env, tmp_path):
    data = Path(launch_env["IMAGENET_ROOT"])
    (data / "train").mkdir(parents=True)
    models = Path(launch_env["PRETRAIN_MODELS_ROOT"])
    models.mkdir()
    (models / "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth").write_bytes(b"resource check only")
    stats = Path(launch_env["FID_STATS_ROOT"])
    stats.mkdir()
    for name in ("jit_in256_stats.npz", "weights-inception-2015-12-05-6726825d.pth"):
        (stats / name).write_bytes(b"resource check only")
    tools = tmp_path / "fake tools"
    tools.mkdir()
    for name, content in {
        "nvidia-smi": '#!/usr/bin/env bash\nprintf "0, fake GPU for launcher test\\n"\n',
        "fake-training": '#!/usr/bin/env bash\nprintf "fake training invoked\\n"\nexit "${FAKE_TRAIN_EXIT:-0}"\n',
        "ssh": '#!/usr/bin/env bash\necho "SSH is forbidden in these tests" >&2\nexit 99\n',
    }.items():
        path = tools / name
        path.write_text(content)
        path.chmod(0o755)
    env = {**launch_env, "PATH": str(tools) + os.pathsep + launch_env["PATH"],
           "PROJECT_ROOT": str(ROOT), "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7"}
    command = [
        "bash", "-c",
        'set -euo pipefail; source "$1"; shift; TRAIN_EXTRA_ARGS=(); '
        'gpus=(0 1 2 3 4 5 6 7); TRAIN_MODE=run; run_training "$@"',
        "test-training-helper", str(ROOT / "scripts/train_env.sh"),
        str(tools / "fake-training"), "main_dino_jit.py",
        "--config", str(ROOT / "configs" / f"{BASE}.yaml"),
        "--batch_size", "128", "--output_dir", launch_env["OUTPUT_DIR"],
        "--resume", "", "--data_path", str(data),
    ]
    return command, env, Path(launch_env["OUTPUT_DIR"])


@pytest.mark.parametrize("exit_code", [0, 7])
def test_helper_records_success_and_failure_without_training(fake_training, exit_code, tmp_path):
    command, env, output = fake_training
    env["FAKE_TRAIN_EXIT"] = str(exit_code)
    result = subprocess.run(command, env=env, cwd=tmp_path, capture_output=True, text=True, timeout=90)
    assert result.returncode == exit_code, result.stderr
    assert (output / "exit_code.txt").read_text().strip() == str(exit_code)
    assert (output / "finished_at.txt").read_text().strip()
    assert (output / "started_at.txt").read_text().strip()
    assert "fake training invoked" in (output / "train.log").read_text()
    report = json.loads((output / "resource_preflight.json").read_text())
    assert report["training_mode"] == "from_scratch"
    assert report["global_batch_size"] == 1024
    assert report["effective_lr"] == pytest.approx(2e-4)


def test_helper_rejects_existing_output_before_launcher(fake_training, tmp_path):
    command, env, output = fake_training
    output.mkdir()
    existing = output / "train.log"
    existing.write_text("existing experiment must remain intact\n")
    result = subprocess.run(command, env=env, cwd=tmp_path, capture_output=True, text=True, timeout=90)
    assert result.returncode != 0
    assert "输出目录已有内容" in result.stderr
    assert "fake training invoked" not in result.stdout
    assert existing.read_text() == "existing experiment must remain intact\n"
    assert sorted(p.name for p in output.iterdir()) == ["train.log"]
