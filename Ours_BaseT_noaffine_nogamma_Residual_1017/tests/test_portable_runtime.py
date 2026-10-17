import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.check_runtime import inspect_runtime, training_args


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "BaseT-Residual(t,1-t)-NoDinoPre-PgT-blend_rmsnorm-NoDDTHead-L-80"
H200_EXPERIMENT = "BaseT-Residual(t,1-t)-NoDinoPre-PgT-blend_rmsnorm_nogamma-NoDDTHead-B-80"


@pytest.fixture
def resources(tmp_path, monkeypatch):
    data = tmp_path / "data with spaces"
    (data / "train").mkdir(parents=True)
    models = tmp_path / "models"
    models.mkdir()
    (models / "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth").write_bytes(b"placeholder")
    (models / "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth").write_bytes(b"placeholder")
    stats = tmp_path / "fid stats"
    stats.mkdir()
    for name in ("jit_in256_stats.npz", "weights-inception-2015-12-05-6726825d.pth"):
        (stats / name).write_bytes(b"placeholder")
    monkeypatch.setenv("PRETRAIN_MODELS_ROOT", str(models))
    monkeypatch.setenv("FID_STATS_ROOT", str(stats))
    return data


def config_args(data, *overrides):
    return training_args([
        "--config", str(ROOT / "configs" / f"{EXPERIMENT}.yaml"),
        "--data_path", str(data), "--resume", "", *overrides,
    ])


@pytest.mark.parametrize("world_size,batch_size,accumulation", [(8, 32, 1), (4, 32, 2), (4, 64, 1)])
def test_preflight_derives_accumulation(resources, world_size, batch_size, accumulation):
    args = config_args(resources, "--batch_size", str(batch_size))
    report = inspect_runtime(args, world_size)
    assert report["accum_iter"] == accumulation
    assert report["global_batch_size"] == 256
    assert report["training_mode"] == "from_scratch"
    assert report["dino_model"] == "dinov3_vitl16"


def test_preflight_rejects_invalid_batch(resources):
    with pytest.raises(ValueError, match="positive multiple"):
        inspect_runtime(config_args(resources, "--batch_size", "33"), 8)


def test_missing_dataset_fails(resources, tmp_path):
    with pytest.raises(FileNotFoundError, match="ImageNet"):
        inspect_runtime(config_args(tmp_path / "missing"), 8)


def test_sampling_preflight_does_not_require_training_data(resources, tmp_path):
    report = inspect_runtime(config_args(tmp_path / "missing"), sampling=True)
    assert "data_path" not in report


def test_missing_online_assets_fails(resources, monkeypatch, tmp_path):
    monkeypatch.setenv("FID_STATS_ROOT", str(tmp_path / "absent"))
    with pytest.raises(FileNotFoundError, match="Online evaluation"):
        inspect_runtime(config_args(resources), 8)


def test_preflight_detects_resume_without_loading_weights(resources, tmp_path):
    resume = tmp_path / "resume"
    resume.mkdir()
    checkpoint = resume / "checkpoint-last.pth"
    checkpoint.write_bytes(b"not a real torch checkpoint")
    report = inspect_runtime(config_args(resources, "--resume", str(resume)), 8)
    assert report["training_mode"] == "resume"
    assert report["resume_checkpoint"] == str(checkpoint)


def test_launcher_preflight_works_from_another_directory(resources, tmp_path):
    output = tmp_path / "must not be created"
    env = dict(os.environ, PYTHON_BIN=sys.executable, CUDA_VISIBLE_DEVICES="0,1,2,3",
               IMAGENET_ROOT=str(resources), PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([
        "bash", str(ROOT / "scripts/train_portable.sh"),
        str(ROOT / "configs" / f"{EXPERIMENT}.yaml"),
        "--batch_size", "32", "--output_dir", str(output), "--resume", "", "--preflight",
    ], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stderr
    report, _ = json.JSONDecoder().raw_decode(result.stdout)
    assert report["accum_iter"] == 2
    assert report["output_dir"] == str(output)
    assert not output.exists()


def test_duplicate_gpu_ids_are_rejected(resources, tmp_path):
    env = dict(os.environ, PYTHON_BIN=sys.executable, CUDA_VISIBLE_DEVICES="0,0")
    result = subprocess.run([
        "bash", str(ROOT / "scripts/train_portable.sh"),
        str(ROOT / "configs" / f"{EXPERIMENT}.yaml"), "--preflight",
    ], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    assert "Duplicate GPU index" in result.stderr


@pytest.mark.parametrize("batch_size,accumulation", [(128, 1), (64, 2)])
def test_h200_preflight_reports_scaled_lr_at_global_batch_1024(resources, batch_size, accumulation):
    args = training_args([
        "--config", str(ROOT / "configs" / f"{H200_EXPERIMENT}.yaml"),
        "--data_path", str(resources), "--batch_size", str(batch_size),
    ])
    report = inspect_runtime(args, 8)
    assert report["global_batch_size"] == 1024
    assert report["accum_iter"] == accumulation
    assert report["configured_blr"] == pytest.approx(5e-5)
    assert report["effective_lr"] == pytest.approx(2e-4)
    assert report["lr_source"] == "scaled_blr"
    args.lr = 3e-4
    explicit = inspect_runtime(args, 8)
    assert explicit["effective_lr"] == pytest.approx(3e-4)
    assert explicit["lr_source"] == "explicit"


def test_h200_launcher_uses_eight_gpus_and_expected_batch_lr(resources, tmp_path):
    output = tmp_path / "h200 preflight must not create"
    env = dict(os.environ, PYTHON_BIN=sys.executable, CUDA_VISIBLE_DEVICES="4,5,6,7",
               IMAGENET_ROOT=str(resources), PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([
        "bash", str(ROOT / "scripts" / f"train_{H200_EXPERIMENT}_01234567.sh"),
        "--output_dir", str(output), "--resume", "", "--preflight",
    ], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stderr
    report, _ = json.JSONDecoder().raw_decode(result.stdout)
    assert report["world_size"] == 8
    assert report["batch_size_per_gpu"] == 128
    assert report["global_batch_size"] == 1024
    assert report["accum_iter"] == 1
    assert report["configured_blr"] == pytest.approx(5e-5)
    assert report["effective_lr"] == pytest.approx(2e-4)
    assert report["lr_source"] == "scaled_blr"
    assert report["training_mode"] == "from_scratch"
    assert report["dino_source"] == str(ROOT / "SSL_code/dinov3")
    assert report["output_dir"] == str(output)
    assert not output.exists()
