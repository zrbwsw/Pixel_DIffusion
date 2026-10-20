"""Configuration isolation and resource routing for the JiT-only ablation."""

from pathlib import Path

import pytest

import main_basejit_irepa as entry


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("schedule", ["b20_gbs512", "b80_gbs1024"])
def test_paired_configs_change_only_mode_and_output(schedule):
    def load(prefix, teacher):
        return vars(entry.parse_args([
            "--config", str(ROOT / "configs" / f"{prefix}_basejit_irepa_{teacher}_{schedule}.yaml"),
        ]))

    baseline = vars(entry.parse_args([
        "--config", str(ROOT / "configs" / f"basejit_irepa_dinov3_{schedule}.yaml"),
    ]))
    normalized = load("jit_rmsnorm", "dinov3")
    differing = {key for key in baseline if baseline[key] != normalized[key]}
    assert differing == {"config", "exp_name", "output_dir", "generation_input_mode"}
    assert baseline["generation_input_mode"] == "baseline"
    assert normalized["generation_input_mode"] == "jit_rmsnorm"
    assert normalized["enc_type"] == "dinov3-vit-b16"
    historical = vars(entry.parse_args([
        "--config", str(ROOT / "configs" / f"basejit_irepa_dinov1_{schedule}.yaml"),
    ]))
    assert {key for key in historical if historical[key] != baseline[key]} == {
        "config", "exp_name", "output_dir", "enc_type",
    }


def test_dinov3_alignment_assets_do_not_require_dinov1_or_generation_weights(tmp_path, monkeypatch):
    weights, repo = tmp_path / "weights", tmp_path / "dinov3"
    weights.mkdir()
    repo.mkdir()
    teacher = weights / "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
    teacher.touch()
    (repo / "hubconf.py").touch()
    monkeypatch.setenv("PRETRAIN_MODELS_ROOT", str(tmp_path / "unused_weights"))
    monkeypatch.setenv("DINOV3_REPO_DIR", str(repo))
    monkeypatch.setenv("DINOV3_CKPT_DIR", str(weights))
    monkeypatch.setattr(entry, "_link_asset", lambda *args: pytest.fail("Unexpected DINOv1/cache link"))
    monkeypatch.setattr(entry.torch.hub, "set_dir", lambda *args: None)
    args = entry.parse_args([
        "--config", str(ROOT / "configs/jit_rmsnorm_basejit_irepa_dinov3_b20_gbs512.yaml"),
        "--dino_weights", str(tmp_path / "missing_generation_weight.pth"),
    ])
    args.online_eval = False
    report = entry.prepare_runtime_assets(args)
    assert report["alignment_weights"] == str(teacher)


def test_dinov3_alignment_missing_weight_fails_before_training(tmp_path, monkeypatch):
    monkeypatch.setenv("DINOV3_CKPT_DIR", str(tmp_path))
    monkeypatch.setattr(entry.torch.hub, "set_dir", lambda *args: None)
    args = entry.parse_args([
        "--config", str(ROOT / "configs/basejit_irepa_dinov3_b20_gbs512.yaml"),
    ])
    with pytest.raises(FileNotFoundError, match="dinov3_vitb16_pretrain"):
        entry.prepare_runtime_assets(args)
