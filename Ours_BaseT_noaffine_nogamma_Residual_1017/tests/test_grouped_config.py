import argparse
from pathlib import Path

import pytest
import torch.nn as nn

import dino_jit_denoiser
from dino_jit_denoiser import DINOJiTDenoiser
from pixeldit2_projector import TimestepProjection
from jit.config_utils import load_grouped_yaml_defaults, parse_args_with_config
from jit.evaluation.prepare_sampling import prepare_config
from jit.main_jit import get_args_parser
from main_dino_jit import add_dino_jit_args
from util.model_util import RMSNorm


ROOT = Path(__file__).resolve().parents[1]
BASE = "Ours-J18-new-drop0.1-l4-BaseT-B-80"
RESIDUAL = "Ours-J18-new-drop0.1-l4-BaseT-Residual-B-80"
RESIDUAL_DINOTNORM = "Ours-J18-new-drop0.1-l4-BaseT-Residual-DinoTNorm-B-80"
G1 = "Ours-J18-new-drop0.1-l4-BaseT-Residual-G1-B-80"
DROPRES = "Ours-J18-new-drop0.1-l4-BaseT-Residual-DropRes-B-80"
TIME_BLEND = "BaseT-Residual(t,1-t)-NoDinoPre-PgT-B-80"
NO_DDT = "BaseT-Residual(t,1-t)-NoDinoPre-PgT-NoDDTHead-B-80"
RMSNORM_NO_DDT = "BaseT-Residual(t,1-t)-NoDinoPre-PgT-blend_rmsnorm-NoDDTHead-B-80"
NOGAMMA_NO_DDT = "BaseT-Residual(t,1-t)-NoDinoPre-PgT-blend_rmsnorm_nogamma-NoDDTHead-B-80"
RMSNORM_DROPRES = "BaseT-Residual(t,1-t)-NoDinoPre-PgT-blend_rmsnorm-DropRes-NoDDTHead-B-80"
NATIVE = "BaseT-noResidual-irepa(DINOv3)-DDTHead-B-80"


def _parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--depth", type=int, default=8, choices=[4, 8])
    parser.add_argument("--use_teacher", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--name", type=str, default="baseline")
    return parser


def _active_parser():
    return add_dino_jit_args(get_args_parser())


def _args(name, *overrides):
    return parse_args_with_config(
        _active_parser(), ["--config", str(ROOT / "configs" / f"{name}.yaml"), *overrides]
    )


def test_grouped_yaml_and_cli_override(tmp_path):
    config = tmp_path / "experiment.yaml"
    config.write_text(
        "model:\n  depth: 4\nalignment:\n  use_teacher: false\nrun:\n  name: yaml\n",
        encoding="utf-8",
    )
    args = parse_args_with_config(
        _parser(), ["--config", str(config), "--name", "cli", "--use_teacher"]
    )
    assert (args.depth, args.name, args.use_teacher) == (4, "cli", True)


def test_pretrain_models_root_is_expanded(tmp_path, monkeypatch):
    config = tmp_path / "weights.yaml"
    config.write_text("paths:\n  name: ${PRETRAIN_MODELS_ROOT}/encoder.pth\n", encoding="utf-8")
    monkeypatch.setenv("PRETRAIN_MODELS_ROOT", "/shared/models")
    args = parse_args_with_config(_parser(), ["--config", str(config)])
    assert args.name == "/shared/models/encoder.pth"


def test_unknown_leaf_is_rejected(tmp_path):
    config = tmp_path / "unknown.yaml"
    config.write_text("model:\n  typo_depth: 4\n", encoding="utf-8")
    with pytest.raises(ValueError, match="typo_depth"):
        load_grouped_yaml_defaults(_parser(), config)


def test_duplicate_leaf_is_rejected(tmp_path):
    config = tmp_path / "duplicate.yaml"
    config.write_text("first:\n  depth: 4\nsecond:\n  depth: 8\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate argparse key 'depth'"):
        load_grouped_yaml_defaults(_parser(), config)


@pytest.mark.parametrize("name", [BASE, RESIDUAL, RESIDUAL_DINOTNORM, G1, DROPRES])
def test_active_configs_parse_strictly_and_preserve_current_objective(name):
    args = _args(name)
    assert args.dino_preprocess_input is True
    assert args.dino_aggregation == "mls"
    assert args.dino_layers == "default"
    assert args.jit_alignment_depth == 4
    assert args.dino_projector_kernel_size == 3
    assert args.dino_zscore_alpha == pytest.approx(0.8)
    assert args.ddt_head_dim == 2048
    assert args.ddt_head_depth == 2
    assert args.use_ddt_head is True
    assert args.hflip_prob == pytest.approx(0.5)
    assert args.label_drop_prob == pytest.approx(0.1)
    assert args.global_batch_size == 256
    assert args.epochs == 80


def test_no_residual_native_irepa_config_routes_only_native_jit_tokens():
    args = _args(NATIVE)
    assert args.dino_jit_input_mode == "native"
    assert args.dino_input_projector == "none"
    assert args.dino_aggregation == "none"
    assert args.dino_layers is None
    assert args.use_ddt_head is True
    assert args.jit_alignment_depth == 4
    assert args.jit_add_pos_embed_to_dino_tokens is True


def test_baset_and_residual_preserve_different_positional_embedding_settings():
    base, residual = _args(BASE), _args(RESIDUAL)
    assert base.dino_jit_input_mode == "replace"
    assert base.jit_add_pos_embed_to_dino_tokens is False
    assert base.dino_residual_gate == "none"
    assert base.dino_residual_hidden_dim == 0
    assert residual.dino_jit_input_mode == "gated_residual"
    assert residual.jit_add_pos_embed_to_dino_tokens is True
    assert residual.dino_residual_gate == "clean_linear"
    assert residual.dino_residual_hidden_dim == 768
    assert residual.dino_residual_drop_on_label_drop is False


def test_residual_ablation_configs_change_only_the_intended_control():
    gate_one, drop_residual = _args(G1), _args(DROPRES)
    assert gate_one.dino_residual_gate == "constant_one"
    assert gate_one.dino_residual_drop_on_label_drop is False
    assert drop_residual.dino_residual_gate == "clean_linear"
    assert drop_residual.dino_residual_drop_on_label_drop is True
    ignored = {
        "config", "dino_residual_gate", "dino_residual_drop_on_label_drop",
        "exp_name", "output_dir", "resume",
    }
    assert {
        key for key in vars(gate_one)
        if getattr(gate_one, key) != getattr(drop_residual, key)
    } == ignored


def test_preprocess_flag_defaults_on_and_cli_can_disable_yaml_value():
    assert _args(RESIDUAL).dino_preprocess_input is True
    assert _args(RESIDUAL, "--no-dino_preprocess_input").dino_preprocess_input is False
    assert _args(RESIDUAL, "--dino_preprocess_input").dino_preprocess_input is True


def test_residual_dinotnorm_only_changes_noisy_student_normalization():
    residual, dinotnorm = _args(RESIDUAL), _args(RESIDUAL_DINOTNORM)
    assert residual.dino_input_normalization == "standard"
    assert dinotnorm.dino_input_normalization == "t_moment"
    assert dinotnorm.dino_jit_input_mode == "gated_residual"
    assert dinotnorm.dino_residual_gate == "clean_linear"
    changed = {
        key for key in vars(residual) if getattr(residual, key) != getattr(dinotnorm, key)
    }
    assert changed == {
        "config", "exp_name", "output_dir", "resume",
        "metrics_log_name", "dino_input_normalization",
    }


def test_time_blend_config_only_changes_the_intended_architecture_and_name():
    old, new = _args(RESIDUAL), _args(TIME_BLEND)
    assert new.dino_preprocess_input is False
    assert new.dino_jit_input_mode == "time_blend"
    assert new.dino_input_projector == "pixeldit2"
    assert new.dino_grounding_drop_on_label_drop is False
    assert new.dino_residual_gate == "none"
    assert new.dino_residual_hidden_dim == 0
    assert new.dino_residual_drop_on_label_drop is False
    changed = {
        key for key in vars(old) if getattr(old, key) != getattr(new, key)
    }
    assert changed == {
        "config", "exp_name", "output_dir", "resume", "metrics_log_name",
        "dino_preprocess_input", "dino_jit_input_mode", "dino_input_projector",
        "dino_residual_gate", "dino_residual_hidden_dim",
    }
    assert new.output_dir == new.resume
    assert new.batch_size == 64
    assert new.global_batch_size == 256
    assert new.epochs == 80


def test_no_ddt_ablation_changes_only_the_output_head_and_experiment_paths():
    old, new = _args(TIME_BLEND), _args(NO_DDT)
    assert old.use_ddt_head is True
    assert new.use_ddt_head is False
    assert _args(TIME_BLEND, "--no-use_ddt_head").use_ddt_head is False
    assert _args(NO_DDT, "--use_ddt_head").use_ddt_head is True
    assert {
        key for key in vars(old) if getattr(old, key) != getattr(new, key)
    } == {"config", "exp_name", "output_dir", "resume", "metrics_log_name", "use_ddt_head"}
    assert new.output_dir == new.resume == f"checkpoints/{NO_DDT}"
    assert new.start_epoch == 0


def test_offline_sampling_preserves_preprocess_setting(tmp_path):
    import yaml

    config = yaml.safe_load((ROOT / "configs" / f"{RESIDUAL}.yaml").read_text(encoding="utf-8"))
    config["model"]["dino_student"]["dino_preprocess_input"] = False
    config_path = tmp_path / "no-preprocess.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    checkpoint_path = tmp_path / "checkpoint-80.pth"
    checkpoint_path.touch()

    sampling = prepare_config(config_path, checkpoint_path, tmp_path / "sampling")
    assert sampling["dino_preprocess_input"] is False
    assert sampling["evaluate_gen"] is True


def test_rmsnorm_dropres_changes_only_null_label_routing_and_experiment_paths():
    base, dropres = _args(RMSNORM_NO_DDT), _args(RMSNORM_DROPRES)
    assert dropres.dino_jit_pre_blend_rmsnorm is True
    assert dropres.dino_grounding_drop_on_label_drop is True
    assert dropres.dino_residual_drop_on_label_drop is False
    assert dropres.use_ddt_head is False
    assert dropres.output_dir == dropres.resume == f"checkpoints/{RMSNORM_DROPRES}"
    assert dropres.start_epoch == 0
    assert {
        key for key in vars(base) if getattr(base, key) != getattr(dropres, key)
    } == {
        "config", "exp_name", "output_dir", "resume", "metrics_log_name",
        "dino_grounding_drop_on_label_drop",
    }


def test_nogamma_config_uses_h200_batch_and_lr_and_preserves_other_settings():
    base, nogamma = _args(RMSNORM_NO_DDT), _args(NOGAMMA_NO_DDT)
    assert nogamma.dino_jit_pre_blend_rmsnorm is True
    assert nogamma.dino_grounding_drop_on_label_drop is False
    assert nogamma.use_ddt_head is False
    assert nogamma.start_epoch == 0
    assert nogamma.output_dir == nogamma.resume == f"checkpoints/{NOGAMMA_NO_DDT}"
    assert nogamma.metrics_log_name == "blend_rmsnorm_nogamma_no_ddt_head_diagnostics"
    assert nogamma.batch_size == 128
    assert nogamma.global_batch_size == 1024
    assert nogamma.accum_iter == 1
    assert nogamma.blr == pytest.approx(5e-5)
    assert nogamma.lr is None
    assert nogamma.blr * nogamma.global_batch_size / 256 == pytest.approx(2e-4)
    assert nogamma.epochs == 80
    assert {
        key for key in vars(base) if getattr(base, key) != getattr(nogamma, key)
    } == {"config", "exp_name", "output_dir", "resume", "metrics_log_name",
          "batch_size", "global_batch_size", "blr"}
    assert _args(NOGAMMA_NO_DDT, "--no-dino_jit_pre_blend_rmsnorm").dino_jit_pre_blend_rmsnorm is False


def test_offline_sampling_preserves_nogamma_settings(tmp_path):
    checkpoint_path = tmp_path / "checkpoint-20.pth"
    checkpoint_path.touch()
    sampling = prepare_config(
        ROOT / "configs" / f"{NOGAMMA_NO_DDT}.yaml",
        checkpoint_path, tmp_path / "sampling", cfg=2.3,
    )
    assert sampling["dino_jit_input_mode"] == "time_blend"
    assert sampling["dino_jit_pre_blend_rmsnorm"] is True
    assert sampling["dino_grounding_drop_on_label_drop"] is False
    assert sampling["use_ddt_head"] is False


def test_offline_sampling_preserves_rmsnorm_dropres_null_label_policy(tmp_path):
    checkpoint_path = tmp_path / "checkpoint-20.pth"
    checkpoint_path.touch()
    sampling = prepare_config(
        ROOT / "configs" / f"{RMSNORM_DROPRES}.yaml",
        checkpoint_path, tmp_path / "sampling", cfg=2.3,
    )
    assert sampling["dino_jit_input_mode"] == "time_blend"
    assert sampling["dino_jit_pre_blend_rmsnorm"] is True
    assert sampling["dino_grounding_drop_on_label_drop"] is True
    assert sampling["use_ddt_head"] is False


@pytest.mark.parametrize("name", [TIME_BLEND, NO_DDT])
def test_offline_sampling_preserves_time_blend_and_cfg_null_policy(tmp_path, name):
    checkpoint_path = tmp_path / "checkpoint-80.pth"
    checkpoint_path.touch()
    sampling = prepare_config(
        ROOT / "configs" / f"{name}.yaml", checkpoint_path, tmp_path / "sampling",
    )
    assert sampling["dino_preprocess_input"] is False
    assert sampling["dino_jit_input_mode"] == "time_blend"
    assert sampling["dino_input_projector"] == "pixeldit2"
    assert sampling["dino_grounding_drop_on_label_drop"] is False
    assert sampling.get("use_ddt_head", True) is (name != NO_DDT)


@pytest.mark.parametrize("obsolete", [
    "report_to", "enable_repa", "dino_aggregation_student_only", "dino_use_teacher",
    "dino_patch_loss_weight", "dino_alignment_time_weight",
    "dino_alignment_time_weight_normalizer", "dino_clamp_input",
    "dino_feature_normalize", "dino_projector_feed_jit",
    "use_time_tokens", "time_token_count",
])
def test_removed_options_are_rejected_by_cli_and_grouped_yaml(obsolete, tmp_path):
    with pytest.raises(SystemExit):
        parse_args_with_config(_active_parser(), [f"--{obsolete}", "true"])
    config = tmp_path / "obsolete.yaml"
    config.write_text(f"model:\n  {obsolete}: true\n", encoding="utf-8")
    with pytest.raises(ValueError, match=obsolete):
        load_grouped_yaml_defaults(_active_parser(), config)


@pytest.mark.parametrize("name,train_native", [
    (BASE, False), (RESIDUAL, True), (TIME_BLEND, True), (NO_DDT, True),
    (NATIVE, True),
    (RMSNORM_DROPRES, True),
    (NOGAMMA_NO_DDT, True),
])
def test_native_patch_embed_trainability_and_residual_initialization(monkeypatch, name, train_native):
    args = _args(name)

    class TinyJiT(nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden_size = 768
            self.num_heads = 12
            self.patch_size = 16
            self.x_embedder = nn.Linear(3, 768)
            self.blocks = nn.ModuleList([nn.Identity() for _ in range(12)])
            self.final_layer = nn.Linear(768, 768)

    class TinyEncoder(nn.Module):
        embed_dim = 768

    class TinyModule(nn.Module):
        def __init__(self, *_args, **_kwargs):
            super().__init__()
            self.proj = nn.Linear(1, 1)

    monkeypatch.setitem(dino_jit_denoiser.JiT_models, args.model, lambda **_: TinyJiT())
    monkeypatch.setattr(dino_jit_denoiser, "JiTDDTHead", TinyModule)
    monkeypatch.setattr(dino_jit_denoiser, "ProjectionLayer", TinyModule)
    monkeypatch.setattr(dino_jit_denoiser, "build_encoder", lambda *_args, **_: TinyEncoder())

    model = DINOJiTDenoiser(args)
    assert all(p.requires_grad == train_native for p in model.net.x_embedder.parameters())
    no_ddt = name in {NO_DDT, RMSNORM_DROPRES, NOGAMMA_NO_DDT}
    assert all(p.requires_grad == no_ddt for p in model.net.final_layer.parameters())
    assert (model.ddt_head is None) is no_ddt
    assert any(key.startswith("ddt_head.") for key in model.state_dict()) is not no_ddt
    if name == RESIDUAL:
        assert model.dino_residual_adapter is not None
        assert sum(p.numel() for p in model.dino_residual_adapter.parameters()) == 1_182_720
        assert model.dino_residual_adapter.fc2.weight.count_nonzero().item() == 0
        assert model.dino_input_projector is None
    elif name in {TIME_BLEND, NO_DDT, RMSNORM_DROPRES, NOGAMMA_NO_DDT}:
        assert model.dino_residual_adapter is None
        assert isinstance(model.dino_input_projector, TimestepProjection)
        assert any(p.requires_grad for p in model.dino_input_projector.parameters())
    elif name == NATIVE:
        assert model.dino_student is None
        assert model.dino_residual_adapter is None
        assert model.dino_input_projector is None
        assert model.dino_student_projector is not None
        assert model.ddt_head is not None
        assert any(p.requires_grad for p in model.dino_student_projector.parameters())
    else:
        assert model.dino_residual_adapter is None
        assert model.dino_input_projector is None

    if name == NOGAMMA_NO_DDT:
        norm = model.dino_jit_pre_blend_norm
        assert isinstance(norm, RMSNorm)
        assert norm.weight is None
        assert list(norm.parameters()) == []
        assert list(norm.buffers()) == []
        assert not any("pre_blend_norm" in key for key in model.state_dict())
