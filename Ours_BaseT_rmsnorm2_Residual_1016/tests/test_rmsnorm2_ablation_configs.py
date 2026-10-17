"""Keep the three new experiments controlled against the trained rmsnorm2 baseline."""
from pathlib import Path

import pytest

from scripts.check_runtime import training_args
from jit.evaluation.prepare_sampling import prepare_config


ROOT = Path(__file__).resolve().parents[1]
BASE = "BaseT-Residual(t,1-t)-NoDinoPre-PgT-blend_rmsnorm2-NoDDTHead-B-80"
CASES = [
    (BASE.replace("-NoDDTHead", "-NoAffine-NoDDTHead"), "dino_generation_final_norm_affine", False),
    (BASE.replace("(t,1-t)", "(1-t,t)"), "dino_time_blend_reverse", True),
    (BASE.replace("-NoDDTHead", "-DropRes-NoDDTHead"), "dino_grounding_drop_on_label_drop", True),
]


@pytest.mark.parametrize("name,control,value", CASES)
def test_ablation_changes_only_one_model_control(name, control, value):
    base = training_args(["--config", str(ROOT / "configs" / f"{BASE}.yaml"), "--batch_size", "256"])
    args = training_args(["--config", str(ROOT / "configs" / f"{name}.yaml")])
    assert getattr(args, control) is value
    changed = {k for k in vars(base) if getattr(base, k) != getattr(args, k)}
    assert changed == {"config", "exp_name", "output_dir", "resume", "metrics_log_name", control}
    assert args.output_dir == args.resume == f"checkpoints/{name}"
    assert args.batch_size * 4 * args.accum_iter == args.global_batch_size == 1024
    assert args.blr * args.global_batch_size / 256 == pytest.approx(2e-4)
    assert args.epochs == 80 and args.warmup_epochs == 5


@pytest.mark.parametrize("name,control,value", CASES)
def test_sampling_preserves_ablation_control(tmp_path, name, control, value):
    ckpt = tmp_path / "checkpoint-80.pth"
    ckpt.touch()
    prepared = prepare_config(ROOT / "configs" / f"{name}.yaml", ckpt, tmp_path / "sampling")
    assert prepared[control] is value


def test_new_controls_can_be_overridden_explicitly():
    args = training_args([
        "--config", str(ROOT / "configs" / f"{BASE}.yaml"),
        "--no-dino_generation_final_norm_affine", "--dino_time_blend_reverse",
    ])
    assert args.dino_generation_final_norm_affine is False
    assert args.dino_time_blend_reverse is True
