"""Only the generation final-LN affine becomes trainable; keep teacher intact."""
from pathlib import Path

import pytest
import torch

from dino_encoder import build_dino_student, build_irepa_teacher
from jit.util.misc import add_weight_decay
from scripts.check_runtime import inspect_runtime, training_args
from jit.evaluation.prepare_sampling import prepare_config
from .test_generation_final_norm_affine import pretrained, _args

ROOT = Path(__file__).resolve().parents[1]
BASE = 'BaseT-Residual(1-t,t)-NoDinoPre-PgT-blend_rmsnorm2-NoDDTHead-B-80'


def test_only_final_norm_trainability_changes_with_identical_initial_forward(pretrained):
    torch.manual_seed(3)
    frozen = build_dino_student(_args())
    frozen_rng = torch.get_rng_state().clone()
    torch.manual_seed(3)
    trainable = build_dino_student(_args(dino_train_final_norm=True))
    assert torch.equal(torch.get_rng_state(), frozen_rng)
    before = dict(frozen.named_parameters())
    after = dict(trainable.named_parameters())
    assert before.keys() == after.keys()
    changed = {name for name in before if before[name].requires_grad != after[name].requires_grad}
    assert changed == {'model.norm.weight', 'model.norm.bias'}
    for name in before:
        torch.testing.assert_close(before[name], after[name], rtol=0, atol=0)
    x = torch.randn(2, 3, 4, dtype=torch.float64)
    for left, right in zip(frozen.forward_with_last_raw(x), trainable.forward_with_last_raw(x)):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_teacher_is_identical_and_frozen_with_trainable_student_ln(pretrained):
    frozen = build_irepa_teacher(_args())
    teacher = build_irepa_teacher(_args(dino_train_final_norm=True))
    assert not any(p.requires_grad for p in teacher.parameters())
    assert not any(module.training for module in teacher.modules())
    for name, value in frozen.state_dict().items():
        torch.testing.assert_close(teacher.state_dict()[name], value, rtol=0, atol=0)
    x = torch.randn(2, 3, 4, dtype=torch.float64)
    torch.testing.assert_close(teacher.forward_features(x), frozen.forward_features(x), rtol=0, atol=0)


def test_final_ln_receives_gradients_and_optimizer_updates_both_affines(pretrained):
    student = build_dino_student(_args(dino_train_final_norm=True))
    frozen_blocks = {name: p.detach().clone() for name, p in student.model.blocks.named_parameters()}
    norm = student.model.norm
    original = {name: p.detach().clone() for name, p in norm.named_parameters()}
    groups = add_weight_decay(student, .1)
    no_decay = {id(p) for group in groups if group['weight_decay'] == 0 for p in group['params']}
    assert id(norm.weight) in no_decay and id(norm.bias) in no_decay
    optimizer = torch.optim.AdamW(groups, lr=2e-4, betas=(.9, .95))
    student(torch.randn(2, 3, 4, dtype=torch.float64)).square().mean().backward()
    for p in norm.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm() > 0
    optimizer.step()
    for name, p in norm.named_parameters():
        assert not torch.equal(p, original[name])
    for name, p in student.model.blocks.named_parameters():
        assert not p.requires_grad and p.grad is None
        torch.testing.assert_close(p, frozen_blocks[name], rtol=0, atol=0)


@pytest.mark.parametrize('overrides', [dict(dino_generation_final_norm_affine=False), dict(dino_jit_input_mode='native')])
def test_preflight_rejects_incompatible_final_ln_controls(overrides):
    args = training_args([])
    args.dino_train_final_norm = True
    for key, value in overrides.items():
        setattr(args, key, value)
    with pytest.raises(ValueError, match='dino_train_final_norm requires'):
        inspect_runtime(args, world_size=8)


def test_config_is_baseline_recipe_with_eight_gpu_batch_and_only_new_control(tmp_path):
    baseline = training_args(['--config', str(ROOT / 'checkpoints' / BASE / 'config_base.yaml'), '--batch_size', '128'])
    config = ROOT / 'configs/reverse_train_final_norm_B80_8gpu.yaml'
    args = training_args(['--config', str(config)])
    assert {key for key in vars(baseline) if getattr(baseline, key) != getattr(args, key)} == {
        'config', 'exp_name', 'output_dir', 'resume', 'metrics_log_name', 'dino_train_final_norm',
    }
    assert args.dino_train_final_norm and args.dino_generation_final_norm_affine
    assert args.batch_size * 8 * args.accum_iter == args.global_batch_size == 1024
    assert args.blr * 1024 / 256 == pytest.approx(2e-4)
    assert args.resume == '' and args.epochs == 80
    assert not training_args(['--config', str(config), '--no-dino_train_final_norm']).dino_train_final_norm
    assert not training_args([]).dino_train_final_norm
    checkpoint = tmp_path / 'checkpoint-80.pth'
    checkpoint.touch()
    sampling = prepare_config(config, checkpoint, tmp_path / 'eval')
    assert sampling['dino_train_final_norm'] is True
    args.dino_generation_final_norm_affine = False
    with pytest.raises(ValueError, match='dino_train_final_norm requires'):
        inspect_runtime(args, world_size=8)
