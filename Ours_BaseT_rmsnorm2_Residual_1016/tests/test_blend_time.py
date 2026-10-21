"""融合变换不能改变原始扩散时间、PgT/JiT 时间条件或 CFG 区间。"""

from pathlib import Path

import pytest
import torch

from blend_time import sigmoid_blend_time
from jit.evaluation.prepare_sampling import prepare_config
from scripts.check_runtime import inspect_runtime, training_args
from scripts.check_training_launch import check_resume
from .test_dino_jit_token_routing import (
    make_blend_router, TrainableScale, TrainableTokens,
)

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "configs/BaseT-Residual(1-t,t)-NoDinoPre-PgT-blend_rmsnorm2-NoDDTHead-B-80.yaml"
SIGMOID = ROOT / "configs/reverse_sigmoid_blend_B80.yaml"


def sigmoid_router(**kwargs):
    router = make_blend_router(reverse=True, **kwargs)
    router.dino_time_blend_transform = "sigmoid"
    router.dino_time_blend_key = 100.0
    router.net.patch_size = 1
    router.dino_input_projector = TrainableScale(0.5)
    return router


def test_default_curve_midpoint_gradient_and_custom_key():
    t = torch.tensor([0., .05, .1, .15, .2], dtype=torch.float64, requires_grad=True)
    before = t.detach().clone()
    f = sigmoid_blend_time(t)
    torch.testing.assert_close(f, torch.tensor(
        [.0000453978687, .006692850924, .5, .993307149076, .9999546021313],
        dtype=torch.float64,
    ))
    f[2].backward()
    assert t.grad[2].item() == 25.0
    assert torch.equal(t.detach(), before)
    torch.testing.assert_close(sigmoid_blend_time(t.detach(), key=20)[1],
                               torch.tensor(.26894142137, dtype=torch.float64))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_sigmoid_is_computed_in_fp32(dtype):
    t = torch.tensor([0., .09, .1, .11, 1.], dtype=dtype)
    f = sigmoid_blend_time(t)
    assert f.dtype == torch.float32 and torch.isfinite(f).all()
    torch.testing.assert_close(f, torch.sigmoid(100 * (t.float() - .1)), rtol=0, atol=0)


def test_legacy_router_and_explicit_linear_remain_bitwise_identical():
    router = make_blend_router(reverse=True)
    t = torch.tensor([0., .1, .9, 1.])
    images = torch.ones(4, 3, 2, 2)
    dino = torch.arange(48, dtype=torch.float32).view(4, 4, 3)
    legacy, _ = router._compose_jit_input_tokens(images, dino, t)
    router.dino_time_blend_transform = "linear"
    router.dino_time_blend_key = 100.0
    explicit, _ = router._compose_jit_input_tokens(images, dino, t)
    expected = t[:, None, None] * 2 + (1 - t[:, None, None]) * dino
    assert torch.equal(legacy, explicit) and torch.equal(explicit, expected)


@pytest.mark.parametrize("reverse", [False, True])
def test_blend_weights_and_parameter_state(reverse):
    router = make_blend_router(reverse=reverse)
    state_before = router.state_dict()
    router.dino_time_blend_transform = "sigmoid"
    router.dino_time_blend_key = 100.0
    t = torch.tensor([.05, .1, .15])
    fused, diag = router._compose_jit_input_tokens(
        torch.zeros(3, 3, 2, 2), torch.full((3, 4, 3), 6.), t,
    )
    f = torch.tensor([.006692850924, .5, .993307149076])[:, None, None]
    jit_weight = f if reverse else 1 - f
    torch.testing.assert_close(fused, (2 * jit_weight + 6 * (1 - jit_weight)).expand_as(fused))
    torch.testing.assert_close(diag["jit_weighted_tokens"] + diag["dino_weighted_tokens"], fused)
    assert state_before.keys() == router.state_dict().keys()


def test_gradients_reach_both_paths_and_pgt_receives_original_t():
    router = sigmoid_router()
    router.net.x_embedder = TrainableTokens(2.)
    t = torch.tensor([.08, .12])
    original_t = t.clone()
    dino = torch.ones(2, 4, 3, requires_grad=True)
    fused, _ = router._compose_jit_input_tokens(torch.zeros(2, 3, 2, 2), dino, t)
    fused.sum().backward()
    for grad in [router.net.x_embedder.value.grad, router.dino_input_projector.scale.grad, dino.grad]:
        assert torch.isfinite(grad).all() and grad.abs().sum() > 0
    assert torch.equal(t, original_t)
    assert torch.equal(router.dino_input_projector.timesteps, original_t)


def test_training_noise_velocity_and_jit_time_are_not_transformed(monkeypatch):
    router = sigmoid_router()
    router.eval()
    router.noise_scale = 1.
    router.t_eps = .05
    router.jit_alignment_depth = 4
    router.dino_zscore_alpha = .8
    raw_t = torch.tensor([.08, .12])
    router.sample_t = lambda n, device=None: raw_t.to(device)
    monkeypatch.setattr(torch, "randn_like", lambda x: torch.full_like(x, .25))
    seen = {}

    def encode_pair(x, z, t):
        seen["z"], seen["teacher_t"] = z.clone(), t.clone()
        return torch.ones(2, 4, 3), torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)

    def jit(tokens, timesteps, labels, noisy_image, alignment_depth):
        seen["jit_t"], seen["tokens"] = timesteps.clone(), tokens.clone()
        return torch.full_like(noisy_image, .1), tokens

    router._encode_dino_pair = encode_pair
    router._jit_forward_from_tokens = jit
    router._project_student_tokens = lambda tokens: tokens
    images = torch.full((2, 3, 2, 2), .8)
    _, losses = router(images, torch.tensor([1, 2]))
    t = raw_t[:, None, None, None]
    torch.testing.assert_close(seen["z"], t * images + (1 - t) * .25)
    torch.testing.assert_close(seen["teacher_t"], t)
    assert torch.equal(seen["jit_t"], raw_t)
    assert torch.equal(router.dino_input_projector.timesteps, raw_t)
    expected_loss = ((images - .1) / (1 - t)).square().mean().item()
    assert losses["denoise_loss"] == pytest.approx(expected_loss)
    f = torch.sigmoid(100 * (raw_t - .1))[:, None, None]
    torch.testing.assert_close(seen["tokens"], (f * 2 + (1 - f) * .5).expand(2, 4, 3))


def test_cfg_interval_and_sampling_velocity_use_original_time():
    router = sigmoid_router()
    router.cfg_scale = 2.3
    router.cfg_interval = (.05, .2)  # 原始 t=.1 在区间内，f(t)=.5 则不在。
    router.t_eps = .05
    seen = {}

    def jit(tokens, timesteps, labels, noisy_image):
        seen["t"], seen["labels"] = timesteps.clone(), labels
        pixels = tokens.transpose(1, 2).reshape_as(noisy_image)
        gain = torch.where(labels == router.num_classes, .5, 1.)[:, None, None, None]
        return pixels * gain

    router._jit_forward_from_tokens = jit
    images = torch.ones(2, 3, 2, 2)
    t = torch.full((2, 1, 1, 1), .1)
    velocity = router._forward_sample(images, t, torch.tensor([1, 2]))
    assert seen["labels"].tolist() == [1, 2, 1000, 1000]
    torch.testing.assert_close(seen["t"], torch.full((4,), .1))
    torch.testing.assert_close(router.dino_input_projector.timesteps, seen["t"])
    # 变换后的融合 token=1.25，cond=1.25，uncond=.625；velocity 分母仍为 1-.1。
    expected = (.625 + 2.3 * (1.25 - .625) - 1.) / .9
    torch.testing.assert_close(velocity, torch.full_like(images, expected))


def test_dropres_override_still_restores_full_jit():
    router = sigmoid_router(drop_grounding=True)
    fused, diag = router._compose_jit_input_tokens(
        torch.ones(2, 3, 2, 2), torch.ones(2, 4, 3), torch.full((2,), .1),
        label_drop_mask=torch.tensor([False, True]),
    )
    torch.testing.assert_close(fused[0], torch.full((4, 3), 1.25))
    torch.testing.assert_close(fused[1], torch.full((4, 3), 2.))
    assert diag["dino_weighted_tokens"][1].count_nonzero() == 0


@pytest.mark.parametrize("overrides", [
    {"dino_time_blend_transform": "sigmoid", "dino_jit_input_mode": "native"},
    {"dino_time_blend_transform": "unknown"},
    {"dino_time_blend_key": float("nan")},
    {"dino_time_blend_key": float("inf")},
])
def test_invalid_settings_rejected_before_model_or_assets_load(overrides):
    args = training_args(["--config", str(BASE)])
    vars(args).update(overrides)
    with pytest.raises(ValueError, match="dino_time_blend"):
        inspect_runtime(args, world_size=4)


def test_config_only_changes_blend_transform_and_sampling_preserves_it(tmp_path):
    base = training_args(["--config", str(BASE)])
    args = training_args(["--config", str(SIGMOID)])
    changed = {k for k in vars(base) if getattr(base, k) != getattr(args, k)}
    assert changed == {
        "config", "exp_name", "output_dir", "resume", "metrics_log_name", "dino_time_blend_transform",
    }
    assert args.dino_time_blend_key == 100 and args.dino_time_blend_reverse
    assert args.batch_size == 256 and args.global_batch_size == 1024 and not args.resume
    ckpt = tmp_path / "checkpoint-80.pth"
    ckpt.touch()
    sampled = prepare_config(SIGMOID, ckpt, tmp_path / "eval", sampler="heun", steps=50)
    assert sampled["dino_time_blend_transform"] == "sigmoid"
    assert sampled["dino_time_blend_key"] == 100
    assert training_args(["--config", str(SIGMOID), "--dino_time_blend_key", "20"]).dino_time_blend_key == 20


def test_resume_rejects_old_linear_checkpoint_without_new_arguments(monkeypatch):
    saved = vars(training_args(["--config", str(BASE)])).copy()
    saved.pop("dino_time_blend_transform")
    saved.pop("dino_time_blend_key")
    saved["lr"] = .0002
    args = training_args(["--config", str(SIGMOID), "--resume", "/existing"])
    monkeypatch.setattr(torch, "load", lambda *a, **k: {"args": saved, "epoch": 10})
    with pytest.raises(ValueError, match="dino_time_blend_transform"):
        check_resume(args, dict(resume_checkpoint="/existing/checkpoint-last.pth",
                                global_batch_size=1024, effective_lr=.0002))


def test_sigmoid_formula_can_compile_without_graph_breaks():
    compiled = torch.compile(sigmoid_blend_time, backend="eager", fullgraph=True)
    t = torch.tensor([.08, .1, .12])
    torch.testing.assert_close(compiled(t, key=100.), sigmoid_blend_time(t), rtol=0, atol=0)
