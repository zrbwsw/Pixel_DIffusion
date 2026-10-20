"""CPU checks for removing only the baseline's generation branch."""

import importlib.util
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest
import torch
import torch.nn as nn

from rmsnorm2_jit_only import DINOJiTDenoiser, RMSNorm2JitOnlyDenoiser
from util.model_util import RMSNorm


def _real_rope(**overrides):
    source = Path(__file__).resolve().parents[1] / "SSL_code/dinov3/dinov3/layers/rope_position_encoding.py"
    spec = importlib.util.spec_from_file_location("parity_rope", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    kwargs = dict(embed_dim=16, num_heads=1, dtype=torch.float32, rescale_coords=2)
    kwargs.update(overrides)
    return module.RopePositionEmbedding(**kwargs)


class _PatchEmbed(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(3, 4, 1)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class _Generation(nn.Module):
    def __init__(self):
        super().__init__()
        self.aggregation = "mls"
        self._layer_indices = [2, 5, 8, 11]
        self._total_blocks = 12
        self.model = nn.Module()
        self.model.blocks = nn.ModuleList([nn.Linear(4, 4) for _ in range(12)])
        self.model.rope_embed = _real_rope()

    def forward(self, *args, **kwargs):
        raise AssertionError("Generation DINO forward must never run.")


def _fake_baseline_init(self, args):
    nn.Module.__init__(self)
    self.net = nn.Module()
    self.net.x_embedder = _PatchEmbed()
    self.net.final_layer = nn.Linear(4, 4)
    teacher = nn.Linear(3, 4).requires_grad_(False).eval()
    self._teacher_box = [teacher]
    self._teacher_device = None
    self.dino_student = _Generation()
    self.dino_student_projector = nn.Linear(4, 4)
    self.dino_input_projector = nn.Linear(4, 4)
    self.jit_pre_blend_norm = RMSNorm(4)
    self.dino_pre_blend_norm = RMSNorm(4)
    self.dino_jit_input_mode = "time_blend"
    self.dino_jit_pre_blend_rmsnorm = True
    self.dino_residual_adapter = None
    self.dino_residual_drop_on_label_drop = False
    self.dino_grounding_drop_on_label_drop = False
    self.num_classes = 1000
    self.use_ddt_head = False


def _args(**overrides):
    values = dict(
        dino_generation_path_enabled=False, dino_jit_input_mode="time_blend",
        dino_jit_pre_blend_rmsnorm=True, dino_input_projector="pixeldit2",
        use_ddt_head=False,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.fixture
def fake_baseline(monkeypatch):
    monkeypatch.setattr(DINOJiTDenoiser, "__init__", _fake_baseline_init)


def test_construct_then_remove_keeps_common_weights_teacher_and_rng(fake_baseline):
    torch.manual_seed(2026)
    baseline = DINOJiTDenoiser(_args())
    expected_rng = torch.get_rng_state().clone()
    torch.manual_seed(2026)
    ablation = RMSNorm2JitOnlyDenoiser(_args())
    assert torch.equal(torch.get_rng_state(), expected_rng)
    removed_prefixes = ("dino_student.", "dino_input_projector.", "dino_pre_blend_norm.")
    expected_keys = {key for key in baseline.state_dict() if not key.startswith(removed_prefixes)}
    assert set(ablation.state_dict()) == expected_keys
    for key, actual in ablation.state_dict().items():
        torch.testing.assert_close(actual, baseline.state_dict()[key], rtol=0, atol=0)
    for key, actual in ablation.teacher.state_dict().items():
        torch.testing.assert_close(actual, baseline.teacher.state_dict()[key], rtol=0, atol=0)
    assert not any(p.requires_grad for p in ablation.teacher.parameters())
    assert ablation.dino_student is None and ablation.dino_input_projector is None
    assert isinstance(ablation.dino_pre_blend_norm, nn.Identity)
    assert ablation.jit_pre_blend_norm.weight.requires_grad
    assert ablation.dino_student_projector.weight.requires_grad
    assert not any(key.startswith(removed_prefixes) for key, _ in ablation.named_parameters())


def test_no_input_norm_preserves_all_other_initial_weights_teacher_and_rng(fake_baseline):
    torch.manual_seed(541)
    control = RMSNorm2JitOnlyDenoiser(_args())
    expected_rng = torch.get_rng_state().clone()
    torch.manual_seed(541)
    candidate = RMSNorm2JitOnlyDenoiser(_args(jit_input_rmsnorm=False))
    assert torch.equal(torch.get_rng_state(), expected_rng)
    assert isinstance(candidate.jit_pre_blend_norm, nn.Identity)
    assert set(control.state_dict()) - set(candidate.state_dict()) == {"jit_pre_blend_norm.weight"}
    for key, value in candidate.state_dict().items():
        torch.testing.assert_close(value, control.state_dict()[key], rtol=0, atol=0)
    for key, value in candidate.teacher.state_dict().items():
        torch.testing.assert_close(value, control.teacher.state_dict()[key], rtol=0, atol=0)
    assert not any(p.requires_grad for p in candidate.teacher.parameters())


@pytest.mark.parametrize("t", [0.0, 0.3, 1.0])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_no_input_norm_uses_raw_tokens_and_patch_gradients(fake_baseline, t, dtype):
    model = RMSNorm2JitOnlyDenoiser(_args(jit_input_rmsnorm=False))
    image = torch.randn(2, 3, 2, 2)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=dtype == torch.bfloat16):
        raw = model.net.x_embedder(image)
        tokens, diagnostic = model._compose_jit_input_tokens(
            image, None, torch.full((2, 1, 1, 1), t), torch.tensor([False, True]))
        torch.testing.assert_close(tokens, raw, rtol=0, atol=0)
        assert tokens.dtype == dtype
        assert diagnostic["dino_weighted_tokens"].count_nonzero() == 0
        loss = model.dino_student_projector(tokens).square().mean()
    loss.backward()
    assert model.net.x_embedder.proj.weight.grad is not None
    assert torch.isfinite(model.net.x_embedder.proj.weight.grad).all()
    assert model.net.x_embedder.proj.weight.grad.abs().sum() > 0
    assert not any(name.startswith("jit_pre_blend_norm.") for name, _ in model.named_parameters())


@pytest.mark.parametrize("t", [0.0, 0.3, 1.0])
def test_jit_weight_one_keeps_learned_rmsnorm_for_all_labels_and_timesteps(fake_baseline, t):
    model = RMSNorm2JitOnlyDenoiser(_args())
    with torch.no_grad():
        model.jit_pre_blend_norm.weight.copy_(torch.tensor([0.5, 2.0, 1.5, 3.0]))
    image = torch.randn(2, 3, 2, 2)
    raw = model.net.x_embedder(image)
    expected = model.jit_pre_blend_norm(raw)
    tokens, diagnostics = model._compose_jit_input_tokens(
        image, None, torch.full((2, 1, 1, 1), t), torch.tensor([False, True]),
    )
    torch.testing.assert_close(tokens, expected, rtol=0, atol=0)
    assert not torch.allclose(tokens, raw)
    assert diagnostics["dino_weighted_tokens"].count_nonzero() == 0
    assert diagnostics["grounding_drop_mask"].all()
    loss = model.dino_student_projector(tokens).square().sum()
    loss.backward()
    for parameter in (
        model.net.x_embedder.proj.weight, model.jit_pre_blend_norm.weight,
        model.dino_student_projector.weight,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0


@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("extra", [{}, {"shift_coords": 0.3, "jitter_coords": 1.5}])
def test_skipped_student_preserves_real_rope_rng(fake_baseline, training, dtype, extra):
    model = RMSNorm2JitOnlyDenoiser(_args())
    # Capture a real vendored RoPE with nondefault settings to guard against
    # hard-coding one draw, float32, or ignoring shift/jitter operations.
    student = _Generation()
    student.model.rope_embed = _real_rope(dtype=dtype, **extra)
    model.dino_student = student
    model.dino_input_projector = nn.Linear(4, 4)
    model._capture_skipped_generation_rng()
    rope = student.model.rope_embed.train(training)
    model.dino_student = None
    model.dino_input_projector = None
    model.train(training)
    torch.manual_seed(731)
    for _ in range(12):
        rope(H=2, W=2)
    expected_rng = torch.get_rng_state().clone()
    torch.manual_seed(731)
    assert model._encode_noisy_student(torch.empty(2, 3, 2, 2), torch.zeros(2)) is None
    assert torch.equal(torch.get_rng_state(), expected_rng)


def test_reference_train_and_sampling_boundaries_use_disabled_branch(fake_baseline):
    model = RMSNorm2JitOnlyDenoiser(_args()).eval()
    seen_teacher = []

    def encode_teacher(self, x):
        seen_teacher.append(x)
        return self.teacher(x.flatten(2).transpose(1, 2))

    def predict(self, tokens, timesteps, labels, noisy_image=None):
        return self.net.final_layer(tokens) + labels[:, None, None].to(tokens) / 1000

    model._encode_teacher_target = MethodType(encode_teacher, model)
    model._jit_forward_from_tokens = MethodType(predict, model)
    clean, noisy = torch.randn(2, 3, 2, 2), torch.randn(2, 3, 2, 2)
    times = torch.tensor([0.2, 0.8])
    student, target = model._encode_dino_pair(clean, noisy, times)
    assert student is None and len(seen_teacher) == 1
    torch.testing.assert_close(target, model.teacher(clean.flatten(2).transpose(1, 2)))
    labels = torch.tensor([17, 1000])
    predicted, tokens = model._predict_clean_from_noisy(noisy, times, labels)
    assert len(seen_teacher) == 1  # Sampling must not execute the frozen teacher.
    expected_tokens = model.jit_pre_blend_norm(model.net.x_embedder(noisy))
    torch.testing.assert_close(tokens, expected_tokens)
    torch.testing.assert_close(predicted, predict(model, expected_tokens, times, labels))


@pytest.mark.parametrize("overrides", [
    {"dino_generation_path_enabled": True}, {"dino_jit_input_mode": "native"},
    {"dino_jit_pre_blend_rmsnorm": False}, {"dino_input_projector": "none"},
    {"use_ddt_head": True},
])
def test_rejects_changes_to_the_rmsnorm2_control(fake_baseline, overrides):
    with pytest.raises(ValueError):
        RMSNorm2JitOnlyDenoiser(_args(**overrides))


@pytest.mark.parametrize("mutation", ["dropout", "stochastic_depth", "layers", "depth"])
def test_rejects_unmodelled_generation_rng(fake_baseline, mutation):
    model = RMSNorm2JitOnlyDenoiser(_args())
    model.dino_student = _Generation()
    model.dino_input_projector = nn.Linear(4, 4)
    if mutation == "dropout":
        model.dino_student.extra = nn.Dropout(0.1)
    elif mutation == "stochastic_depth":
        model.dino_student.model.blocks[0].sample_drop_ratio = 0.2
    elif mutation == "layers":
        model.dino_student._layer_indices = [2, 5, 8]
    else:
        model.dino_student.model.blocks = nn.ModuleList([nn.Linear(4, 4)])
    with pytest.raises(ValueError):
        model._capture_skipped_generation_rng()
