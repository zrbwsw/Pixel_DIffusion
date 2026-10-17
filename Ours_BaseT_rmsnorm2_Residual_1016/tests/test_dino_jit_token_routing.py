import csv
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from dino_jit_denoiser import DINOJiTDenoiser, DINOResidualAdapter
from jit.engine_jit import _append_csv, _iter_named_parameters_by_group, _summarize_train_scalars
from jit_ddt_head import JiTDDTHead
from model_jit import JiT
from util.model_util import RMSNorm


def make_router():
    router = DINOJiTDenoiser.__new__(DINOJiTDenoiser)
    nn.Module.__init__(router)
    router.use_ddt_head = True
    router.dino_jit_pre_blend_rmsnorm = False
    router.jit_pre_blend_norm = nn.Identity()
    router.dino_pre_blend_norm = nn.Identity()
    return router


def enable_blend_rmsnorm(router):
    router.dino_jit_pre_blend_rmsnorm = True
    router.jit_pre_blend_norm = RMSNorm(3)
    router.dino_pre_blend_norm = RMSNorm(3)


class ConstantPatchEmbedder(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, images):
        return torch.full((images.shape[0], 4, 3), self.value, device=images.device)


class ScaleTokens(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, tokens):
        return self.scale * tokens


class TrainableTokens(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(float(value)))

    def forward(self, images):
        return self.value.expand(images.shape[0], 4, 3)


class TrainableScale(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(float(value)))
        self.input_proj = nn.Linear(3, 3, bias=False)
        nn.init.eye_(self.input_proj.weight)

    def forward(self, tokens, timesteps, grid_hw):
        self.timesteps = timesteps.detach().clone()
        self.grid_hw = grid_hw
        return self.scale * self.input_proj(tokens)


class RecordingEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.last_input = None

    def forward(self, images):
        self.last_input = images.detach().clone()
        return images.flatten(2).transpose(1, 2)


class ZeroEmbedder(nn.Module):
    def forward(self, values):
        return torch.zeros(values.shape[0], 3, device=values.device)


class AddBlock(nn.Module):
    def __init__(self, value, sequence_lengths):
        super().__init__()
        self.value = value
        self.sequence_lengths = sequence_lengths

    def forward(self, tokens, condition, rope):
        self.sequence_lengths.append(tokens.shape[1])
        return tokens + self.value


class IdentityHead(nn.Module):
    def forward(self, noisy_image, tokens, t_emb):
        return tokens


class FakeJiT(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = 3
        self.in_context_len = 2
        self.in_context_start = 4
        self.patch_size = 1
        self.sequence_lengths = []
        self.t_embedder = ZeroEmbedder()
        self.y_embedder = ZeroEmbedder()
        self.blocks = nn.ModuleList(
            [AddBlock(float(i + 1), self.sequence_lengths) for i in range(6)]
        )
        self.register_buffer("pos_embed", torch.ones(1, 4, 3))
        self.register_buffer("in_context_posemb", torch.zeros(1, 2, 3))
        self.feat_rope = None
        self.feat_rope_incontext = None

    def unpatchify(self, tokens, patch_size):
        return tokens


def test_residual_adapter_starts_at_exact_zero():
    adapter = DINOResidualAdapter(input_dim=3, hidden_dim=5, output_dim=3)
    output = adapter(torch.randn(2, 4, 3))
    torch.testing.assert_close(output, torch.zeros_like(output))
    torch.testing.assert_close(adapter.fc2.weight, torch.zeros_like(adapter.fc2.weight))


@pytest.mark.parametrize("gate_kind,expected", [
    ("clean_linear", [2.0, 5.0]),
    ("constant_one", [5.0, 5.0]),
])
def test_residual_gate_behavior(gate_kind, expected):
    router = make_router()
    router.dino_jit_input_mode = "gated_residual"
    router.dino_residual_gate = gate_kind
    router.dino_residual_drop_on_label_drop = False
    router.net = nn.Module()
    router.net.x_embedder = ConstantPatchEmbedder(2.0)
    router.dino_residual_adapter = ScaleTokens(3.0)
    fused, diagnostics = router._compose_jit_input_tokens(
        torch.zeros(2, 3, 2, 2), torch.ones(2, 4, 3), torch.tensor([0.0, 1.0])
    )
    torch.testing.assert_close(fused[:, 0, 0], torch.tensor(expected))
    assert diagnostics is not None
    torch.testing.assert_close(diagnostics["gate"][:, 0, 0], torch.tensor([(v - 2) / 3 for v in expected]))


def test_label_drop_zeros_only_residual_path_during_sampling():
    router = make_router()
    router.dino_jit_input_mode = "gated_residual"
    router.dino_residual_gate = "clean_linear"
    router.dino_residual_drop_on_label_drop = True
    router.dino_grounding_drop_on_label_drop = False
    router.dino_preprocess_input = False
    router.num_classes = 1000
    router.dino_student = RecordingEncoder()
    router.net = nn.Module()
    router.net.x_embedder = ConstantPatchEmbedder(2.0)
    router.dino_residual_adapter = ScaleTokens(3.0)
    router._jit_forward_from_tokens = lambda tokens, *_, **__: tokens

    _, tokens = router._predict_clean_from_noisy(
        torch.ones(2, 3, 2, 2), torch.tensor([0.25, 0.75]),
        torch.tensor([5, router.num_classes]),
    )
    torch.testing.assert_close(tokens[:, 0, 0], torch.tensor([2.75, 2.0]))


def make_blend_router(drop_grounding=False, reverse=False):
    router = make_router()
    router.dino_jit_input_mode = "time_blend"
    router.dino_grounding_drop_on_label_drop = drop_grounding
    router.dino_time_blend_reverse = reverse
    router.dino_residual_drop_on_label_drop = False
    router.dino_preprocess_input = False
    router.num_classes = 1000
    router.dino_student = RecordingEncoder()
    router.net = nn.Module()
    router.net.x_embedder = ConstantPatchEmbedder(2.0)
    router.dino_input_projector = None
    router._jit_forward_from_tokens = lambda tokens, *_, **__: tokens
    return router


def test_pre_blend_rmsnorm_parameters_and_updates_are_not_shared():
    router = make_blend_router()
    enable_blend_rmsnorm(router)
    jit_gain = router.jit_pre_blend_norm.weight
    dino_gain = router.dino_pre_blend_norm.weight
    assert jit_gain is not dino_gain
    assert jit_gain.data_ptr() != dino_gain.data_ptr()
    torch.testing.assert_close(jit_gain, torch.ones(3))
    torch.testing.assert_close(dino_gain, torch.ones(3))
    assert set(router.state_dict()) == {
        "jit_pre_blend_norm.weight", "dino_pre_blend_norm.weight",
    }

    optimizer = torch.optim.SGD(router.parameters(), lr=0.1)
    _, diagnostics = router._compose_jit_input_tokens(
        torch.zeros(2, 3, 2, 2), torch.ones(2, 4, 3), torch.full((2,), 0.5),
    )
    diagnostics["native_tokens"].square().mean().backward()
    assert jit_gain.grad is not None
    assert dino_gain.grad is None
    optimizer.step()
    assert not torch.equal(jit_gain, torch.ones(3))
    torch.testing.assert_close(dino_gain, torch.ones(3))


@pytest.mark.parametrize("timestep", [0.0, 0.25, 1.0])
@pytest.mark.parametrize("reverse", [False, True])
def test_time_blend_uses_each_paths_own_rmsnorm_gain(timestep, reverse):
    router = make_blend_router(reverse=reverse)
    enable_blend_rmsnorm(router)
    with torch.no_grad():
        router.jit_pre_blend_norm.weight.fill_(2.0)
        router.dino_pre_blend_norm.weight.fill_(3.0)
    fused, diagnostics = router._compose_jit_input_tokens(
        torch.zeros(2, 3, 2, 2), torch.full((2, 4, 3), 6.0),
        torch.full((2,), timestep),
    )
    torch.testing.assert_close(diagnostics["native_tokens"], torch.full((2, 4, 3), 2.0))
    torch.testing.assert_close(diagnostics["dino_projected_tokens"], torch.full((2, 4, 3), 3.0))
    dino_weight = 1 - timestep if reverse else timestep
    torch.testing.assert_close(fused, torch.full((2, 4, 3), 2.0 + dino_weight))

    fused.sum().backward()
    jit_grad = router.jit_pre_blend_norm.weight.grad
    dino_grad = router.dino_pre_blend_norm.weight.grad
    assert torch.isfinite(jit_grad).all() and torch.isfinite(dino_grad).all()
    assert bool(jit_grad.abs().sum() > 0) is (dino_weight < 1)
    assert bool(dino_grad.abs().sum() > 0) is (dino_weight > 0)


@pytest.mark.parametrize("reverse,expected", [
    (False, [2.0, 3.0, 6.0]), (True, [6.0, 5.0, 2.0]),
])
def test_time_blend_weights_sum_to_one_at_endpoints_and_midpoint(reverse, expected):
    router = make_blend_router(reverse=reverse)
    fused, diagnostics = router._compose_jit_input_tokens(
        torch.zeros(3, 3, 2, 2), torch.full((3, 4, 3), 6.0),
        torch.tensor([0.0, 0.25, 1.0]),
    )
    torch.testing.assert_close(fused[:, 0, 0], torch.tensor(expected))
    torch.testing.assert_close(
        diagnostics["jit_weighted_tokens"] + diagnostics["dino_weighted_tokens"], fused,
    )


def test_time_blend_without_reverse_attribute_keeps_legacy_weights():
    router = make_blend_router()
    del router.dino_time_blend_reverse
    fused, _ = router._compose_jit_input_tokens(
        torch.zeros(3, 3, 2, 2), torch.full((3, 4, 3), 6.0),
        torch.tensor([0.0, 0.25, 1.0]),
    )
    torch.testing.assert_close(fused[:, 0, 0], torch.tensor([2.0, 3.0, 6.0]))


@pytest.mark.parametrize("reverse", [False, True])
def test_time_blend_backpropagates_to_both_paths_and_input_projector(reverse):
    router = make_blend_router(reverse=reverse)
    router.net.x_embedder = TrainableTokens(2.0)
    router.net.patch_size = 1
    router.dino_input_projector = TrainableScale(0.5)
    dino_tokens = torch.ones(2, 4, 3, requires_grad=True)
    fused, _ = router._compose_jit_input_tokens(
        torch.zeros(2, 3, 2, 2), dino_tokens, torch.tensor([0.25, 0.75]),
    )
    fused.sum().backward()
    assert router.net.x_embedder.value.grad.item() > 0
    assert router.dino_input_projector.scale.grad.item() > 0
    assert dino_tokens.grad.abs().sum().item() > 0
    torch.testing.assert_close(router.dino_input_projector.timesteps, torch.tensor([0.25, 0.75]))
    assert router.dino_input_projector.grid_hw == (2, 2)


@pytest.mark.parametrize("drop_grounding", [False, True])
@pytest.mark.parametrize("reverse,expected_cond", [(False, 1.25), (True, 1.75)])
def test_cfg_null_label_uses_dino_by_default_and_native_only_when_enabled(
    drop_grounding, reverse, expected_cond,
):
    router = make_blend_router(drop_grounding, reverse=reverse)
    _, tokens = router._predict_clean_from_noisy(
        torch.ones(2, 3, 2, 2), torch.full((2,), 0.75),
        torch.tensor([5, router.num_classes]),
    )
    expected_null = 2.0 if drop_grounding else expected_cond
    torch.testing.assert_close(tokens[:, 0, 0], torch.tensor([expected_cond, expected_null]))
    assert router.dino_student.last_input is not None


@pytest.mark.parametrize("reverse", [False, True])
def test_rmsnorm_dropres_restores_full_normalized_native_tokens_at_all_timesteps(reverse):
    router = make_blend_router(drop_grounding=True, reverse=reverse)
    enable_blend_rmsnorm(router)
    images = torch.zeros(6, 3, 2, 2)
    dino_tokens = torch.arange(1, 73, dtype=torch.float32).reshape(6, 4, 3)
    timesteps = torch.tensor([0.0, 0.5, 1.0, 0.0, 0.5, 1.0])
    drop_mask = torch.tensor([False, False, False, True, True, True])
    fused, diagnostics = router._compose_jit_input_tokens(
        images, dino_tokens, timesteps, label_drop_mask=drop_mask,
    )
    native = router.jit_pre_blend_norm(router.net.x_embedder(images))
    projected = router.dino_pre_blend_norm(dino_tokens)
    t = timesteps.view(-1, 1, 1)
    expected = t * native + (1 - t) * projected if reverse else (1 - t) * native + t * projected
    torch.testing.assert_close(fused[~drop_mask], expected[~drop_mask])
    torch.testing.assert_close(fused[drop_mask], native[drop_mask])
    torch.testing.assert_close(diagnostics["jit_weighted_tokens"][drop_mask], native[drop_mask])
    assert diagnostics["dino_weighted_tokens"][drop_mask].count_nonzero() == 0
    assert torch.equal(diagnostics["grounding_drop_mask"], drop_mask)


@pytest.mark.parametrize("drop_all", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_rmsnorm_dropres_blocks_dino_gradients_only_for_dropped_samples(drop_all, reverse):
    router = make_blend_router(drop_grounding=True, reverse=reverse)
    enable_blend_rmsnorm(router)
    router.net.patch_size = 1
    router.dino_input_projector = TrainableScale(0.5)
    dino_tokens = torch.arange(1, 25, dtype=torch.float32).reshape(2, 4, 3).requires_grad_()
    drop_mask = torch.tensor([drop_all, True])
    fused, _ = router._compose_jit_input_tokens(
        torch.zeros(2, 3, 2, 2), dino_tokens, torch.full((2,), 0.75),
        label_drop_mask=drop_mask,
    )
    fused.sum().backward()
    assert dino_tokens.grad[drop_mask].count_nonzero() == 0
    projector_grad = router.dino_input_projector.input_proj.weight.grad
    assert projector_grad is not None
    # All trainable paths remain connected even for an all-dropped batch.
    for parameter in router.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
    assert router.jit_pre_blend_norm.weight.grad.abs().sum() > 0
    if drop_all:
        assert projector_grad.count_nonzero() == 0
        assert router.dino_pre_blend_norm.weight.grad.count_nonzero() == 0
    else:
        assert dino_tokens.grad[~drop_mask].abs().sum() > 0
        assert projector_grad.abs().sum() > 0


@pytest.mark.parametrize("timestep", [0.0, 0.75, 1.0])
@pytest.mark.parametrize("reverse", [False, True])
def test_rmsnorm_dropres_sampling_null_label_uses_the_same_native_only_rule(timestep, reverse):
    router = make_blend_router(drop_grounding=True, reverse=reverse)
    enable_blend_rmsnorm(router)
    images = torch.arange(1, 25, dtype=torch.float32).reshape(2, 3, 2, 2)
    _, tokens = router._predict_clean_from_noisy(
        images, torch.full((2,), timestep), torch.tensor([5, router.num_classes]),
    )
    native = router.jit_pre_blend_norm(router.net.x_embedder(images))
    dino = router.dino_pre_blend_norm(images.flatten(2).transpose(1, 2))
    dino_weight = 1 - timestep if reverse else timestep
    torch.testing.assert_close(tokens[0], (1 - dino_weight) * native[0] + dino_weight * dino[0])
    torch.testing.assert_close(tokens[1], native[1])


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("drop_grounding", [False, True])
def test_cfg_velocity_uses_conditional_blend_and_matching_null_branch(reverse, drop_grounding):
    router = make_blend_router(drop_grounding=drop_grounding, reverse=reverse)
    enable_blend_rmsnorm(router)
    router.cfg_scale = 2.7
    router.cfg_interval = (0.0, 1.0)
    router.t_eps = 0.05
    with torch.no_grad():
        router.jit_pre_blend_norm.weight.fill_(2.0)
        router.dino_pre_blend_norm.weight.fill_(3.0)
    seen = {}

    def fake_jit(tokens, timesteps, labels, noisy_image):
        seen["labels"] = labels
        seen["timesteps"] = timesteps
        return tokens.transpose(1, 2).reshape_as(noisy_image)

    router._jit_forward_from_tokens = fake_jit
    images = torch.arange(1, 25, dtype=torch.float32).reshape(2, 3, 2, 2)
    timestep = 0.25
    t = torch.full((2, 1, 1, 1), timestep)
    velocity = router._forward_sample(images, t, torch.tensor([5, 7]))
    native = router.jit_pre_blend_norm(router.net.x_embedder(images))
    dino = router.dino_pre_blend_norm(images.flatten(2).transpose(1, 2))
    dino_weight = 1 - timestep if reverse else timestep
    cond = ((1 - dino_weight) * native + dino_weight * dino).transpose(1, 2).reshape_as(images)
    uncond = native.transpose(1, 2).reshape_as(images) if drop_grounding else cond
    expected = (uncond + router.cfg_scale * (cond - uncond) - images) / (1 - timestep)
    torch.testing.assert_close(velocity, expected)
    torch.testing.assert_close(seen["timesteps"], torch.full((4,), timestep))
    assert torch.equal(seen["labels"], torch.tensor([5, 7, 1000, 1000]))
    # Both conditional and null examples still execute the DINO path.
    assert router.dino_student.last_input.shape[0] == 4


@pytest.mark.parametrize("add_pos,expected", [(False, 10.0), (True, 11.0)])
def test_l4_capture_and_positional_embedding(add_pos, expected):
    router = make_router()
    router.net = FakeJiT()
    router.ddt_head = IdentityHead()
    router.jit_add_pos_embed = add_pos
    patches = torch.zeros(1, 4, 3)
    output, alignment = router._jit_forward_from_tokens(
        patches, torch.zeros(1), torch.zeros(1, dtype=torch.long),
        noisy_image=torch.zeros(1, 3, 2, 2), alignment_depth=4,
    )
    assert router.net.sequence_lengths == [4, 4, 4, 4, 6, 6]
    torch.testing.assert_close(alignment, torch.full_like(patches, expected))
    torch.testing.assert_close(output, torch.full_like(patches, expected + 11))


@pytest.mark.parametrize("alignment_depth", [None, 4])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Native JiT RoPE requires CUDA")
def test_no_ddt_output_matches_native_jit_and_backpropagates(alignment_depth):
    router = make_router()
    router.use_ddt_head = False
    router.ddt_head = None
    router.jit_add_pos_embed = True
    router.net = JiT(
        input_size=4, patch_size=2, hidden_size=48, depth=6, num_heads=4,
        bottleneck_dim=8, in_context_len=2, in_context_start=4, encoder_depth=4,
    ).cuda()
    # Avoid a vacuous equality check against the zero-initialized pixel output.
    with torch.no_grad():
        router.net.final_layer.linear.weight.normal_(std=0.02)
        router.net.final_layer.adaLN_modulation[-1].weight.normal_(std=0.02)
    images = torch.randn(2, 3, 4, 4, device="cuda")
    timesteps = torch.tensor([0.2, 0.8], device="cuda")
    labels = torch.tensor([3, 1000], device="cuda")
    expected, _, _ = router.net(images, timesteps, labels)
    actual = router._jit_forward_from_tokens(
        router.net.x_embedder(images), timesteps, labels, alignment_depth=alignment_depth,
    )
    if alignment_depth is not None:
        actual, alignment = actual
        assert alignment.shape == (2, 4, 48)
    torch.testing.assert_close(actual, expected)
    actual.square().mean().backward()
    for parameter in (
        router.net.final_layer.linear.weight,
        router.net.final_layer.adaLN_modulation[-1].weight,
        router.net.x_embedder.proj1.weight,
        router.net.y_embedder.embedding_table.weight,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def make_preprocess_router(enabled):
    router = make_router()
    router.dino_preprocess_input = enabled
    router.dino_jit_input_mode = "replace"
    router.dino_residual_drop_on_label_drop = False
    router.dino_grounding_drop_on_label_drop = False
    router.dino_student = RecordingEncoder()
    router._teacher_box = [RecordingEncoder()]
    router._teacher_device = torch.device("cpu")
    router.register_buffer("dino_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
    router.register_buffer("dino_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    return router


@pytest.mark.parametrize("enabled", [True, False])
def test_student_toggle_applies_to_training_and_sampling_but_not_teacher(enabled):
    router = make_preprocess_router(enabled)
    router.label_drop_prob = 0.0
    router.num_classes = 1000
    router.noise_scale = 1.0
    router.t_eps = 0.05
    router.dino_zscore_alpha = 0.8
    router.dino_student_projector = nn.Identity()
    router.sample_t = lambda n, device: torch.full((n,), 0.5, device=device)
    noisy_seen = []

    def fake_jit(tokens, timesteps, labels, noisy_image=None, alignment_depth=None):
        noisy_seen.append(noisy_image)
        prediction = torch.zeros_like(noisy_image)
        return (prediction, tokens) if alignment_depth is not None else prediction

    router._jit_forward_from_tokens = fake_jit
    router.jit_alignment_depth = 4
    clean = torch.linspace(-1, 1, steps=24).reshape(2, 3, 2, 2)
    loss, metrics = router(clean, torch.tensor([3, 7]))

    train_input = router._preprocess_for_dino(noisy_seen[0]) if enabled else noisy_seen[0]
    torch.testing.assert_close(router.dino_student.last_input, train_input)
    torch.testing.assert_close(router.teacher.last_input, router._preprocess_for_dino(clean))
    assert router.teacher.last_input.requires_grad is False
    assert loss.item() == pytest.approx(metrics["denoise_loss"] + 0.5 * metrics["alignment_loss"], abs=1e-5)
    assert metrics["alignment_loss_weighted"] == pytest.approx(0.5 * metrics["alignment_loss"])

    noisy = torch.full_like(clean, 0.25)
    router._predict_clean_from_noisy(noisy, torch.full((2,), 0.5), torch.tensor([3, 7]))
    sample_input = router._preprocess_for_dino(noisy) if enabled else noisy
    torch.testing.assert_close(router.dino_student.last_input, sample_input)


def test_t_moment_normalization_matches_diffusion_moments():
    router = make_router()
    router.dino_input_normalization = "t_moment"
    router.dino_preprocess_input = True
    router.noise_scale = 1.0
    router.register_buffer(
        "dino_mean_pm1", torch.tensor([0.0, 0.1, 0.2]).view(1, 3, 1, 1)
    )
    router.register_buffer(
        "dino_std_pm1", torch.tensor([0.5, 0.6, 0.7]).view(1, 3, 1, 1)
    )
    noisy = torch.tensor(
        [
            [[[0.5]], [[0.5]], [[0.5]]],
            [[[0.5]], [[0.5]], [[0.5]]],
        ], dtype=torch.float32,
    )
    timesteps = torch.tensor([0.0, 1.0])
    actual = router._preprocess_noisy_student_for_dino(noisy, timesteps)
    mean_t = torch.tensor(
        [
            [[[0.0]], [[0.0]], [[0.0]]],
            [[[0.0]], [[0.1]], [[0.2]]],
        ]
    )
    std_t = torch.tensor(
        [
            [[[1.0]], [[1.0]], [[1.0]]],
            [[[0.5]], [[0.6]], [[0.7]]],
        ]
    )
    torch.testing.assert_close(actual, (noisy - mean_t) / std_t)


def test_t_moment_student_requires_timesteps():
    router = make_router()
    router.dino_input_normalization = "t_moment"
    router.dino_preprocess_input = True
    router.dino_student = RecordingEncoder()
    with pytest.raises(ValueError, match="requires diffusion timesteps"):
        router._encode_noisy_student(torch.zeros(1, 3, 2, 2))


def test_time_blend_reports_native_projected_weighted_and_fused_norms():
    router = make_preprocess_router(False)
    router.dino_jit_input_mode = "time_blend"
    router.dino_input_projector = None
    router.net = nn.Module()
    router.net.x_embedder = ConstantPatchEmbedder(2.0)
    router.num_classes = 1000
    router.label_drop_prob = 0.0
    router.noise_scale = 1.0
    router.t_eps = 0.05
    router.dino_zscore_alpha = 0.8
    router.dino_student_projector = nn.Identity()
    router.sample_t = lambda n, device: torch.full((n,), 0.5, device=device)
    router.jit_alignment_depth = 4
    router._jit_forward_from_tokens = lambda tokens, _, __, noisy_image, alignment_depth: (
        torch.zeros_like(noisy_image), tokens
    )
    _, metrics = router(torch.zeros(2, 3, 2, 2), torch.tensor([3, 7]))
    for name in (
        "native_jit_patch_token_norm", "dino_projected_token_norm",
        "jit_weighted_token_norm", "dino_weighted_token_norm",
        "jit_blended_token_norm", "dino_grounding_drop_fraction",
    ):
        assert name in metrics
    assert metrics["dino_grounding_drop_fraction"] == 0.0
    assert metrics["jit_blended_token_norm"] == metrics["jit_input_token_norm"]


@pytest.mark.parametrize("reverse", [False, True])
def test_rmsnorm_dropres_training_uses_the_actual_label_drop_mask(reverse):
    router = make_preprocess_router(False)
    router.dino_jit_input_mode = "time_blend"
    router.dino_grounding_drop_on_label_drop = True
    router.dino_time_blend_reverse = reverse
    enable_blend_rmsnorm(router)
    router.dino_input_projector = None
    router.net = nn.Module()
    router.net.x_embedder = ConstantPatchEmbedder(2.0)
    router.num_classes = 1000
    router.noise_scale = 1.0
    router.t_eps = 0.05
    router.dino_zscore_alpha = 0.8
    router.dino_student_projector = nn.Identity()
    router.jit_alignment_depth = 4
    router.sample_t = lambda n, device: torch.full((n,), 0.75, device=device)
    drop_mask = torch.tensor([False, True])
    router.drop_labels_with_mask = lambda labels: (
        labels.masked_fill(drop_mask, router.num_classes), drop_mask,
    )
    seen = {}

    def fake_jit(tokens, timesteps, labels, noisy_image, alignment_depth):
        seen["tokens"] = tokens
        seen["labels"] = labels
        return torch.zeros_like(noisy_image), tokens

    router._jit_forward_from_tokens = fake_jit
    clean = torch.linspace(-1, 1, steps=24).reshape(2, 3, 2, 2)
    loss, metrics = router(clean, torch.tensor([3, 7]))
    native = router.jit_pre_blend_norm(router.net.x_embedder(clean))
    dino = router.dino_pre_blend_norm(
        router.dino_student.last_input.flatten(2).transpose(1, 2),
    )
    dino_weight = 0.25 if reverse else 0.75
    expected_cond = (1 - dino_weight) * native + dino_weight * dino
    torch.testing.assert_close(seen["tokens"][~drop_mask], expected_cond[~drop_mask])
    torch.testing.assert_close(seen["tokens"][drop_mask], native[drop_mask])
    assert torch.equal(seen["labels"], torch.tensor([3, 1000]))
    assert metrics["dino_grounding_drop_fraction"] == 0.5
    assert torch.isfinite(loss)


def test_alignment_is_fixed_negative_cosine_with_spatial_teacher_normalization():
    router = make_router()
    router.dino_zscore_alpha = 0.0
    tokens = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    loss, metrics = router._alignment_loss(tokens, tokens)
    torch.testing.assert_close(loss, torch.tensor(-1.0))
    torch.testing.assert_close(metrics["loss_cosine"], torch.tensor(1.0))


def test_ddt_time_fusion_remains_enabled():
    head = JiTDDTHead.__new__(JiTDDTHead)
    nn.Module.__init__(head)
    head.time_reinject = True
    head.z_proj = nn.Linear(3, 2, bias=False)
    z_t = torch.tensor([[[-2.0, 0.5, 1.0], [1.0, -1.0, 0.0]]])
    t_emb = torch.tensor([[0.5, 1.0, -0.5]])
    expected = head.z_proj(torch.nn.functional.silu(z_t + t_emb.unsqueeze(1)))
    torch.testing.assert_close(head._build_condition(z_t, t_emb), expected)


def test_structured_metrics_use_current_objective_names():
    groups = (
        "total", "jit", "jit_pre_alignment", "jit_post_alignment", "ddt_head",
        "jit_plus_ddt", "lora", "dino_non_lora", "dino_patch_embed",
        "alignment_projector", "input_projector", "residual_adapter",
    )
    norms = {group: (0.0, 0) for group in groups}
    summary = _summarize_train_scalars(
        1.0, {"denoise_loss": 0.25, "alignment_loss": -0.4,
              "alignment_loss_weighted": -0.2}, 1e-4, norms, norms,
    )
    assert summary["denoise_loss"] == 0.25
    assert summary["alignment_loss"] == -0.4
    assert summary["alignment_loss_weighted"] == -0.2
    assert "grad_norm/alignment_projector" in summary


def test_input_projector_has_its_own_training_norm_group():
    router = make_blend_router()
    router.dino_input_projector = TrainableScale(0.5)
    names = [name for name, _ in _iter_named_parameters_by_group(router, "input_projector")]
    assert names == ["dino_input_projector.scale", "dino_input_projector.input_proj.weight"]


def test_structured_csv_expands_schema(tmp_path):
    path = Path(tmp_path) / "metrics.csv"
    _append_csv(path, {"step": 0, "loss": 1.0})
    _append_csv(path, {"step": 1, "loss": 0.9, "cosine": 0.2})
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert rows[0]["cosine"] == ""
    assert rows[1]["cosine"] == "0.2"
