"""CPU integration checks for the optional input, using real JiT/PgT/RMSNorm.

The small DINO substitute avoids downloading/constructing pretrained models.
As in verify_baseline_parity.py's CPU probe, Tensor.cuda is an identity only
within each test. Production source and its CUDA execution are not modified.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]


class FakeGenerationDINO(nn.Module):
    embed_dim = 24
    patch_size = 16

    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, self.embed_dim, 16, stride=16)
        self.norm = nn.LayerNorm(self.embed_dim)
        self.norm.requires_grad_(False)
        self.last_input = None
        self.calls = 0

    def forward(self, x):
        self.last_input = x.detach().clone()
        self.calls += 1
        return self.norm(self.patch_embed(x).flatten(2).transpose(1, 2))


@pytest.fixture
def tiny_factory(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "jit"))
    monkeypatch.setenv("TORCHDYNAMO_DISABLE", "1")
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *args, **kwargs: self)
    from torch import _dynamo
    import generation_input
    import model_jit

    # Disable the original @torch.compile wrappers only in this CPU harness.
    monkeypatch.setattr(_dynamo.config, "disable", True)
    monkeypatch.setattr(generation_input, "_build_generation_encoder", lambda args: FakeGenerationDINO())
    monkeypatch.setitem(
        model_jit.JiT_models,
        "test-tiny",
        lambda **kwargs: model_jit.JiT(
            depth=5, hidden_size=48, num_heads=3, bottleneck_dim=8,
            in_context_len=2, in_context_start=4, patch_size=16, **kwargs,
        ),
    )

    def make(mode="rmsnorm2"):
        args = SimpleNamespace(
            model="test-tiny", img_size=32, class_num=5,
            attn_dropout=0.0, proj_dropout=0.0, encoder_depth=4,
            z_dims=[24], projector_dim=64, projection_layer_type="conv",
            proj_kwargs_kernel_size=3, label_drop_prob=0.1,
            P_mean=-0.8, P_std=0.8, t_eps=0.05, noise_scale=1.0,
            ema_decay1=0.9999, ema_decay2=0.9996, sampling_method="heun",
            num_sampling_steps=2, cfg=1.0, interval_min=0.0, interval_max=1.0,
            projection_loss_type="cosine", proj_coeff="0.5",
            generation_input_mode=mode,
        )
        return generation_input.build_denoiser(args)

    return make


def test_baseline_has_native_classes_projector_and_no_generation_hooks(tiny_factory, monkeypatch):
    import generation_input
    from denoiser import Denoiser
    from model_jit import JiT

    def forbid_generation(args):
        pytest.fail("Baseline attempted to construct a generation encoder")

    monkeypatch.setattr(generation_input, "_build_generation_encoder", forbid_generation)
    model = tiny_factory("baseline")
    assert type(model) is Denoiser
    assert type(model.net) is JiT
    assert not hasattr(model.net, "generation_input")
    assert not any(module._forward_hooks or module._forward_pre_hooks for module in model.modules())
    assert isinstance(model.net.projectors[0].projection_layer, nn.Conv2d)
    assert model.net.projectors[0].projection_layer.kernel_size == (3, 3)
    assert not any("generation_input" in name for name in model.state_dict())


def test_jit_rmsnorm_adds_only_gain_without_changing_baseline_initialization(tiny_factory, monkeypatch):
    import generation_input

    def forbid_generation(args):
        pytest.fail("JiT-only RMSNorm attempted to construct a generation encoder")

    monkeypatch.setattr(generation_input, "_build_generation_encoder", forbid_generation)
    torch.manual_seed(41)
    baseline = tiny_factory("baseline")
    rng = torch.get_rng_state().clone()
    torch.manual_seed(41)
    actual = tiny_factory("jit_rmsnorm")
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    expected_state, actual_state = baseline.state_dict(), actual.state_dict()
    assert actual_state.keys() - expected_state.keys() == {"net.generation_input.patch_norm.weight"}
    assert not expected_state.keys() - actual_state.keys()
    for name, value in expected_state.items():
        torch.testing.assert_close(actual_state[name], value, rtol=0, atol=0)
    gain = actual.net.generation_input.patch_norm.weight
    torch.testing.assert_close(gain, torch.ones(48), rtol=0, atol=0)
    assert gain.requires_grad
    assert not hasattr(actual.net.generation_input, "dino")
    assert not hasattr(actual.net.generation_input, "projector")
    assert len(actual.net.x_embedder._forward_hooks) == 1
    assert not actual.net._forward_pre_hooks


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_jit_rmsnorm_matches_channel_normalization_and_preserves_dtype(tiny_factory, dtype):
    adapter = tiny_factory("jit_rmsnorm").net.generation_input
    x = torch.randn(2, 4, 48).to(dtype)
    with torch.no_grad():
        adapter.patch_norm.weight.copy_(torch.linspace(0.5, 1.5, 48))
    normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
    expected = (adapter.patch_norm.weight * normalized).to(dtype)
    actual = adapter(x)
    assert actual.dtype == dtype
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_jit_rmsnorm_hook_precedes_ape_without_time_or_label_scaling(tiny_factory):
    net = tiny_factory("jit_rmsnorm").eval().net
    noisy = torch.randn(1, 3, 32, 32).expand(3, -1, -1, -1).clone()
    observed = {}

    def save_fused(module, inputs, output):
        observed["fused"] = output.detach().clone()

    def save_block_input(module, inputs):
        observed["block"] = inputs[0].detach().clone()

    handles = [net.x_embedder.register_forward_hook(save_fused),
               net.blocks[0].register_forward_pre_hook(save_block_input)]
    with torch.no_grad():
        raw = type(net.x_embedder).forward(net.x_embedder, noisy)
        expected = net.generation_input.patch_norm(raw)
        image, aligned, original = net(
            noisy, torch.tensor([0.0, 0.35, 1.0]), torch.tensor([0, 1, 5]),
        )
    for handle in handles:
        handle.remove()
    torch.testing.assert_close(observed["fused"], expected, rtol=0, atol=0)
    torch.testing.assert_close(observed["block"], expected + net.pos_embed, rtol=0, atol=0)
    torch.testing.assert_close(observed["fused"][0], observed["fused"][2], rtol=0, atol=0)
    torch.testing.assert_close(aligned[0], net.projectors[0](original[0]), rtol=0, atol=0)
    assert image.shape == noisy.shape


def test_jit_rmsnorm_original_training_ema_sampling_and_state_reload(tiny_factory):
    model = tiny_factory("jit_rmsnorm").train()
    nn.init.normal_(model.net.final_layer.linear.weight, std=0.02)
    gain = model.net.generation_input.patch_norm.weight
    initial_gain = gain.detach().clone()
    parameters = list(model.parameters())
    model.ema_params1 = [p.detach().clone() for p in parameters]
    model.ema_params2 = [p.detach().clone() for p in parameters]
    expected_ema1, expected_ema2 = initial_gain.clone(), initial_gain.clone()
    optimizer = torch.optim.AdamW(parameters, lr=1e-4, betas=(0.9, 0.95), weight_decay=0.0)
    for _ in range(2):
        optimizer.zero_grad()
        loss, metrics = model(torch.randn(2, 3, 32, 32), torch.tensor([1, 2]), [torch.randn(2, 4, 24)])
        assert torch.isfinite(loss)
        assert "cosine" in metrics and "denoise_loss" in metrics
        loss.backward()
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                assert parameter.grad is not None, name
                assert torch.isfinite(parameter.grad).all(), name
        assert gain.grad.abs().sum() > 0
        optimizer.step()
        expected_ema1.mul_(model.ema_decay1).add_(gain.detach(), alpha=1 - model.ema_decay1)
        expected_ema2.mul_(model.ema_decay2).add_(gain.detach(), alpha=1 - model.ema_decay2)
        model.update_ema()
    assert not torch.equal(gain, initial_gain)
    gain_index = next(i for i, p in enumerate(parameters) if p is gain)
    torch.testing.assert_close(model.ema_params1[gain_index], expected_ema1, rtol=0, atol=0)
    torch.testing.assert_close(model.ema_params2[gain_index], expected_ema2, rtol=0, atol=0)
    restored = tiny_factory("jit_rmsnorm").eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    restored.cfg_scale = 2.9
    restored.cfg_interval = (0.1, 1.0)
    generated = restored.generate(torch.tensor([1, 2]))
    assert generated.shape == (2, 3, 32, 32)
    assert torch.isfinite(generated).all()


def test_hook_blends_endpoints_before_original_ape_and_keeps_alignment(tiny_factory):
    model = tiny_factory().eval()
    net, adapter = model.net, model.net.generation_input
    noisy = torch.randn(3, 3, 32, 32)
    t, labels = torch.tensor([0.0, 1.0, 0.35]), torch.tensor([0, 1, 2])
    observed = {}

    def save_fused(module, inputs, output):
        observed["fused"] = output.detach().clone()

    def save_first_block(module, inputs):
        observed["first_block"] = inputs[0].detach().clone()

    handles = [
        net.x_embedder.register_forward_hook(save_fused),
        net.blocks[0].register_forward_pre_hook(save_first_block),
    ]
    with torch.no_grad():
        # Bypass hooks only for this reference raw patch calculation.
        raw_patch = type(net.x_embedder).forward(net.x_embedder, noisy)
        native = adapter.patch_norm(raw_patch)
        projected = adapter.dino_norm(adapter.projector(adapter.dino(noisy), t, (2, 2)))
        expected = (1 - t[:, None, None]) * native + t[:, None, None] * projected
        calls_before = adapter.dino.calls
        image, aligned, original = net(x=noisy, t=t, y=labels)
    for handle in handles:
        handle.remove()

    torch.testing.assert_close(observed["fused"], expected, rtol=0, atol=0)
    torch.testing.assert_close(observed["fused"][0], native[0], rtol=0, atol=0)
    torch.testing.assert_close(observed["fused"][1], projected[1], rtol=0, atol=0)
    torch.testing.assert_close(observed["first_block"], expected + net.pos_embed, rtol=0, atol=0)
    torch.testing.assert_close(adapter.dino.last_input, noisy, rtol=0, atol=0)
    assert adapter.dino.calls == calls_before + 1
    assert adapter._pending_timestep is None
    assert image.shape == noisy.shape
    assert aligned[0].shape == (3, 4, 24)
    assert original[0].shape == (3, 4, 48)
    torch.testing.assert_close(aligned[0], net.projectors[0](original[0]), rtol=0, atol=0)
    assert isinstance(net.projectors[0].projection_layer, nn.Conv2d)


def test_generation_and_original_alignment_parameters_receive_gradients(tiny_factory):
    model = tiny_factory().train()
    net = model.net
    # Test-only nonzero output head exposes upstream image-loss gradients.
    nn.init.normal_(net.final_layer.linear.weight, std=0.02)
    noisy = torch.randn(2, 3, 32, 32)
    image, aligned, _ = net(noisy, torch.tensor([0.25, 0.75]), torch.tensor([1, 2]))
    (image.square().mean() + aligned[0].square().mean()).backward()
    parameters = dict(model.named_parameters())
    required = (
        "net.x_embedder.proj1.weight",
        "net.projectors.0.projection_layer.weight",
        "net.generation_input.dino.patch_embed.weight",
        "net.generation_input.projector.input_proj.weight",
        "net.generation_input.projector.output_proj.weight",
        "net.generation_input.patch_norm.weight",
        "net.generation_input.dino_norm.weight",
    )
    for name in required:
        grad = parameters[name].grad
        assert grad is not None, name
        assert torch.isfinite(grad).all(), name
        assert grad.abs().sum() > 0, name
    assert net.generation_input.dino.norm.weight.grad is None
    assert net.generation_input._pending_timestep is None


def test_failed_forward_clears_timestep_and_next_call_recovers(tiny_factory, monkeypatch):
    net = tiny_factory().net
    adapter = net.generation_input
    noisy, t, labels = torch.randn(1, 3, 32, 32), torch.tensor([0.4]), torch.tensor([1])

    def fail_time_embedding(*args, **kwargs):
        raise RuntimeError("intentional failure before patch embedding")

    with monkeypatch.context() as context:
        context.setattr(net.t_embedder, "forward", fail_time_embedding)
        with pytest.raises(RuntimeError, match="intentional failure"):
            net(noisy, t, labels)
    assert adapter._pending_timestep is None
    with pytest.raises(RuntimeError, match="Call JiT.forward"):
        net.x_embedder(noisy)
    assert adapter._pending_timestep is None

    def fail_generation(*args, **kwargs):
        raise RuntimeError("intentional failure in generation encoder")

    with monkeypatch.context() as context:
        context.setattr(adapter.dino, "forward", fail_generation)
        with pytest.raises(RuntimeError, match="intentional failure"):
            net(noisy, t, labels)
    assert adapter._pending_timestep is None
    assert net(noisy, t, labels)[0].shape == noisy.shape
    assert adapter._pending_timestep is None
