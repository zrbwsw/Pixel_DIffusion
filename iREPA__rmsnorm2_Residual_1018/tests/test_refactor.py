"""Contract and numerical checks for the small extension of original iREPA."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from test_generation_input import tiny_factory


def test_pinned_original_irepa_and_only_requested_config_differences():
    from main import ROOT, parse_args, verify_upstream
    assert verify_upstream() == 23
    base = parse_args(['--config', str(ROOT / 'configs/reverse_b80.yaml')])
    for variant, controls in [('nomls', {'dino_aggregation', 'dino_layers'}), ('pmean0', {'P_mean'})]:
        args = parse_args(['--config', str(ROOT / f'configs/reverse_{variant}_b80.yaml')])
        changes = {k for k in vars(base) if getattr(args, k) != getattr(base, k)}
        assert changes == controls | {'config', 'exp_name', 'output_dir'}
        assert args.batch_size * 8 == args.expected_global_batch_size == 1024
        assert args.blr * 1024 / 256 == pytest.approx(2e-4)
        assert args.enc_type == 'dinov3-vit-b16' and args.encoder_depth == 4
        assert args.dino_time_blend_reverse


@pytest.mark.parametrize('reverse', [False, True])
def test_reverse_blend_keeps_projector_and_jit_timestep(tiny_factory, reverse):
    model = tiny_factory().eval()
    net, adapter = model.net, model.net.generation_input
    adapter.reverse = reverse
    x, t, labels = torch.randn(3, 3, 32, 32), torch.tensor([0., .3, 1.]), torch.tensor([0, 1, 2])
    seen = {}
    def capture(module, inputs):
        seen['tokens'] = inputs[0].detach().clone()
    handle = net.blocks[0].register_forward_pre_hook(capture)
    with torch.no_grad():
        native = adapter.patch_norm(type(net.x_embedder).forward(net.x_embedder, x))
        dino = adapter.dino_norm(adapter.projector(adapter.dino(x), t, (2, 2)))
        wt = t[:, None, None] if reverse else (1-t[:, None, None])
        expected = wt * native + (1-wt) * dino + net.pos_embed
        net(x, t, labels)
    handle.remove()
    torch.testing.assert_close(seen['tokens'], expected)


def test_baseline_factory_matches_original_rng_losses_gradients_adam_ema_and_cfg(tiny_factory):
    from denoiser import Denoiser
    from generation_input import build_denoiser
    from util.misc import add_weight_decay
    # tiny_factory configures the same original JiT class at a small test size.
    torch.manual_seed(901)
    candidate = tiny_factory('baseline')
    rng_after_factory = torch.get_rng_state()
    torch.manual_seed(901)
    original = Denoiser(deepcopy(tiny_factory.args))
    assert torch.equal(rng_after_factory, torch.get_rng_state())
    for name, value in candidate.state_dict().items():
        torch.testing.assert_close(value,original.state_dict()[name],rtol=0,atol=0)
    assert type(candidate) is type(original) is Denoiser
    assert candidate.forward.__func__ is Denoiser.forward
    torch.manual_seed(5)
    torch.nn.init.normal_(candidate.net.final_layer.linear.weight, std=.03)
    original.load_state_dict(candidate.state_dict())
    for model in [candidate, original]:
        model.ema_params1 = [p.detach().clone() for p in model.parameters()]
        model.ema_params2 = [p.detach().clone() for p in model.parameters()]
    optimizers = [torch.optim.AdamW(add_weight_decay(m, 0), lr=2e-4, betas=(.9,.95)) for m in [candidate,original]]
    x, labels, targets = torch.randn(2,3,32,32), torch.tensor([0,2]), [torch.randn(2,4,24)]
    for step in range(2):
        outputs = []
        for model, optimizer in zip([candidate, original], optimizers):
            torch.manual_seed(100+step)
            loss, detail = model(x, labels, targets)
            loss.backward()
            outputs.append((loss.detach(), detail, torch.get_rng_state(), {k:p.grad.clone() for k,p in model.named_parameters() if p.grad is not None}))
            optimizer.step(); optimizer.zero_grad(); model.update_ema()
        torch.testing.assert_close(outputs[0][0],outputs[1][0],rtol=0,atol=0)
        assert outputs[0][1] == outputs[1][1]
        assert torch.equal(outputs[0][2],outputs[1][2])
        for name, value in outputs[0][3].items():
            torch.testing.assert_close(value,outputs[1][3][name],rtol=0,atol=0)
    for a,b in zip(candidate.parameters(),original.parameters()):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    for attr in ['ema_params1','ema_params2']:
        for a,b in zip(getattr(candidate,attr),getattr(original,attr)):
            torch.testing.assert_close(a,b,rtol=0,atol=0)
    for cfg in [1., 2.7]:
        for model in [candidate,original]:
            model.eval(); model.cfg_scale=cfg; model.cfg_interval=(.1,1.)
        for value in [0.,.5,1.]:
            t=torch.full((2,1,1,1),value)
            torch.testing.assert_close(candidate._forward_sample(x,t,labels),original._forward_sample(x,t,labels),rtol=0,atol=0)
        torch.manual_seed(999)
        generated=candidate.generate(labels)
        torch.manual_seed(999)
        torch.testing.assert_close(generated,original.generate(labels),rtol=0,atol=0)


def test_residual_cfg_calls_original_batched_sampler(tiny_factory):
    model = tiny_factory().eval()
    torch.nn.init.normal_(model.net.final_layer.linear.weight,std=.03)
    model.cfg_scale=2.7
    model.cfg_interval=(.1,1.)
    x, t, labels=torch.randn(2,3,32,32),torch.full((2,1,1,1),.5),torch.tensor([0,1])
    with torch.no_grad():
        cond=model.net(x,t.flatten(),labels)[0]
        uncond=model.net(x,t.flatten(),torch.full_like(labels,model.num_classes))[0]
        expected=(uncond+model.cfg_scale*(cond-uncond)-x)/(1-t).clamp_min(model.t_eps)
        actual=model._forward_sample(x,t,labels)
    torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-5)
    assert model.net.generation_input._pending_timestep is None


def test_generation_factory_selects_only_last_norm_for_nomls(monkeypatch):
    import dino_v3_lora
    from generation_input import _build_generation_encoder
    seen={}
    monkeypatch.setattr(dino_v3_lora,'DINOv3PatchTokenEncoder',lambda **kw:seen.update(kw))
    _build_generation_encoder(SimpleNamespace(dino_aggregation='none',dino_layers=None))
    assert seen['aggregation']=='none' and seen['layer_indices'] is None
    assert seen['final_norm_affine'] and not seen['train_norm']
    assert seen['use_lora'] and seen['lora_r']==64
