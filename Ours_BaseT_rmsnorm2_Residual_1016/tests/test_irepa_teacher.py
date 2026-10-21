"""The alignment teacher is independent, frozen and deterministic in training."""

from copy import deepcopy
import os
from pathlib import Path
import subprocess
import sys

import torch
import torch.nn as nn

import irepa_teacher
from dino_encoder import build_dino_student, build_irepa_teacher
from dino_jit_denoiser import DINOJiTDenoiser
from .test_generation_final_norm_affine import pretrained, _args


def test_teacher_factory_imports_without_student_or_lora_modules():
    # A fresh interpreter catches accidental transitive imports hidden by pytest.
    code = '''
import sys
sys.modules["dino_v3_lora"] = None
sys.modules["dino_lora"] = None
import torch.nn as nn
import irepa_teacher
from types import SimpleNamespace
from dino_encoder import build_irepa_teacher
irepa_teacher.load_dino_v3 = lambda **kwargs: nn.Linear(4, 4)
teacher = build_irepa_teacher(SimpleNamespace(dino_model_name="dinov3_vitb16", dino_weights=""))
assert not teacher.training
assert not any(p.requires_grad for p in teacher.parameters())
'''
    root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [sys.executable, "-c", code], cwd=root,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(root)},
        check=True, capture_output=True, text=True, timeout=90,
    )


def test_teacher_reads_only_backbone_identifiers(pretrained, monkeypatch):
    calls = []

    class BackboneOnly:
        dino_model_name = "dinov3_vitb16"
        dino_weights = "pretrained.pth"

        def __getattr__(self, key):
            raise AssertionError(f"Teacher read student option: {key}")

    def load(**kwargs):
        calls.append(kwargs)
        return deepcopy(pretrained)

    monkeypatch.setattr(irepa_teacher, "load_dino_v3", load)
    teacher = build_irepa_teacher(BackboneOnly())
    assert calls == [{"model_name": "dinov3_vitb16", "weights_path": "pretrained.pth"}]
    assert teacher.patch_size == 16 and teacher.embed_dim == 4
    assert not hasattr(teacher, "aggregation")
    assert not hasattr(teacher, "use_lora")
    assert not hasattr(teacher, "final_norm_affine")


def test_teacher_owns_eval_and_freeze_even_under_parent_train(pretrained, monkeypatch):
    backbone = deepcopy(pretrained).train()
    # Model the train-mode randomness of the real DINOv3 RoPE.
    backbone.blocks[0] = nn.Sequential(backbone.blocks[0], nn.Dropout(0.8))
    monkeypatch.setattr(irepa_teacher, "load_dino_v3", lambda **_: backbone)
    teacher = build_irepa_teacher(_args())
    assert not any(module.training for module in teacher.modules())
    assert not any(p.requires_grad for p in teacher.parameters())
    assert teacher.train() is teacher
    parent = nn.ModuleDict({"teacher": teacher})
    parent.train()
    assert parent.training and not any(m.training for m in teacher.modules())
    x = torch.randn(2, 3, 4, dtype=torch.float64, requires_grad=True)
    rng = torch.get_rng_state().clone()
    first, second = teacher.forward_features(x), teacher.forward_features(x)
    assert set(first) == {"x_norm_clstoken", "x_norm_patchtokens"}
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert torch.equal(rng, torch.get_rng_state())
    for value in first.values():
        assert not value.requires_grad and value.grad_fn is None
        assert not torch.is_inference(value)
    assert all(p.grad is None for p in teacher.parameters())


def test_teacher_target_allows_alignment_backward_without_teacher_gradients(pretrained):
    teacher = build_irepa_teacher(_args())
    x = torch.randn(2, 3, 4, dtype=torch.float64, requires_grad=True)
    target = teacher.forward_features(x)["x_norm_patchtokens"]
    projector = nn.Linear(4, 4).double()
    optimizer = torch.optim.AdamW(projector.parameters(), lr=1e-3)
    before = projector.weight.detach().clone()
    loss = (projector(x) - target).square().mean()
    loss.backward()
    assert projector.weight.grad is not None and projector.weight.grad.norm() > 0
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(p.grad is None for p in teacher.parameters())
    optimizer.step()
    assert not torch.equal(before, projector.weight)


def test_teacher_is_separate_from_student_and_excluded_from_checkpoint(pretrained):
    teacher = build_irepa_teacher(_args())
    student = build_dino_student(_args(dino_train_final_norm=True))
    assert {p.data_ptr() for p in teacher.parameters()}.isdisjoint(
        {p.data_ptr() for p in student.parameters()}
    )
    teacher_norm = teacher.model.norm.weight.detach().clone()
    with torch.no_grad():
        student.model.norm.weight.add_(1)
    assert torch.equal(teacher.model.norm.weight, teacher_norm)

    # Preserve the denoiser's established plain-list storage contract.
    denoiser = DINOJiTDenoiser.__new__(DINOJiTDenoiser)
    nn.Module.__init__(denoiser)
    denoiser._teacher_box = [teacher]
    denoiser.dino_student = student
    denoiser.train()
    assert denoiser.teacher is teacher
    assert not any(m.training for m in teacher.modules())
    assert all(name.startswith("dino_student.") for name in denoiser.state_dict())
    assert {id(p) for p in denoiser.parameters()}.isdisjoint(
        {id(p) for p in teacher.parameters()}
    )
