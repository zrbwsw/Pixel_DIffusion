# Independent pre-blend RMSNorm variant

Working-tree snapshot of `../Ours_BaseT_Residual_1008`, created on 2026-10-16.
The original project is unchanged. This directory contains the core Python
modules, JiT training/evaluation code, local DINOv3 source, licenses,
configuration files, runtime helpers, and tests. Checkpoints, training logs,
sample images, analysis artifacts, caches, and pretrained weights were not
copied. DINOv3 source is a real local copy, not a symlink to the old version.

For another server, use the [remote setup guide](../REMOTE_SETUP.md) and
`scripts/train_portable.sh`; the GPU-4567 launcher below is a local preset.
Install this version with `bash scripts/install_requirements.sh` in a
Python 3.10 environment. No sibling method directory is needed at runtime.

## Architecture

The active experiment is
`BaseT-Residual(t,1-t)-NoDinoPre-PgT-blend_rmsnorm2-NoDDTHead-B-80`.
It keeps the RMSNorm-B/NoDDTHead settings, without DropRes:

```text
noisy image -> JiT patchify -----------------> jit_pre_blend_norm  -> (1-t) --+
                                                                          +-> JiT
noisy image -> DINOv3-B/LoRA/MLS -> PgT(t) --> dino_pre_blend_norm -> t ------+

JiT block 4 -> iREPA ConvProj -> frozen clean-image DINOv3-B target
JiT final block -> native JiT final layer -> predicted clean image
```

Both new modules use the existing per-token RMSNorm implementation:

```python
u = token / sqrt(token.square().mean(dim=-1, keepdim=True) + 1e-6)
output = gamma * u
```

Each module has its own learned channel vector, initialized to all ones.
They do not share a parameter or batch statistics. Their state-dict keys are
`jit_pre_blend_norm.weight` and `dino_pre_blend_norm.weight`. This adds 768
trainable parameters at B scale compared with the shared-gain version.

The `dino_jit_pre_blend_rmsnorm` switch is retained: true enables BOTH
independent modules; false makes both paths identity. Norms inside JiT/PgT
are unchanged. This is an independent-learned-gain ablation, NOT an
affine-free or strictly equal-output-norm ablation. Independent gains can
still produce different branch norms after training.

The new config differs from the original RMSNorm-B config only in experiment
paths and structured-metric filename. The architectural change is local to
this version's `dino_jit_denoiser.py`. Inherited configs are retained for
tests/reference; any RMSNorm-enabled config in THIS directory uses the new
independent implementation.

## Training on GPUs 4,5,6,7

Defaults: 80 epochs; per-GPU batch 64; global batch 256; accumulation 1;
learning rate 1e-4 with 5 warmup epochs; online no-CFG heun50 evaluation of
50K images every 20 epochs. The encoder is initialized from the existing
DINOv3-B pretrained weights; the new experiment starts at epoch 0.

Read-only configuration/resource check:

```bash
bash 'scripts/train_BaseT-Residual(t,1-t)-NoDinoPre-PgT-blend_rmsnorm2-NoDDTHead-B-80_4567.sh' --preflight
```

Launch from this directory (the script also works from another directory):

```bash
bash 'scripts/train_BaseT-Residual(t,1-t)-NoDinoPre-PgT-blend_rmsnorm2-NoDDTHead-B-80_4567.sh'
```

The launcher uses the existing `irepa` Python environment and shared assets:

- ImageNet: `/data0/datasets/imagenet`.
- DINOv3-B weights: `/data0/pretrain_models`.
- Online FID/IS references and Inception weights: `../fid_stats`.
- Offline evaluation assets: `../nanogen-evals`.

Paths can be overridden with `PYTHON_BIN`, `IMAGENET_ROOT`,
`PRETRAIN_MODELS_ROOT`, `FID_STATS_ROOT`, and `NANOGEN_EVALS_ROOT`.
Outputs go to this version's own `checkpoints/<experiment>/` directory;
relaunching automatically resumes ITS `checkpoint-last.pth` if present.
Do not point resume at a shared-RMSNorm checkpoint: the parameter keys and
optimizer layout differ. No old checkpoint was copied or migrated.

Training was not launched as part of preparation. The launcher does not
stop or queue behind existing GPU jobs; free GPUs 4-7 before launching.

## Verification

CPU regression tests, including independent parameters/gradients, pre-blend
routing, dropout compatibility, config parsing, and sampling preparation:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=.:jit \
  python -m pytest -q tests
```

Existing token-norm diagnostics remain after the branch-specific RMSNorm
and before time weighting; the two learned gain vectors are saved in both
training and EMA checkpoint state dictionaries.
