# Independent pre-blend RMSNorm variant

Working-tree snapshot of `../Ours_BaseT_Residual_1008`, created on 2026-10-16.
The original project is unchanged. This directory contains the core Python
modules, JiT training/evaluation code, local DINOv3 source, licenses,
configuration files, runtime helpers, and tests. Checkpoints, training logs,
sample images, analysis artifacts, caches, and pretrained weights were not
copied. DINOv3 source is a real local copy, not a symlink to the old version.

For another server, use the [remote setup guide](../REMOTE_SETUP.md) and
`scripts/train_portable.sh`; the GPU-01234567 launcher below is an 8-H200 preset.
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

The active config uses global batch 1024 and base LR 5e-5 (effective LR 2e-4),
instead of the original RMSNorm-B config's batch 256 and LR 1e-4, as well as
new experiment paths and a structured-metric filename. The architectural change is local to
this version's `dino_jit_denoiser.py`. Inherited configs are retained for
tests/reference; any RMSNorm-enabled config in THIS directory uses the new
independent implementation.

## Training on eight H200 GPUs (0-7)

Defaults: 80 epochs; per-GPU batch 128; global batch 1024; accumulation 1;
base LR 5e-5 and effective learning rate 2e-4 after 5 warmup epochs;
online no-CFG heun50 evaluation of
50K images every 20 epochs. The encoder is initialized from the existing
DINOv3-B pretrained weights; the new experiment starts at epoch 0.
The YAML keeps `lr: null`, so `lr = blr * global_batch_size / 256`.
Both variants have identical optimization settings for comparison.

Read-only configuration/resource check:

```bash
bash 'scripts/train_BaseT-Residual(t,1-t)-NoDinoPre-PgT-blend_rmsnorm2-NoDDTHead-B-80_01234567.sh' --preflight
```

Launch from this directory (the script also works from another directory):

```bash
bash 'scripts/train_BaseT-Residual(t,1-t)-NoDinoPre-PgT-blend_rmsnorm2-NoDDTHead-B-80_01234567.sh'
```

Activate the `irepa` environment first; the launcher uses `python` from PATH
unless `PYTHON_BIN` is set. Default resource paths:

- ImageNet: `/mnt/shared-storage-user/datafactory-share/gushishen/imagenet`.
- DINOv3-B weights: `../pretrain_models` (set `PRETRAIN_MODELS_ROOT` to the actual location).
- Online FID/IS references and Inception weights: `../fid_stats`.
- Offline evaluation assets: `../nanogen-evals`.

Paths can be overridden with `PYTHON_BIN`, `IMAGENET_ROOT`,
`PRETRAIN_MODELS_ROOT`, `FID_STATS_ROOT`, and `NANOGEN_EVALS_ROOT`.
Outputs go to this version's own `checkpoints/<experiment>/` directory;
relaunching automatically resumes ITS `checkpoint-last.pth` if present.
Do not point resume at a shared-RMSNorm checkpoint: the parameter keys and
optimizer layout differ. No old checkpoint was copied or migrated.

Training was not launched as part of preparation. The launcher does not
stop or queue behind existing GPU jobs; free GPUs 0-7 before launching.
Do not run both experiments simultaneously on the same eight GPUs. Preflight
reports batch and effective LR but does not verify H200 memory capacity.
If this experiment already has an old batch-256 checkpoint, use a new
output directory and `--resume ''` for a clean batch-1024 experiment.

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
