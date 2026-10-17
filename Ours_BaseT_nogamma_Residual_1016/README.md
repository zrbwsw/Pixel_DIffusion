# Affine-free pre-blend RMSNorm variant

Core-code snapshot of `../Ours_BaseT_Residual_1008`, created on 2026-10-16.
The original and `../Ours_BaseT_rmsnorm2_Residual_1016` are unchanged.
Checkpoints, logs, sample images, pretrained weights, caches, and analysis
artifacts were not copied. JiT, local DINOv3 source, licenses, configs,
runtime helpers, and regression tests are included.

For another server, use the [remote setup guide](../REMOTE_SETUP.md) and
`scripts/train_portable.sh`; the GPU-01234567 launcher below is an 8-H200 preset.
Install this version with `bash scripts/install_requirements.sh` in a
Python 3.10 environment. No sibling method directory is needed at runtime.

## Architecture

Active experiment:
`BaseT-Residual(t,1-t)-NoDinoPre-PgT-blend_rmsnorm_nogamma-NoDDTHead-B-80`.
This keeps the RMSNorm-B/NoDDTHead baseline, without DropRes, and removes
ONLY the learned gain from the pre-blend norm:

```text
noisy image -> JiT patchify -----------------> affine-free RMSNorm -> (1-t) --+
                                                                           +-> JiT
noisy image -> DINOv3-B/LoRA/MLS -> PgT(t) --> affine-free RMSNorm -> t -------+

JiT block 4 -> iREPA ConvProj -> frozen clean-image DINOv3-B target
JiT final block -> native JiT final layer -> predicted clean image
```

For each `[B, N, C]` token tensor:

```python
input_dtype = tokens.dtype
x = tokens.float()
x = x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + 1e-6)
return x.to(input_dtype)
```

There is no gamma, beta, parameter, buffer, or running statistic in this
pre-blend layer. A single stateless module is called separately for both
branches. One reused instance and two identical instances are numerically
equivalent; there is no shared or independently learned normalization.
Gradients still pass through the normalization to both upstream paths.

Only the hidden dimension is normalized, not batch, patches, or the two
paths together. For a nonzero token with mean squared activation `m`:

```text
||normalized_token||_2 = sqrt(C * m / (m + eps))
```

At B scale this is approximately `sqrt(768) = 27.7128` unless the token is
near zero. Zero tokens remain zero. Epsilon and reduced-precision rounding
prevent a claim of exact equal norms for arbitrary inputs. Time weighting
and vector addition still change the weighted/fused norms, as intended.

## Scope and checkpoint behavior

The local `RMSNorm` gains an `elementwise_affine` argument, defaulting to
true. Existing JiT/PgT internal RMSNorms retain their original learned
weights. Only `dino_jit_pre_blend_norm` explicitly uses
`elementwise_affine=False`. The existing
`dino_jit_pre_blend_rmsnorm` flag enables/disables this parameter-free layer.

Compared with the shared-gamma B baseline, this removes 768 trainable
parameters; compared with the two-gamma version, it removes 1536. The
pre-blend layer has no entries in the state dict or EMA parameter lists.
Inherited configs are kept for tests/reference; every RMSNorm-enabled
config in THIS directory uses the affine-free pre-blend implementation.

The active config uses global batch 1024 and base LR 5e-5 (effective LR 2e-4),
instead of the original RMSNorm-B YAML's batch 256 and LR 1e-4, as well as
new experiment paths and metric filenames. It starts at epoch 0 and does NOT
reuse old training/optimizer states. Relaunching resumes only the new
experiment's own `checkpoint-last.pth` if present. Old affine checkpoints
must not be used as a drop-in resume because their parameter layouts differ.

## Eight H200 GPUs (0-7)

Defaults: 80 epochs, per-GPU batch 128, global batch 1024, accumulation 1,
base LR 5e-5, effective learning rate 2e-4 after 5 warmup epochs,
and 50K online no-CFG heun50 evaluation
every 20 epochs. The existing DINOv3-B pretrained weights initialize the
encoder. ImageNet, `../fid_stats`, and `../nanogen-evals` are reused in place.
The YAML keeps `lr: null`, so `lr = blr * global_batch_size / 256`.
Both variants have identical optimization settings for comparison.

From this directory, check resources without loading a model or starting
training:

```bash
bash 'scripts/train_BaseT-Residual(t,1-t)-NoDinoPre-PgT-blend_rmsnorm_nogamma-NoDDTHead-B-80_01234567.sh' --preflight
```

Launch:

```bash
bash 'scripts/train_BaseT-Residual(t,1-t)-NoDinoPre-PgT-blend_rmsnorm_nogamma-NoDDTHead-B-80_01234567.sh'
```

The script resolves its own project root, so it also works from another
working directory. Activate `irepa` first: the default interpreter is `python`
from PATH. Default resources are `/mnt/shared-storage-user/datafactory-share/gushishen/imagenet` and
`../pretrain_models`. Override paths with
`PYTHON_BIN`, `IMAGENET_ROOT`, `PRETRAIN_MODELS_ROOT`, `FID_STATS_ROOT`, or
`NANOGEN_EVALS_ROOT`. Local DINOv3 source is explicitly pinned to this version.

Training is not started by preparation or preflight. The script neither
stops existing jobs nor queues behind them; free GPUs 0-7 before launching.
Do not run both experiments simultaneously on the same eight GPUs. Preflight
reports batch and effective LR but does not verify H200 memory capacity.
If this experiment already has an old batch-256 checkpoint, use a new
output directory and `--resume ''` for a clean batch-1024 experiment.

## Tests

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=.:jit \
  python -m pytest -q tests
```

Tests cover absence of norm parameters/buffers, last-dimension-only
normalization, shared/separate-instance equivalence, input gradients,
zero/small tokens, dtype preservation, unchanged affine defaults, branch
routing, config parsing, and sampling settings.
