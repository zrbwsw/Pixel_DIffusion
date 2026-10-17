# Affine-free DINOv3 final LayerNorm + affine-free pre-blend RMSNorm

Core-code snapshot of `../Ours_BaseT_nogamma_Residual_1016`, created on 2026-10-17.
The source directory and all sibling experiments are unchanged.
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
Config and launcher names are intentionally inherited without changes. The
portable launcher resolves paths relative to THIS directory, so its checkpoints
and logs are isolated from the original `nogamma` experiment.

This keeps the source's affine-free pre-blend RMSNorm and NoDDTHead architecture,
without DropRes. The only model change relative to `nogamma_Residual_1016` is
removing the pretrained final DINOv3 LayerNorm gamma and beta:

```text
noisy image -> JiT patchify -----------------> affine-free RMSNorm -> (1-t) --+
                                                                           +-> JiT
noisy image -> DINOv3-B/LoRA/MLS -> PgT(t) --> affine-free RMSNorm -> t -------+

JiT block 4 -> iREPA ConvProj -> frozen clean-image DINOv3-B target
JiT final block -> native JiT final layer -> predicted clean image
```

### DINOv3 final LayerNorm: follow the RAEv2 default

Immediately after loading the original DINOv3 pretrained weights,
`dino_v3_lora.py` performs the same replacement as RAEv2's
`DINOv3Encoder.load_model`:

```python
self.model.norm = nn.LayerNorm(self.embed_dim, elementwise_affine=False)
```

This preserves per-token LayerNorm, uses the RAEv2 default epsilon of `1e-5`,
and removes both pretrained affine parameters rather than merely freezing them.
It affects BOTH the LoRA student and the frozen teacher because they share the
encoder wrapper. All intermediate feature calls retain `norm=True`; the block
LayerNorms are untouched. The compatibility argument `train_norm` remains but
there are no final-LN parameters to unfreeze, even when it is true.

Student MLS still averages layers `[2,5,8,11]` and adds the last selected layer's
spatial mean; the teacher still uses the normalized raw last layer without MLS.
Teacher features (and consequently the alignment target values) change because
their final affine is now absent; the alignment loss implementation is unchanged.
LoRA injection, patch-embedding trainability, NoDinoPre, PgT, fusion weights,
pre-blend RMSNorm, optimizer, learning rate, batch size and sampling are unchanged.
No RAE latent statistics, `0.485` scaling or new input preprocessing is added.

### Inherited affine-free pre-blend RMSNorm

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

The inherited local `RMSNorm` has an `elementwise_affine` argument, defaulting to
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

In addition, this fork removes `model.norm.weight` and `model.norm.bias` from
each DINOv3 encoder state dict (1536 frozen parameters per B/16 encoder).
The trainable parameter count is unchanged relative to `nogamma_Residual_1016`.
Do not resume that source experiment's checkpoints directly: strict state-dict
loading and EMA parameter layouts are not interchangeable. Start fresh with the
same pretrained DINOv3 weights; resume only this fork's own checkpoints.

## Eight H200 GPUs (0-7)

Defaults: 80 epochs, per-GPU batch 128, global batch 1024, accumulation 1,
base LR 5e-5, effective learning rate 2e-4 after 5 warmup epochs,
and 50K online no-CFG heun50 evaluation
every 20 epochs. The existing DINOv3-B pretrained weights initialize the
encoder. ImageNet, `../fid_stats`, and `../nanogen-evals` are reused in place.
The YAML keeps `lr: null`, so `lr = blr * global_batch_size / 256`.
This fork and its `nogamma_Residual_1016` source have identical optimization
settings for comparison.

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
from PATH. Default resources are `/data0/datasets/imagenet` and
`../pretrain_models`. Override paths with
`PYTHON_BIN`, `IMAGENET_ROOT`, `PRETRAIN_MODELS_ROOT`, `FID_STATS_ROOT`, or
`NANOGEN_EVALS_ROOT`. Local DINOv3 source is explicitly pinned to this version.
On the current server the weights are in `/data0/pretrain_models`, so set
`PRETRAIN_MODELS_ROOT=/data0/pretrain_models` when using the inherited launcher
(its default sibling `../pretrain_models` directory does not exist here).

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
Additional DINOv3 regression tests cover final-LN affine removal for both roles,
unchanged block norms, raw/MLS outputs, state-dict keys and student gradients.

Verified on 2026-10-17 with the `irepa` Python environment:

- Full CPU suite: **104 passed, 4 skipped**; the skips require CUDA RoPE or
  CUDA BF16 autocast. No GPU training was started.
- Real pretrained DINOv3-B CPU forward: student and teacher both return
  `[1,256,768]`; both final LNs have no affine parameters and `eps=1e-5`.
  Student has 48 LoRA modules; teacher has no trainable parameters.
- The inherited 8-GPU launcher passed `--preflight` with
  `PRETRAIN_MODELS_ROOT=/data0/pretrain_models`; output/resume resolve to this
  new project directory, with `training_mode=from_scratch`.
- Relative to the source snapshot, only `dino_v3_lora.py`, this README and
  the added `tests/test_dino_v3_noaffine.py` differ. Configs, launchers and all
  other model files are unchanged. The repository publish allowlist also
  includes this new directory.
