# Generation DINOv3 without final-LN affine + unchanged iREPA regularization encoder

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
without DropRes. The two DINOv3 encoders have different functions; they are not
a student/teacher pair. The generation encoder shapes the generative input
representation, while the independent iREPA encoder supplies training-only
representation regularization. Only the generation encoder removes the
pretrained final DINOv3 LayerNorm gamma and beta:

```text
noisy image -> JiT patchify -----------------> affine-free RMSNorm -> (1-t) --+
                                                                           +-> JiT
noisy image -> generation DINOv3-B/LoRA/MLS -> PgT(t) -> RMSNorm -> t --------+

clean image -> frozen iREPA DINOv3-B -> regularization features
JiT block 4 -> alignment ConvProj ----> existing iREPA alignment loss
JiT final block -> native JiT final layer -> predicted clean image
```

### Generation encoder: follow the RAEv2 final-LN default, with LoRA

Immediately after loading the original DINOv3 pretrained weights,
`dino_v3_lora.py` performs the same replacement as RAEv2's
`DINOv3Encoder.load_model`:

```python
self.model.norm = nn.LayerNorm(self.embed_dim, elementwise_affine=False)
```

This preserves per-token LayerNorm, uses the RAEv2 default epsilon of `1e-5`,
and removes both pretrained affine parameters rather than merely freezing them.
This replacement applies ONLY to the generation encoder. All its intermediate
feature calls retain `norm=True`; the block LayerNorms are untouched. The
compatibility argument `train_norm` remains but there are no generation
final-LN parameters to unfreeze, even when it is true.

Generation MLS still averages layers `[2,5,8,11]` and adds the last selected
layer's spatial mean. LoRA injection, patch-embedding trainability, NoDinoPre,
PgT, fusion weights, pre-blend RMSNorm, optimizer, learning rate, batch size and
sampling are unchanged. No RAE latent statistics or `0.485` latent scaling is
added. In this active config the generation encoder still consumes the noisy
RGB input without ImageNet preprocessing.

### iREPA encoder: independent, frozen, with pretrained final-LN affine

The iREPA regularization encoder now has its own implementation in
`dino_alignment_encoder.py`; it does not instantiate the generation/LoRA
wrapper. `dino_encoder.py` exposes separate `build_generation_encoder` and
`build_alignment_encoder` builders. The two paths use the same DINOv3 model
type and pretrained source, but load separate parameter instances. Reusing a
read-only weight-loading helper does not share parameters or impose the
generation path's norm replacement. No sibling iREPA checkout is required at
runtime.

The regularization encoder follows the local iREPA DINOv3 encoder behavior:

- Retain the official pretrained final LayerNorm, including gamma and beta;
  do not replace it with an affine-free LayerNorm.
- Do not inject LoRA, train the patch embedding, or apply MLS. Keep the whole
  encoder frozen and in evaluation mode; extract features under `no_grad`.
- Preprocess clean raw `[0,255]` images with `/255`, then
  `Resize((img_size, img_size), antialias=True)`, then ImageNet normalization
  with mean `(0.485, 0.456, 0.406)` and std `(0.229, 0.224, 0.225)`.
- Call the official `forward_features` and take `x_norm_patchtokens`: the
  normalized final-layer patch tokens, `[B,256,768]` for B/16 at 256px.

The denoiser converts its clean diffusion-domain `[-1,1]` tensor back to raw
pixels with `(clean_x.float() + 1) * 127.5` before this preprocessing. No clamp
or rounding is added. This inverse conversion can introduce tiny float32
roundtrip differences from starting directly with original uint8 pixels.

Source references: [iREPA DINOv3 encoder](../iREPA/jit/vision_encoder.py),
[iREPA preprocessing and model loader](../iREPA/jit/models/dinov3_loader.py),
and [iREPA no-gradient feature extraction](../iREPA/jit/engine_jit.py).
The `0.485` in ImageNet's first-channel mean is unrelated to LatentForcing's
global latent scaling by `0.485`.

This corrects the first published 1017 implementation, which inadvertently
removed final-LN affine from both encoders. Relative to that implementation,
the regularization features change back to the pretrained-affine convention;
the generation encoder remains affine-free. Relative to the original 1016
baseline, the intended final-LN affine removal is on the generation side only.
The existing alignment-loss implementation is unchanged: matching the iREPA
encoder does not claim that the entire training pipeline, spatial feature
processing, or loss is numerically identical to standalone iREPA.

Runtime modules are named by function: `dino_generation`, `alignment_encoder`,
and `alignment_projector`. `alignment_encoder` is held in a plain list,
outside the generative model's registered state dict and EMA; it remains
frozen and is not optimized. The YAML group is renamed to `dino_generation`,
with all leaf option names and values retained. The diagnostic log column
`dino_student_token_norm` is renamed to `dino_generation_token_norm`.

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

The generation encoder has no `model.norm.weight` or `model.norm.bias`
(1536 frozen affine parameters removed for B/16). The iREPA encoder retains
those parameters, but was and remains outside the generative model's state
dict and EMA. Its affine restoration therefore changes regularization features,
not the generative EMA parameter list. The trainable parameter count is
unchanged relative to `nogamma_Residual_1016`.

This correction also renames registered checkpoint prefixes:

- `dino_student.*` -> `dino_generation.*`.
- `dino_student_projector.*` -> `alignment_projector.*`.

Neither the original 1016 checkpoints nor the first, dual-affine-free 1017
checkpoints are supported as drop-in resumes. The former also have a different
generation final-LN parameter layout; the latter use different state-dict
names and different regularization features. Do not continue an already-running
old experiment in place or silently reuse its output directory. Use a fresh
output directory and a separate resume directory with no old checkpoint, or
explicitly disable resume (`--resume ''`), and start from the same pretrained
DINOv3 weights. Resume only checkpoints produced by this corrected version.
No automatic checkpoint/optimizer/EMA migration is provided.

## Eight H200 GPUs (0-7)

Defaults: 80 epochs, per-GPU batch 128, global batch 1024, accumulation 1,
base LR 5e-5, effective learning rate 2e-4 after 5 warmup epochs,
and 50K online no-CFG heun50 evaluation
every 20 epochs. The existing DINOv3-B pretrained weights initialize the
encoder. ImageNet, `../fid_stats`, and `../nanogen-evals` are reused in place.
The YAML keeps `lr: null`, so `lr = blr * global_batch_size / 256`.
This fork and its `nogamma_Residual_1016` source have identical optimization
settings for comparison.

The H200 deployment root is
`/mnt/shared-storage-user/datafactory-share/gushishen/agent-z/Pixel_DIffusion`.
The H200 tmux launcher uses the shared CUDA 12.6 environment and checks all
8 H200 GPUs on `agent-gpu-2` (H200-0257) before starting:

```bash
# Run on agent-gpu-2 from this method directory.
bash scripts/train_noaffine_nogamma_B80_8gpu.sh --preflight
bash scripts/train_noaffine_nogamma_B80_8gpu.sh
```

The default mode creates tmux session
`train-noaffine-nogamma-B80-8gpu-b128-gbs1024`; preflight only checks resources
and resolved hyperparameters. Logs and checkpoints are under this method's
`checkpoints/BaseT-Residual(t,1-t)-NoDinoPre-PgT-blend_rmsnorm_nogamma-NoDDTHead-B-80/`.
This directory had no previous checkpoint when the corrected version was
prepared. The sibling 1016 method's checkpoints are not used for resume.

H200 resource defaults (all local, no GPU-side download):

- Shared interpreter:
  `/mnt/shared-storage-user/datafactory-share/gushishen/agent-z/envs/irepa-py310-cu126-20261015/bin/python`.
- ImageNet: `/mnt/shared-storage-user/datafactory-share/gushishen/imagenet`.
- Pretrained weights: repository-root `pretrain_models/`.
- Online statistics and Inception weights: repository-root `fid_stats/`.
- Offline evaluation weights/statistics: repository-root `nanogen-evals/`.
- DINOv3 source: this method's `SSL_code/dinov3/`.

Overrides are `PYTHON_BIN`, `IMAGENET_ROOT`, `PRETRAIN_MODELS_ROOT`,
`FID_STATS_ROOT`, and `NANOGEN_EVALS_ROOT`.


## Tests

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=.:jit \
  python -m pytest -q tests
```

Tests cover absence of norm parameters/buffers, last-dimension-only
normalization, shared/separate-instance equivalence, input gradients,
zero/small tokens, dtype preservation, unchanged affine defaults, branch
routing, config parsing, and sampling settings.
The DINOv3 checks distinguish the generation encoder's affine-free final LN
from the iREPA encoder's retained pretrained affine, and verify that block
norms and generation gradients are unaffected. The independent regularization
path must remain frozen, have no LoRA/MLS, use iREPA preprocessing, and remain
outside the registered generative state dict.

Upstream verification for this corrected version on 2026-10-17
(before H200 deployment):

- Full CPU suite: **113 passed, 4 skipped**. Skips require CUDA JiT RoPE or
  CUDA BF16 autocast. Tests cover independent parameter instances, retained
  alignment affine, generation-only no-affine, gradients, no-grad/eval target
  extraction, absence of alignment weights in the generative state dict, and
  no iREPA feature extraction during sampling.
- Real pretrained DINOv3-B CPU check against methods loaded from the local
  iREPA `DINOv3Encoder`, using the same pretrained backbone: preprocessing and
  last-layer patch features match exactly for the same raw input. The denoiser's
  diffusion-pixel roundtrip has maximum feature absolute difference about
  `1.86e-5` on the tested image; this is not a claim of bitwise equality after
  pixel-domain conversion, or a GPU/mixed-precision performance test.
- Real generation encoder: 48 LoRA modules, no final-LN affine. Real iREPA
  encoder: original final-LN affine present, all parameters frozen. Both output
  `[1,256,768]` in the B/16 smoke check.
- All 12 configs preserve every leaf option/value; only YAML group names and
  comments change. Launchers, PgT, pre-blend RMSNorm, spatial normalization and
  alignment-loss implementations remain unchanged.
- The 8-GPU launcher passed `--preflight` using
  `PRETRAIN_MODELS_ROOT=/data0/pretrain_models`, with the new project as its
  output root and no existing checkpoint detected on this machine.

The upstream verification above did not launch training. H200 deployment
validation and launch status are recorded separately below.


## H200 local verification (2026-10-17)

Synced upstream `f8dbfc368f3d8b74193ea19c9ae73e2c0c85fead` with shared-storage
path adaptations retained. The original local source was backed up first.

- CPU regression suite: **110 passed, 7 skipped**. Two cases require CUDA;
  five require a separate official PixelDiT2 reference checkout.
- Real pretrained DINOv3-B: generation has 48 LoRA modules and no final LN
  affine; alignment has no LoRA, all parameters frozen, and final LN gamma/beta
  exactly equal to the pretrained checkpoint. Their parameters are independent.
- Against local iREPA `DINOv3Encoder` methods using the same backbone and raw
  input, CPU preprocessing and patch features match exactly (max difference 0).
- Flattened configuration leaf values match the prior nogamma B80 config.
  Resolved preflight: 8 GPUs x 128, global batch 1024, accumulation 1, target
  LR 2e-4 after five warmup epochs, 80 epochs, EMA 0.9999/0.9996.
- Training was launched on agent-gpu-2 at **2026-10-17 15:52:52 UTC+8** in the
  tmux session above. The output directory had no old checkpoint; this is a
  fresh run from pretrained DINOv3 weights. Consult `train.log` and, on exit,
  `exit_code.txt` for current state.

Verification records and the source backup are under
`/mnt/shared-storage-user/datafactory-share/gushishen/agent-z/work/sync-noaffine-generation-alignment-20261017-154750/`.
The run records its source hashes and encoder protocol in `source_sha256.txt`
and `run_info.txt` beside `train.log`.
