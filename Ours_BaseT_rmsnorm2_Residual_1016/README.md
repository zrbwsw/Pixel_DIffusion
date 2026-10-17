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

## 2026-10-17: three controlled four-GPU ablations

These experiments use the trained rmsnorm2 B-80 recipe as the baseline and
start from scratch in three new output directories. Each changes exactly one
model control. The frozen DINOv3 alignment teacher, preprocessing, alignment
loss, LoRA/MLS/PgT, and independent learned pre-blend RMSNorm gains retain
the baseline settings.

| Ablation | Control | DINO / JiT weights | Host and physical GPUs | Launcher under `scripts/` |
|---|---|---|---|---|
| Generation final LN no-affine | `dino_generation_final_norm_affine: false` | `t` / `1-t` | agent-gpu, 4–7 | `train_rmsnorm2_noaffine_B80_4gpu.sh` |
| Reversed blend | `dino_time_blend_reverse: true` | `1-t` / `t` | agent-gpu-2, 0–3 | `train_rmsnorm2_reverse_B80_4gpu.sh` |
| DropRes | `dino_grounding_drop_on_label_drop: true` | `t` / `1-t`; null label: `0` / `1` | agent-gpu-2, 4–7 | `train_rmsnorm2_dropres_B80_4gpu.sh` |

The no-affine option removes gamma/beta only from the generation DINO's
final output LayerNorm, retaining its normalization and epsilon. MLS calls
this same output LN on its selected layers; DINO block LNs are unchanged.
The frozen teacher retains its pretrained final LN gamma/beta and the
existing rmsnorm2 teacher implementation. Both pre-blend RMSNorm gamma
vectors remain trainable in every ablation.

Reversing the blend changes only the two mixing coefficients: projector,
JiT, flow target, and sampler still receive the original timestep.
DropRes shares the actual label-drop mask (probability 0.1); for a dropped
label, the normalized JiT branch has weight one and the DINO branch zero.
CFG's null-label branch uses the same rule. DINO stays in the computation
graph, including all-null batches, for consistent DDP parameter usage.

Common training settings: **4 GPUs × batch 256 = global batch 1024**,
accumulation 1, base LR 5e-5 (effective LR **2e-4**), 80 epochs, five warmup
epochs, constant LR, AdamW (0.9, 0.95), weight decay 0, seed 0,
EMA 0.9999/0.9996, horizontal flip 0.5. Online evaluation remains
Heun50 / CFG1 / 50K images / generation batch 64 every 20 epochs.

The YAML filenames use the original baseline name with, respectively,
`-NoAffine-`, `Residual(1-t,t)`, and `-DropRes-`. Their `exp_name`,
`output_dir`, `resume`, and metric filenames are independent of the baseline.

Run the appropriate launcher on its assigned host:

```bash
bash scripts/train_rmsnorm2_noaffine_B80_4gpu.sh --preflight
bash scripts/train_rmsnorm2_noaffine_B80_4gpu.sh
```

Each launcher creates a detached tmux session, checks the selected GPUs,
uses a dedicated output lock and compilation cache, and records resolved
resources, model controls, source hashes, and the command. Default launches
start from scratch and refuse a populated output directory. `--resume`
explicitly resumes that ablation's own `checkpoint-last.pth` after checking
its model and training settings; it never uses the baseline checkpoint.
The two GPU2 jobs use separate torchrun rendezvous ports via `--standalone`.
