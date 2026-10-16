# Deploy the RMSNorm ablations

The repository publishes source code only. ImageNet, DINOv3 weights,
checkpoints, and evaluation resources must be provisioned separately.
Both versions include their own JiT and DINOv3 source; neither imports
the old method directory at runtime. GPU runs require Linux, Bash 4+,
Python 3.10, an NVIDIA driver compatible with the chosen PyTorch CUDA
wheels, and enough free GPU memory.

## 1. Clone and choose a version

```bash
git clone git@github.com:zrbwsw/Pixel_DIffusion.git
cd Pixel_DIffusion

# Choose ONE of these pairs.
version=Ours_BaseT_rmsnorm2_Residual_1016
experiment='BaseT-Residual(t,1-t)-NoDinoPre-PgT-blend_rmsnorm2-NoDDTHead-B-80'

# Alternative:
# version=Ours_BaseT_nogamma_Residual_1016
# experiment='BaseT-Residual(t,1-t)-NoDinoPre-PgT-blend_rmsnorm_nogamma-NoDDTHead-B-80'

cd "$version"
config="configs/$experiment.yaml"
```

For an existing clone, inspect `git status` before pulling. Do not switch
implementation files underneath a running experiment. Private repositories
require the remote server's own authorized SSH key; never copy private keys
or commit access tokens.

The first version learns separate pre-blend channel gains. The second
removes the pre-blend gain entirely. JiT/PgT internal normalization is
unchanged. The active configs otherwise retain the same B-scale settings.

## 2. Install the shared environment

```bash
conda create -n irepa python=3.10 pip -y
conda activate irepa
bash scripts/install_requirements.sh
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
```

One environment serves both versions. The install script pins PyTorch
2.7.0 / torchvision 0.22.0 cu126 wheels and installs `require.txt`, including
the fixed LTH14/torch-fidelity commit required by online FID/IS. System CUDA
toolkit installation alone does not establish driver compatibility.
Do not replace this torch-fidelity fork with the ordinary PyPI release.
The local DINOv3 source does not need a separate editable installation.
If OpenCV reports missing `libGL.so.1`, install the corresponding system
runtime library rather than changing all Python dependency versions.

## 3. Point to the resources already transferred

Replace these example paths with paths on the remote server:

```bash
export IMAGENET_ROOT=/your/data/imagenet
export PRETRAIN_MODELS_ROOT=/your/models
export FID_STATS_ROOT=/your/resources/fid_stats
export NANOGEN_EVALS_ROOT=/your/resources/nanogen-evals
export PYTHON_BIN="$(command -v python)"
export DINOV3_REPO="$PWD/SSL_code/dinov3"
```

Required assets:

- ImageNet: `$IMAGENET_ROOT/train/<class>/*.JPEG`, or the supported HF Arrow
  layout under `$IMAGENET_ROOT/imagenet-latents-images`.
- B encoder: `$PRETRAIN_MODELS_ROOT/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth`.
- Online evaluation: `jit_in256_stats.npz` and
  `weights-inception-2015-12-05-6726825d.pth` in `$FID_STATS_ROOT`.
- Offline FID/IS/FDR6: the existing `huggingface/hub/`,
  `torch/hub/checkpoints/`, and `stats/` under `$NANOGEN_EVALS_ROOT`.
  Preserve Hugging Face `refs`, `snapshots`, `blobs`, and relative symlinks.

No dataset, model weights, or evaluation weights are downloaded by the
training preflight. External weights and stats do not belong in Git.

## 4. Preflight and train

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
bash scripts/train_portable.sh "$config" --preflight
bash scripts/train_portable.sh "$config"
```

Defaults are 80 epochs, batch 128 per GPU, global batch 1024, no accumulation
on eight H200 GPUs, base LR 5e-5, effective learning rate 2e-4 after
5 warmup epochs, and online 50K no-CFG
heun50 evaluation every 20 epochs. Select GPUs that actually exist and are
free on that server. If memory is insufficient, use `--batch_size 64`;
global batch 1024 then requires accumulation 2 on eight GPUs, with the same LR.
The YAML uses `lr: null`: `lr = blr * global_batch_size / 256`.
Each experiment uses all eight GPUs on its own server (or runs sequentially).

The preflight checks paths/configuration/batch arithmetic. It does not load
the model, allocate training GPU memory, or prove that the batch fits.
The version-local output is `checkpoints/$experiment/`, with `train.log`
and structured diagnostics. Relaunching resumes that experiment's own
`checkpoint-last.pth` if it exists; otherwise training starts from scratch.
Do not resume a different normalization variant's optimizer/EMA state.
If the selected output already contains an old batch-256 checkpoint, use a
new output directory and `--resume ''` for this batch-1024 experiment.

For an isolated smoke run, keep the main experiment output untouched:

```bash
bash scripts/train_portable.sh "$config" --epochs 1 --num_images 100 --gen_bsz 8 \
  --output_dir checkpoints/remote_smoke --resume ''
```

This smoke run evaluates only 100 generated images at the end; its metrics
are not comparable to the 50K experiment results.

The named `*_01234567.sh` launchers set all eight GPUs and the same batch/LR
settings explicitly. They use the activated Python environment and respect
the resource-path variables above. Launchers never terminate existing jobs
or automatically queue behind them.

## 5. CFG sampling and offline evaluation

Run from the selected version directory with the matching checkpoint:

```bash
checkpoint="checkpoints/$experiment/checkpoint-20.pth"
destination="checkpoints/eval_checkpoint/$experiment/epoch20_cfg2p3_flow100"
bash scripts/sample_portable.sh "$config" "$checkpoint" "$destination" --preflight
bash scripts/sample_portable.sh "$config" "$checkpoint" "$destination"
```

Defaults: EMA1, 50K images, flow100, CFG 2.3, interval [0,1], generation
batch 16 per GPU. Use `--gen-bsz=8` if necessary. Each output directory is
tied to one sampling configuration; do not mix checkpoints or scales.
Sampling preflight writes sampling YAML and asset reports, but does not
generate images. Offline metrics use the first selected GPU after sampling.
Online and offline evaluation protocols differ; keep their results separate.

## 6. Verify and collaborate

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 PYTHONPATH=.:jit \
  python -m pytest -q tests
git status --short
```

Tests that need CUDA or the external official PixelDiT2 checkout may skip
on a source-only server. CPU tests do not replace a GPU smoke run.
Commit only the selected version's source/configs/docs. Checkpoints,
samples, metrics, datasets, credentials, and evaluation assets are ignored;
do not force-add them.
