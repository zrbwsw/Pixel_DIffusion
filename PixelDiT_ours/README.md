<p align="center">
  <img src="assets/pixeldit-logo.png" alt="PixelDiT" width="360">
</p>

<h1 align="center"><a href="https://pixeldit.github.io/">PixelDiT </a> &amp;  <a href="https://pixeldit.github.io/pixeldit2/">PixelDiT2 </a> </h1>
<p align="center"><strong>Pixel-space diffusion transformers for image generation</strong></p>
<p align="center">NVIDIA · University of Rochester</p>

## PixelDiT_ours：DINO+LoRA 双路 patchify

详细阅读顺序、项目结构和新增/修改文件清单见 [CODE_GUIDE_ZH.md](CODE_GUIDE_ZH.md)。

本目录是上游 PixelDiT 的本地研究版本，只修改 PixelDiT1 C2I 的 patch-level 输入；PixelDiT2 和 T2I 不变。数据流为：

```text
带噪图像 x_t
  ├─ DINOv3+LoRA（可选 y）→ MLS → PgT(t) → RMSNorm ────── × (1-f(t))
  └─ PixelDiT raw patchify → RMSNorm ──────────────────── × f(t)
                                                             ↓ 相加
                                            PixelDiT patch blocks → PiT pixel decoder
```

- 时间定义与上游一致：`t=0` 为噪声、`t=1` 为干净图像。默认 `(DINO,raw)=(1-f(t),f(t))`；`f(t)` 只控制融合，扩散过程和 PgT 使用原始 `t`。
- 基础配置 `c2i/configs/pix256_xl.yaml` 使用 DINOv3-B/16、LoRA rank/alpha=64、`smoother`，不启用 DinoYCond。DINO 输入不做额外归一化；冻结原 backbone 和 final LN，训练 LoRA 与 DINO patch embedding。
- `dino_time_blend_transform` 可选 `linear`、`smooth`、`smoother`、`sigmoid`；sigmoid 使用 `dino_time_blend_key`（k）和 `dino_time_blend_a`（a），并归一化到 `f(0)=0,f(1)=1`。
- `dino_patchify_mode: dino_only` 时只计算 DINO token，跳过原生 `unfold/s_embedder/RMSNorm`；默认 `blend` 保持双路加权。PiT 的 pixel 级输入和解码不受影响。
- `dino_time_blend_reverse: true` 将 `(DINO,raw)` 从默认 `(1-f(t),f(t))` 改为 `(f(t),1-f(t))`，仅在 `blend` 模式生效。
- DINO 超参数包括：`dino_label_cond`、`dino_aggregation`（`mls/mean/none`）、`dino_layers`（`default/last/last-N/显式索引`）、`dino_lora_target_modules`、`dino_train_patch_embed`、`dino_train_final_norm`、`dino_generation_final_norm_affine`。启用 y cond 时，独立 `ClassEmbedder` 在 DINO patch embedding 后给每个 patch token 加 `y_embedder(y)`；null 类仍为 1000。
- `dino_weights: null` 时，从 `PRETRAIN_MODELS_ROOT` 或主仓库 `pretrain_models/` 读取对应 DINOv3 权重；也可在 YAML 中填写显式路径。DINOv2/v3 官方源码与许可证分别位于 `third_party/dinov2/`、`third_party/dinov3/`。
- 四个 PixelDiT1 预训练 checkpoint 已整理到本目录 `pretrain_models/`；发布文件名会优先解析到该目录。新增分支使上游 checkpoint 不能直接严格加载；本方法训练或恢复需使用完整的新 checkpoint。

### XL DINOv3-B YCond 归一化 sigmoid 配方

当前 XL 配置为 `a=0.1`：`c2i/configs/pix256_xl_dinov3b_ycond_sigmoid_k20_a0p1_norm.yaml` + `scripts/train_pixeldit_xl_dinov3b_ycond_sigmoid_k20_a0p1_norm_8gpu.sh`。

该配置采用 MLS/default 选层、仅 y cond、LoRA rank/alpha=64、目标层 `qkv,proj,fc1,fc2`、训练 DINO patch embedding、final LN 冻结且保留 affine、PgT、两路 RMSNorm，以及归一化 sigmoid `k=20`。PixelDiT-XL、DINOv2 REPA、优化器、EMA 和采样保持默认值：八卡每卡32、全局 batch256、有效 LR `1e-4`。

### B DINOv3-B YCond 80 epoch 消融

基于 `pix256_b_80ep.yaml` 的 B 主干和训练周期，提供两组配置与对应 `_4gpu.sh` 脚本：

- `pix256_b_dinov3b_ycond_sigmoid_k20_a0.1_norm.yaml`：归一化 sigmoid，`k=20,a=0.1`。
- `pix256_b_dinov3b_ycond_linear.yaml`：线性融合 `f(t)=t`。

两组均为四卡每卡64、全局 batch256、80 epoch，每20 epoch 进行在线采样并保存周期 checkpoint；输出目录仍为 `train_logs/exp_<实验名>/`，在线样本写入 `val_256_b/`。

### 按论文协议分阶段训练

A0.1 DINOv3-B YCond 配方提供以下阶段配置和同名训练脚本：

- XL `_80`：到400K steps（约80 epoch），`lr=1e-4、clip=1.0`。
- XL `_160`：从80续训到800K steps（约160 epoch），保持 `lr=1e-4、clip=1.0`。
- XL `_320`：从160续训到1.6M steps（约320 epoch），通过 `override_lr_on_resume=1e-5` 覆盖恢复的学习率，并设置 `clip=0.5`。
- B `_160`：对应论文 B 的800K steps，全程 `lr=1e-4、clip=1.0`，只需这一份配置，无阶段切换。

XL 应依次执行 `_80 → _160 → _320`。三个阶段的 `tags.exp` 相同，因此默认写入同一个 `train_logs/exp_<实验名>/`，依靠 `auto_resume` 读取最新 checkpoint；`_320` 不应在没有前序 checkpoint 的空目录中从头启动。B 的 `_160` 同样可从已有80 epoch目录继续训练。

本机没有上游 HDF5，因此这些配方使用 PixelDiT 自带的 `ImageFolderDataset` 读取原始 ImageNet，采用与 REPA-E 相同的 ADM 256 中心裁剪，不做水平翻转；已用真实样本逐像素验证与官方 HDF5 输入一致。DINOv2 teacher 从 `pretrain_models/torch_hub/` 离线加载。

```bash
# 先检查命令和资源，不启动训练。
bash scripts/train_pixeldit_xl_dinov3b_ycond_sigmoid_k20_a0p1_norm_8gpu.sh --dry-run
bash scripts/train_pixeldit_xl_dinov3b_ycond_sigmoid_k20_a0p1_norm_8gpu.sh --preflight

# 按项目约定，在开发机创建同名 tmux 后前台执行；默认连接 GPU2 八卡。
tmux new -s 'pixeldit_ours_xl_dinov3b_ycond_sigmoid_norm_k20_a0p1_bs256'
bash scripts/train_pixeldit_xl_dinov3b_ycond_sigmoid_k20_a0p1_norm_8gpu.sh
```

所有脚本固定 `WANDB_MODE=offline`，训练指标由 `WandbLogger` 写入对应 `RUN_DIR/wandb/` 下的离线 run，不做在线上传。`train.log`、checkpoint 和采样 `output.npz` 也保存在本地。

脚本默认 Python 为 `/mnt/shared-storage-user/datafactory-share/gushishen/agent-z/envs/pixeldit-ours-py310-cu126-20261029/bin/python`。可用 `GPU_HOST`、`CUDA_VISIBLE_DEVICES`、`OUTPUT_ROOT`、`PYTHON_BIN` 覆盖默认运行路径。

以下为上游原始说明。

![PixelDiT text-to-image samples](assets/pixeldit-t2i.jpg)


## Papers and Resources

| Work | Publication | Resources |
|---|---|---|
| **PixelDiT2: Representation-Grounded Pixel Diffusion Transformers** | **NeurIPS 2026** | <a href="https://pixeldit.github.io/pixeldit2/"><img src="https://img.shields.io/badge/%F0%9F%8C%90_-Project-2ea44f" alt="Project"></a> <a href="https://arxiv.org/abs/2609.24919"><img src="https://img.shields.io/badge/%F0%9F%93%84_-arXiv-b31b1b.svg" alt="arXiv"></a> <a href="https://huggingface.co/nvidia/PixelDiT2-ImageNet"><img src="https://img.shields.io/badge/%F0%9F%A4%97_Model-ImageNet-yellow" alt="ImageNet models"></a> |
| **PixelDiT: Pixel Diffusion Transformers for Image Generation** | **CVPR 2026 Oral**<br><a href="https://paperswithcode.co/conferences/cvpr-2026/best-paper-finalists"><img src="https://img.shields.io/badge/%F0%9F%8F%86_CVPR_2026-Best_Paper_Finalist-gold" alt="CVPR 2026 Best Paper Finalist"></a> | <a href="https://pixeldit.github.io/"><img src="https://img.shields.io/badge/%F0%9F%8C%90_-Project-2ea44f" alt="Project"></a> <a href="https://arxiv.org/abs/2511.20645"><img src="https://img.shields.io/badge/%F0%9F%93%84_-arXiv-b31b1b.svg" alt="arXiv"></a> <a href="https://huggingface.co/nvidia/PixelDiT-ImageNet"><img src="https://img.shields.io/badge/%F0%9F%A4%97_Model-ImageNet-yellow" alt="ImageNet models"></a> <a href="https://huggingface.co/nvidia/PixelDiT-1300M-1024px"><img src="https://img.shields.io/badge/%F0%9F%A4%97_Model-T2I-yellow" alt="T2I model"></a><br> <a href="https://huggingface.co/Comfy-Org/PixelDiT"><img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fmodels%2FComfy-Org%2FPixelDiT%3Fexpand%3DdownloadsAllTime&query=%24.downloadsAllTime&label=%F0%9F%A4%97%20Comfy-Org%2FPixelDiT%20downloads&color=yellow" alt="Comfy-Org/PixelDiT downloads"></a>  |

## Authors

**PixelDiT2**

[Yongsheng Yu](https://www.yongshengyu.com/) ·
[Wei Xiong](https://wxiong.me/) ·
[Yichen Sheng](https://shengcn.github.io/) ·
[Shiqiu Liu](http://behindthepixels.io/) ·
[Jiebo Luo](https://www.cs.rochester.edu/u/jluo/)

**PixelDiT**

[Yongsheng Yu](https://www.yongshengyu.com/) ·
[Wei Xiong](https://wxiong.me/) ·
[Weili Nie](https://weilinie.github.io/) ·
[Yichen Sheng](https://shengcn.github.io/) ·
[Shiqiu Liu](http://behindthepixels.io/) ·
[Jiebo Luo](https://www.cs.rochester.edu/u/jluo/)

**Affiliation**: NVIDIA · University of Rochester

**Project lead and main advisor**: [Wei Xiong](https://wxiong.me/)


## News

- **2026/09** — [PixelDiT2](https://pixeldit.github.io/pixeldit2/) is accepted to **NeurIPS 2026**, a representation-grounded pixel diffusion model that converges much faster.
- **2026/06** — [PixelDiT](https://pixeldit.github.io/) is a **CVPR 2026 Best Paper Finalist**.
- **2026/06** — Added [post-modulation for PiT blocks](c2i/README.md#training-stability-post-modulation-for-pit-blocks) to improve training stability.
- **2026/04** — PixelDiT training and inference code, and pretrained models, are released.
- **2026/02** — PixelDiT is accepted to **CVPR 2026 Oral**.
- **2025/11** — The [PixelDiT paper](https://arxiv.org/abs/2511.20645) is released.

<a id="pixeldit2"></a>
<a id="pixeldit"></a>
<a id="performance"></a>

## Introduction
**PixelDiT** is a series of work on pixel-space generative modeling, aiming to remove the need of VAEs which are widely used in traditional latent generative models and perform geneartion directly in pixel space. 
- **[PixelDiT](https://pixeldit.github.io/)**: Our first pixel-space diffusion models. We release both the models on ImageNet and a 1.3B Text-to-Image model that can generate 1024x1024 resolution images.
- **[PixelDiT2](https://pixeldit.github.io/pixeldit2/)**: We have a deeper investigation on how representation learning can guide the content generation for pixel diffusion. We propose to use representations to ground the generation process. It converges significantly faster and obtains better results on ImageNet. Please read the details in our [paper](https://arxiv.org/pdf/2609.24919). 

## Models and results

> Note: Our models are resumed every 4 hours, using the timestamp as the random seed each time. As a result, the final training outcome may have a slight gap compared to a continuous run without intermediate resumes.

### Class-to-image · ImageNet

| Model | Resolution | Epochs | FID ↓ | IS ↑ |
|---|:---:|---:|---:|---:|
| **PixelDiT2-H/16** | 256×256 | 600 | **1.46** | **301.6** |
| **PixelDiT2-H/16** | 512×512 | 680 | **1.48** | **295.7** |
| PixelDiT-XL | 256×256 | 320 | 1.61 | — |
| PixelDiT-XL | 512×512 | 850 | 1.81 | — |

Results use 50,000 samples. PixelDiT2 uses Heun with 50 steps; PixelDiT uses
FlowDPMSolver with 100 steps. The
[class-to-image guide](c2i/README.md) lists all checkpoints and the corresponding
sampling commands.

### Text-to-image · PixelDiT-T2I


| Resolution | GenEval ↑ | DPG-Bench ↑ |
|:---:|---:|---:|
| 512×512 | 0.78 | 83.7 |
| 1024×1024 | 0.74 | 83.5 |

See the [text-to-image guide](t2i/README.md) for training and inference.

**ComfyUI.** PixelDiT-T2I is also available in [ComfyUI](https://github.com/comfyanonymous/ComfyUI). The [Comfy-Org/PixelDiT](https://huggingface.co/Comfy-Org/PixelDiT) repository provides repackaged weights (bf16 and mxfp8) and a ready-to-use [text-to-image workflow](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_pixeldit_t2i.json). The repository also hosts [PiD](https://github.com/nv-tlabs/PiD).


<a id="getting-started"></a>

## Quick start

Install the requirements for the model family you want to use, in separate
Python environments:

| Model family | Install from the repository root | Setup guide |
|---|---|---|
| PixelDiT2 | `python -m pip install -r requirements-pixeldit2.txt` | [PixelDiT2 environment](c2i/README.md#pixeldit2-environment) |
| PixelDiT (C2I / T2I) | `python -m pip install -r requirements-pixeldit1.txt` | [PixelDiT environment](c2i/README.md#pixeldit-environment) |

PixelDiT2 uses Python 3.10 and pinned PyTorch/CUDA dependencies. The PixelDiT
file retains the original dependency specification; `requirements.txt` remains
its compatibility entrypoint. C2I evaluation also differs: PixelDiT2 uses the
LTH14 torch-fidelity fork with JiT statistics, while PixelDiT uses ADM in a
separate evaluation environment. See the guides for details.

For class-to-image generation, both model families use `c2i/main.py`.
Choose a matching checkpoint and configuration in the
[model list](c2i/README.md#pretrained-models), then follow the shared
[inference](c2i/README.md#inference-and-evaluation) or
[training](c2i/README.md#training) workflow.

| Model family | Class-to-image configurations | Guide |
|---|---|---|
| PixelDiT2 | `pixeldit2_h16_in256.yaml`, `pixeldit2_h16_in512.yaml` | [Inference](c2i/README.md#pixeldit2-inference) |
| PixelDiT | `pix256_xl.yaml`, `pix512_xl.yaml` | [Inference](c2i/README.md#pixeldit-inference) |

### Use the model definitions in another project

The model code in `pixdit_core` can be installed with pip:

```bash
python -m pip install "git+https://github.com/NVlabs/PixelDiT"
# PixelDiT2's grounding encoder also needs timm:
python -m pip install "pixeldit[pixeldit2] @ git+https://github.com/NVlabs/PixelDiT"
```

Append `@<commit>` to the URL to pin a version. The package is named `pixeldit`
and is imported as `pixdit_core`:

```python
from pixdit_core.pixeldit_t2i import PixDiT_T2I    # PixelDiT text-to-image
from pixdit_core.pixeldit_c2i import PixDiT        # PixelDiT class-to-image
from pixdit_core.pixeldit2_c2i import PixelDiT2    # PixelDiT2 class-to-image
```

## Repository Structure

```
├── pixdit_core/      # Shared model definitions
│   ├── pixeldit2_c2i.py   # PixelDiT2  (single-path patch DiT + grounding)
│   ├── grounding.py       # PixelDiT2  frozen DINOv3 encoder and projection P_g
│   ├── pixeldit_c2i.py    # PixelDiT   (dual-level: patch DiT + pixel PiT)
│   └── pixeldit_t2i.py    # PixelDiT   text-to-image
├── tools/            # Checkpoint download, evaluation and GFLOPs computation
├── c2i/              # Class-to-image (PixelDiT and PixelDiT2)
└── t2i/              # Text-to-image
```

<details>
<summary>Compute model GFLOPs</summary>

### Compute GFLOPs

Measure single-forward-pass GFLOPs for any model in this repository (**run from
project root**). For PixelDiT2 the count includes the frozen grounding encoder.

```bash
# PixelDiT2 (ImageNet 256x256 and 512x512)
python tools/compute_flops.py --config c2i/configs/pixeldit2_h16_in256.yaml
python tools/compute_flops.py --config c2i/configs/pixeldit2_h16_in512.yaml --height 512 --width 512
```

```bash
# PixelDiT C2I (ImageNet 256x256, default resolution)
python tools/compute_flops.py --config c2i/configs/pix256_xl.yaml
```

```bash
# PixelDiT T2I at 1024x1024
python tools/compute_flops.py --config t2i/configs/PixelDiT_1024px_pixel_diffusion_stage3.yaml --height 1024 --width 1024
```

</details>

## Acknowledgements

We would like to thank the authors of [PixNerd](https://github.com/MCG-NJU/PixNerd) and [SANA](https://github.com/NVlabs/SANA) for sharing their code. We also thank the [SANA team](https://arxiv.org/pdf/2410.10629) for sharing their text-to-image training data.

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{yu2026pixeldit2,
      title={PixelDiT2: Representation-Grounded Pixel Diffusion Transformers},
      author={Yongsheng Yu and Wei Xiong and Yichen Sheng and Shiqiu Liu and Jiebo Luo},
      booktitle={Conference on Neural Information Processing Systems (NeurIPS)},
      year={2026},
}

@inproceedings{yu2026pixeldit,
      title={PixelDiT: Pixel Diffusion Transformers for Image Generation},
      author={Yongsheng Yu and Wei Xiong and Weili Nie and Yichen Sheng and Shiqiu Liu and Jiebo Luo},
      booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
      year={2026},
}
```
