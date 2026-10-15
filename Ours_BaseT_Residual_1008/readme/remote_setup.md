# 远端服务器部署

本仓库只同步 `Ours_BaseT_Residual_1008` 版本的源码、配置、脚本与测试。
ImageNet、DINOv3 预训练权重、在线/离线评测资源已经单独传输，不会通过 Git 下载。
本机仍在运行的训练脚本、YAML、模型及损失实现没有为迁移而更改。

## 1. 获取代码与环境

远端需要 Linux、Bash 4+、Git、NVIDIA GPU 和兼容 CUDA 12.6 wheel 的驱动。
预装系统 CUDA toolkit 不等于 PyTorch wheel 的 CUDA 版本；安装后需检查
`torch.version.cuda` 和 `torch.cuda.is_available()`。

```bash
git clone git@github.com:zrbwsw/Pixel_DIffusion.git
cd Pixel_DIffusion
conda create -n irepa python=3.10 pip -y
conda activate irepa
bash Ours_BaseT_Residual_1008/scripts/install_requirements.sh
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
```

私有仓库需要给远端自己的 SSH 公钥授权；仅用于下载可使用只读 deploy key，
需要提交修改则使用有仓库写权限的 GitHub 身份。不要复制本机私钥或分享 Token。
没有 Conda 时，可使用 Python 3.10 的 venv，再执行相同安装脚本。
`require.txt` 在仓库根目录和方法目录都可使用；根目录文件引用方法目录的依赖。
安装脚本先安装 PyTorch 2.7.0 / torchvision 0.22.0 的 cu126 wheels，再安装其余依赖。
不需要 `pip install -e SSL_code/dinov3`；该源代码由模型本地加载。

在线评测依赖固定的 LTH14/torch-fidelity 提交
`bfe72b995473e4fee9351705d7b005f17442a4ca`，不能换成普通 PyPI 发行版。
requirements 覆盖当前 Residual 训练、评测、诊断和测试，不保证其他项目的环境。
若 `import cv2` 提示缺少 `libGL.so.1`，需安装服务器系统的 OpenGL 运行库；
不要通过升级整个依赖环境来修复此类系统库缺失。

## 2. 设置已经上传的资源路径

以下占位路径需替换为远端实际位置。配置只影响当前 shell；确认后可保存到
仓库外的个人启动脚本，不要提交机器路径或凭据。

```bash
export IMAGENET_ROOT=/your/data/imagenet
export PRETRAIN_MODELS_ROOT=/your/models
export FID_STATS_ROOT=/your/resources/fid_stats
export NANOGEN_EVALS_ROOT=/your/resources/nanogen-evals
cd Ours_BaseT_Residual_1008
```

- 原始 ImageNet 训练路径应为 `$IMAGENET_ROOT/train/<class>/*.JPEG`。
  原始 HF Arrow 布局 `$IMAGENET_ROOT/imagenet-latents-images` 也可读取。
- B 模型使用 `dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth`；
  L 模型使用 `dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth`。
  同一份对应规模的权重初始化 student 和 frozen teacher。
- 256px 在线评测需要 `jit_in256_stats.npz` 和
  `weights-inception-2015-12-05-6726825d.pth`，两者位于 `$FID_STATS_ROOT`。
- 离线 FID/IS/FDR6 需要 `$NANOGEN_EVALS_ROOT` 下的 `huggingface/hub/`、
  `torch/hub/checkpoints/`、`stats/`；保留 HF `refs`、`snapshots`、`blobs`
  和相对软链接。当前指标不需要 `features/`。

未显式设置时，三个资源根路径默认是 Git 仓库根目录下的 `pretrain_models/`、
`fid_stats/`、`nanogen-evals/`。这些目录不会进入 Git。
`PYTHON_BIN` 可覆盖 Python 解释器，默认使用当前激活环境的 `python`。

## 3. 训练预检与启动

先选择配置和实际可用的 GPU。下面以八卡 L 模型为例：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
config='configs/BaseT-Residual(t,1-t)-NoDinoPre-PgT-blend_rmsnorm-NoDDTHead-L-80.yaml'
bash scripts/train_portable.sh "$config" --batch_size 32 --global_batch_size 256 --preflight
bash scripts/train_portable.sh "$config" --batch_size 32 --global_batch_size 256
```

`--preflight` 只导入模块、解析配置、检查文件和 batch 关系，不加载模型、
不读取完整 checkpoint、不启动训练、不验证 GPU 显存是否足够。
实际训练可以放入自己新建的 tmux 会话；启动脚本不会停止或占用其他实验的进程。

B 模型可将配置换成同名 `-B-80.yaml`。保持全局 batch 256 时，
八卡每卡 32 为累积 1；四卡每卡 32 为累积 2；四卡每卡 64 为累积 1。
这些是 batch 换算，不代表远端 GPU 一定能容纳对应模型。先根据显存选择每卡 batch。
实际累积次数由训练代码根据全局 batch 自动推导，预检输出会显示它。

脚本遵循 YAML 的 `output_dir` / `resume`，并将 stdout/stderr 追加到
`output_dir/train.log`。存在 `resume/checkpoint-last.pth` 就恢复 model、optimizer
和两套 EMA；不存在则从头训练，预检显示 `training_mode`。
本次不传 checkpoint，因此远端默认是从头训练，不是继续本机实验。
从头跑一个新实验时，显式使用新输出目录并清空 resume：

```bash
bash scripts/train_portable.sh "$config" --batch_size 32 --global_batch_size 256 \
    --output_dir checkpoints/remote_trial --resume ''
```

配置文件路径相对执行命令时的目录解析；CLI 中的 `output_dir` / `resume`
等相对路径在方法目录内解析。路径含括号、逗号或空格时应加引号。
历史的 `train_*_0123.sh` / `train_*_4567.sh` 等仍保留原服务器路径，
远端使用 `train_portable.sh`，不要直接运行历史排队脚本。

## 4. CFG 采样与离线评测

必须使用与 checkpoint 匹配的 YAML；不同模型结构的权重不可混用。

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
experiment='BaseT-Residual(t,1-t)-NoDinoPre-PgT-blend_rmsnorm-NoDDTHead-L-80'
config="configs/$experiment.yaml"
checkpoint="checkpoints/$experiment/checkpoint-20.pth"
destination="checkpoints/eval_checkpoint/$experiment/epoch20_cfg2p3_flow100"
bash scripts/sample_portable.sh "$config" "$checkpoint" "$destination" --preflight
bash scripts/sample_portable.sh "$config" "$checkpoint" "$destination"
```

默认参数为 EMA1、50K 图像、flow_dpm_solver 100 steps、CFG 2.3、interval [0,1]、
每卡采样 batch 16。CFG 前向会扩展 conditional/unconditional batch，需保留显存余量。
可通过 `--cfg=2.5`、`--gen-bsz=8` 等覆盖；同一输出目录禁止变更采样配置。
已有 `samples.npz` 时跳过生成，但仍进行离线评测。
采样 `--preflight` 会写入 `sampling_config.yaml` 和 `evaluation_assets.json`，
但不会生成图像或加载模型。其他方法或设置须使用新的输出目录。
生成阶段使用所选全部 GPU，离线评测使用其中第一张。

单独检查离线资源不需要 checkpoint，也不会联网下载权重：

```bash
PYTHONPATH=.:jit python -m evaluation.evaluate_fdr6 \
    --check-assets --cache-root="$NANOGEN_EVALS_ROOT"
```

在线 JiT FID/IS 与离线 nanogen FID/IS/FDR6 协议不同，汇总结果时继续区分。

## 5. 验证与协作

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 PYTHONPATH=.:jit \
    python -m pytest -q tests
git status --short
```

没有相邻 `PixelDiT2` 官方源码目录时，projector 官方实现对照测试会跳过；
不影响本仓库内独立 projector 的运行。单元测试和资源检查不能替代实际 GPU
训练/采样验收，远端正式跑 80 epoch 前应先做独立输出目录的小规模试运行。

协作使用功能分支提交代码和配置，不提交实验输出或资源。
更新代码前先检查本地改动；运行中的实验不要直接切换模型实现或配置。
不得使用 `git add -f` 把被忽略的 checkpoint、资源或凭据加回仓库。
