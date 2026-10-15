# Ours_BaseT_Residual_1008

跨服务器安装、训练和采样请先阅读 [远端部署说明](readme/remote_setup.md)。
可迁移入口为 `scripts/train_portable.sh` 和 `scripts/sample_portable.sh`；
下文带 GPU 编号的历史脚本保留本机路径，仅用于原服务器实验复现。

本目录是从 `Ours-J18-new-drop0.1-l4-BaseT-B-80` 分离出的 gated
representation-residual 实验代码，不修改 `Ours_0523` 的既有实现。

## 核心改动

历史 BaseT 路径使用 DINOv3-LoRA MLS token 替换 JiT 原生 PatchEmbed token。本实验改为：

```text
z_t -> JiT BottleneckPatchEmbed -------------------------------+
                                                               +-> JiT-B -> BaseT DDTHead
z_t -> DINOv3-B-LoRA -> MLS -> residual adapter -> multiply t -+
```

对于 ImageNet-256 B-scale：

```text
h_native = JiT_PatchEmbed(z_t)                         # [B, 256, 768]
f_dino   = DINO_MLS(preprocess(z_t))                   # [B, 256, 768]
r         = Linear(SiLU(Linear(LayerNorm(f_dino))))    # [B, 256, 768]
h_0       = h_native + t[:, None, None] * r + pos      # [B, 256, 768]
```

当前 `t` 是 clean-image weight：`t=0` 为纯噪声，`t=1` 为干净图像。
residual adapter 的末层采用零初始化，因此初始前向严格退化为 native JiT
PatchEmbed 路径。alignment 的 `3x3 ConvProj` 保持独立，只处理 JiT block 4
输出并与 frozen DINO teacher 对齐，不参与 residual 融合。

### DINO 输入预处理开关

`--dino_preprocess_input` 默认开启，与已有 checkpoint 的训练行为一致：
带噪图像 `z_t` 在进入 DINO-LoRA student 前，先计算 `(z_t+1)/2`，
再使用 ImageNet mean/std 标准化。干净图像在 `[-1,1]`，加噪后的值可能超出
该范围；这里没有 clamp，也不保证映射后在 `[0,1]`。通过
`--no-dino_preprocess_input` 可让带噪 student 直接接收原始 `z_t`；
训练前向与 CFG/非 CFG 采样共用此设置。Frozen teacher 始终接收经过
同一预处理的干净图像，保持预训练特征目标不变。

原有四份 YAML 都显式设置 `dino_preprocess_input: true`。新实验使用独立
YAML 设置为 `false`；离线采样也要使用对应 YAML，避免加载 checkpoint 时
使用错误的输入约定。
旧 checkpoint 的参数元数据没有此字段，但用当前 YAML 加载时仍保持默认开启。

## BaseT-Residual(t,1-t)-NoDinoPre-PgT 实验

与 iREPA、RAEv2、PixelDiT2 的逐组件对照、已有差异和评测路径见
[实现审计](readme/implementation_comparison_irepa_raev2_pixeldit2.md)。

新配置 `configs/BaseT-Residual(t,1-t)-NoDinoPre-PgT-B-80.yaml`
相对原 Residual 配置只改变 JiT 输入路径：

```text
native = JiT_PatchEmbed(z_t)
dino   = DINOv3-LoRA-MLS(z_t)  # 带噪 student 不做 ImageNet 预处理
g_t    = Pg(dino, t, grid_hw)  # Linear -> official time-conditioned DiT block -> Linear
h_0    = (1-t) * native + t * g_t + pos
```

`Pg` 在 DINO 输出之后、JiT 输入之前；原有的 alignment ConvProj 和
`denoise + 0.5 * alignment` 损失不变。`pixeldit2_projector.py` 独立保存
PixelDiT2 官方 `TimestepProjection` 及其必要组件，来源为相邻 `PixelDiT2`
仓库的 `8f83fa88a7b9dab575e5a45dafe3bb3c308685e0` 版本。
B 模型使用 1 个 block、768 维、12 个 attention heads，独立时间嵌入
（`max_period=10`）、RMSNorm/QK norm、二维 RoPE 和 SwiGLU；Linear 使用
PyTorch 默认初始化，仅 block 的 adaLN modulation 末层置零。无额外输入
LayerNorm，也不缩小输出层初始化。两者的 `t` 都是 clean-image weight。
融合权重始终严格为 `1-t` 和 `t`，但这并不保证两路 token 范数相等或不增长。
训练结构化日志记录原生、投影后 DINO、两条加权分量和融合后 token 的范数，
以及新 projector 的参数/梯度范数。

本实验仅对齐官方 projector；LoRA/MLS、带噪 student 不做输入预处理、
JiT 输入的时间加权融合均保留本项目设计。官方 GroundingConditioner 仍做
ImageNet 预处理，并将 frozen DINO 表征用于 block 条件，这与本实验不同。

2026-10-07 已停止旧的近似 PgT 实验；旧日志、checkpoint 及当时的核心代码、
配置保存于
`checkpoints/aborted_runs/Ours-J18-new-drop0.1-l4-BaseT-Residual-NoDinoPre-PgT-BlendT-B-80_legacy_projector_20261007_0125/`。
修正后的实验沿用原脚本/输出目录，从 epoch 0 重新开始，不恢复旧 projector
的 optimizer、EMA 或 checkpoint。测试 `tests/test_pixeldit2_projector.py`
直接对照官方源码，检查 B 模型初始化、FP32/BF16 前向与反向梯度。

`dino_grounding_drop_on_label_drop: false` 是默认值：CFG null-label 分支仍
使用 DINO path。将该开关设为 `true` 后，训练中的 dropped-label 样本和 CFG
采样中的 null-label 样本都改用完整 native JiT token；本次实验不开启此消融。

该结构不能从旧 Residual checkpoint 恢复 EMA/优化器；新配置有独立输出目录。
单文件脚本固定使用 0,1,2,3 四卡，不依赖 `runtime_env.sh`，只需：

```bash
bash 'scripts/train_BaseT-Residual(t,1-t)-NoDinoPre-PgT-B-80_0123.sh' --preflight
bash 'scripts/train_BaseT-Residual(t,1-t)-NoDinoPre-PgT-B-80_0123.sh'
```

2026-10-08 将本实验目录、YAML 和脚本改为与实验表一致的简称。现有
`checkpoints/BaseT-Residual(t,1-t)-NoDinoPre-PgT-B-80/checkpoint-last.pth`
保存至 31 epoch；恢复时从第 32 个 epoch 开始，保留 optimizer 和 EMA。
诊断日志文件名保留原样，继续追加到已有文件。

### NoDDTHead 消融

实验名为 `BaseT-Residual(t,1-t)-NoDinoPre-PgT-NoDDTHead-B-80`，对应同名
YAML 和 checkpoint 目录。配置设置 `use_ddt_head: false`，不构建 DDTHead，
并启用原生 `net.final_layer` 的训练；JiT 空间 token 使用 `t_emb + y_emb`
调制后输出像素 patch。训练、online eval 和独立 CFG 采样共用此开关。
其余 PgT、时间融合、iREPA 和训练超参数与上面的实验保持一致。

```bash
bash 'scripts/train_BaseT-Residual(t,1-t)-NoDinoPre-PgT-NoDDTHead-B-80_4567.sh' --preflight
bash 'scripts/train_BaseT-Residual(t,1-t)-NoDinoPre-PgT-NoDDTHead-B-80_4567.sh'
```

该实验已于 2026-10-08 在 4567 四卡从头启动，之后只恢复自身目录的 checkpoint。
不要加载含 DDTHead 实验的 optimizer 或 EMA。旧配置未设置
`use_ddt_head` 时默认开启，保留原结构。命令行可用 `--no-use_ddt_head`
关闭。脚本名和实验路径包含括号，执行命令时必须加引号。

## 入口

- 核心模型：`dino_jit_denoiser.py`
- 参数解析：`main_dino_jit.py`
- 默认配置：`configs/Ours-J18-new-drop0.1-l4-BaseT-Residual-B-80.yaml`
- 四卡脚本：`scripts/train_j18_new_drop0p1_l4_baset_residual_b80_local.sh`
- 设计依据：`readme/dino_lora_frontend_cfg_failure_and_redesign.md`

该模型新增 residual adapter，并重新启用 JiT PatchEmbed，不能从旧 BaseT
checkpoint 恢复 optimizer/EMA。脚本只会从本实验自己的
`checkpoints/Ours-J18-new-drop0.1-l4-BaseT-Residual-B-80/checkpoint-last.pth`
断点续训；目录中没有该文件时从头训练。

## 验证与启动

```bash
cd /data0/zhangruobei/Work/Pixel_Diffusion/Ours_BaseT_Residual_1008
PYTHONPATH=.:jit /data0/zhangruobei/anaconda3/envs/irepa/bin/python -m pytest -q tests
bash scripts/train_j18_new_drop0p1_l4_baset_residual_b80_local.sh --preflight
bash scripts/train_j18_new_drop0p1_l4_baset_residual_b80_local.sh
```

## 两个 Residual 消融

当前基准是 `Ours-J18-new-drop0.1-l4-BaseT-Residual-B-80`，其 residual
融合为：

```text
h_native = JiT_PatchEmbed(z_t)
r = ResidualAdapter(DINO-LoRA-MLS(z_t))
h_0 = h_native + t * r + pos
```

其中 `t` 是 clean-image weight。两个消融只改变 residual 的门控规则，其他
数据、模型、损失、优化和采样参数都复制基准配置。

### `Ours-J18-new-drop0.1-l4-BaseT-Residual-G1-B-80`

配置文件为
`configs/Ours-J18-new-drop0.1-l4-BaseT-Residual-G1-B-80.yaml`，训练脚本为
`scripts/train_j18_new_drop0p1_l4_baset_residual_g1_b80_local.sh`。它设置
`dino_residual_gate=constant_one`，因此：

```text
h_0 = h_native + r + pos
```

这个实验检验性能变化是否来自 `t` 对 representation residual 的逐样本加权。

### `Ours-J18-new-drop0.1-l4-BaseT-Residual-DropRes-B-80`

配置文件为
`configs/Ours-J18-new-drop0.1-l4-BaseT-Residual-DropRes-B-80.yaml`，训练脚本为
`scripts/train_j18_new_drop0p1_l4_baset_residual_dropres_b80_local.sh`。它保留
`dino_residual_gate=clean_linear`，并设置
`dino_residual_drop_on_label_drop=true`：

```text
h_0 = h_native + t * r + pos,       label is kept
h_0 = h_native + 0 * r + pos,       label is dropped
```

训练时模型保存实际的 label-drop mask，只对被替换为 null class 的样本清零
residual。CFG 采样的 unconditional 分支使用同一规则：当 label 为 null class
时 residual 也被清零。这样该分支不会通过 DINO representation residual 获得
类别相关信息，有利于检验更干净的 unconditional 表征是否改善 CFG。

两个新实验都从各自的输出目录独立开始；不会加载现有
`Ours-J18-new-drop0.1-l4-BaseT-Residual-B-80` 的 checkpoint。

## 4567 四卡续训

已有 Residual checkpoint 的续训使用独立脚本，缺少 `checkpoint-last.pth` 时直接报错，避免从头训练：

```bash
cd /data0/zhangruobei/Work/Pixel_Diffusion/Ours_BaseT_Residual_1008
bash scripts/resume_j18_new_drop0p1_l4_baset_residual_b80_4567.sh
```

脚本固定使用 4,5,6,7 四卡，直接调用 `/data0/zhangruobei/anaconda3/envs/irepa/bin/torchrun --standalone --nproc_per_node=4`，并加载本项目的 Residual YAML。无需手动激活 conda，不依赖公共环境脚本。每卡 batch 64、全局 batch 256、梯度累积 1 次、学习率 1e-4、每卡 DataLoader workers 12；`epochs=80` 表示训练到总计 80 epoch。模型、两套 EMA、优化器和 epoch 从本实验的 `checkpoint-last.pth` 恢复，未保存的 epoch 中途进度会重跑。日志追加到同一目录的 `train.log`，Residual 诊断指标继续写入 `residual_diagnostics.jsonl/csv`。

续训保留 `g(t)=t` 的 representation residual、BaseT、对齐 depth 4、alignment weight 0.5 和 `hflip_prob=0.5`。每 20 epoch 使用 Heun-50、CFG=1、50K 张图进行 online FID/IS。online eval 的真实图像统计文件和 Inception 评测网络固定从共享 `/data0/zhangruobei/Work/Pixel_Diffusion/fid_stats` 读取：`jit_in256_stats.npz` 与 `weights-inception-2015-12-05-6726825d.pth`，不会再回退到用户 Torch 缓存。所有训练脚本直接赋值 GPU 和资源路径，加载对应 YAML，不设置 NCCL/P2P 或编译禁用项；首次运行可能有编译开销。

## 独立采样与评测

`scripts/sample_and_evaluate_fdr6.sh CONFIG CHECKPOINT EVAL_DIR` 直接加载指定 checkpoint，使用共享 `nanogen-evals` 评测 FID、IS 和 FDR6。默认 4,5,6,7 四卡、FlowDPMSolver-100、CFG 2.5、区间 `[0,1]`、50K 图片、每卡 batch 128。三个路径参数后可追加 `--gpus=0,1,2,3 --cfg=2.3`；其他选项为 `--sampler`、`--steps`、`--gen-bsz`、`--num-images`、`--interval-min`、`--interval-max`，统一使用 `--参数=值`。`--preflight` 只验证资源并生成采样配置，不启动 GPU 任务。

`eval_j18_new_baset_residual_b20_cfg2p5_flowdpm100_local.sh` 是基准 Residual epoch-20 的固定入口；其他 checkpoint、G1 和 DropRes 使用通用采样脚本传入各自 YAML 和 checkpoint。旧的环境变量参数覆盖已移除。不同采样设置使用不同输出目录；已有 `samples.npz` 时跳过生成，直接重算离线指标。
