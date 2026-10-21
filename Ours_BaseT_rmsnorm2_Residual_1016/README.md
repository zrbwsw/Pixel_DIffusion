# DINOv3 + LoRA + JiT：两路独立 RMSNorm 实验

本目录从 `Ours_BaseT_Residual_1008` 的本地版本衍生，研究**生成侧 DINO token 与 JiT 原生 patch token，分别经过 RMSNorm 后再按时间权重融合**。训练时另有冻结的 DINO teacher，为 JiT 中间层提供 iREPA 对齐目标。

先读根目录的接线和模型文件，不必从整个 `jit/` 或 `SSL_code/` 开始。当前关注的 reverse / NoDDTHead 配置使用 `(1-t) × DINO + t × JiT`，输出走 JiT 自带的 final layer。

## 1. 从哪些文件开始读

| 顺序 | 文件 | 职责与来源 |
|---|---|---|
| 1 | [configs/reverse_train_final_norm_B80_8gpu.yaml](configs/reverse_train_final_norm_B80_8gpu.yaml) | 当前 final LN 消融配方：模型开关、训练与在线评测参数。 |
| 2 | [main_dino_jit.py](main_dino_jit.py) | 我们的入口：追加参数，将模型和数据集接入 iREPA 训练框架。 |
| 3 | [dino_jit_denoiser.py](dino_jit_denoiser.py) | 我们的核心模型：两路 token 融合、JiT 前向、去噪与对齐损失、采样。 |
| 4 | [dino_encoder.py](dino_encoder.py) | 我们的工厂：`build_irepa_teacher` / `build_dino_student` 独立构建两条编码器路径。 |
| teacher | [irepa_teacher.py](irepa_teacher.py) | 独立冻结编码器，按 iREPA 的 `DINOv3Encoder` 取最后一层 normalized patch tokens；内部冻结并保持 eval，不使用 student 包装或开关。 |
| backbone | [dinov3_backbone.py](dinov3_backbone.py) | 只解析本地源码/权重并加载 DINOv3。两条路径各自加载独立实例，不含 LoRA 或训练策略。 |
| 5 | [dino_v3_lora.py](dino_v3_lora.py) | 生成侧 student 包装：选层、MLS 聚合、final LN、冻结/解冻；teacher 不依赖此文件。 |
| 6 | [dino_lora.py](dino_lora.py) | 我们的 LoRA 工具：冻结原 Linear，向 qkv/proj/fc1/fc2 注入可训练低秩增量。 |
| 7 | [pixeldit2_projector.py](pixeldit2_projector.py) | 改编自 PixelDiT2 的生成侧 PgT：处理 DINO token，并注入时间条件。 |
| 融合权重 | [blend_time.py](blend_time.py) | 仅变换融合系数中的 t；可选 `sigmoid(key*(t-0.1))`，默认 key=100。 |
| 可后看 | [jit_ddt_head.py](jit_ddt_head.py) | 参考 RAEv2/JiT 的可选 DDT 输出头；当前 NoDDTHead 配置不启用。 |

“我们”包括从 1008 继承的项目扩展，不表示全部在 1016 新写。上游改编文件保留了来源及许可。

| 目录 / 文件 | 内容 |
|---|---|
| `jit/` | iREPA 的 JiT 基础代码，以及本地训练、配置和评测适配，见下一节。 |
| `SSL_code/dinov3/` | DINOv3 上游源码的本地副本；包装器通过 `torch.hub.load(..., source="local")` 加载。 |
| `configs/` | 各实验 YAML，决定实际启用的模型路径和超参数。 |
| `scripts/` | 环境、资源预检、训练启动、采样队列、分析工具。 |
| `tests/` | 本地回归测试，不参与训练或采样。 |
| `third_party/` | 上游许可证和来源声明，不是另一套模型实现。 |
| `require.txt` | 安装脚本使用的 Python 依赖清单。 |
| `checkpoints/` | 训练输出、评测结果、队列及源码快照，不是需要阅读的模型源码。 |

## 2. `jit/` 哪些是原版，哪些已修改？

**JiT 模型主体基本沿用 iREPA，但整个 `jit/` 并非未经修改的原版。** 以下与当前本机 [iREPA/jit](../iREPA/jit/) 对照；“计算一致”不表示注释、格式或文件字节完全相同。

| 文件 / 目录 | 来源与当前用途 |
|---|---|
| `model_jit.py` | 沿用 JiT/iREPA 的模型主体，核心计算一致；当前模型复用其中各层。 |
| `projectors.py` | 沿用 iREPA 投影层，当前使用 `ProjectionLayer("conv")` 做特征对齐。 |
| `projection_loss.py`、`spnorm.py` | 保留 iREPA 对齐组件，核心计算一致；当前 wrapper 的对齐计算写在根目录模型文件中，没有直接调用这两个文件。 |
| `util/`、`models/` | 基础工具和模型组件，核心计算与本机 iREPA 一致，一般不必先读。 |
| `main_jit.py` | 沿用训练框架，已有本地配置、路径、梯度累积和采样入口适配；负责数据、DDP、AdamW、checkpoint。 |
| `engine_jit.py` | 已适配当前模型调用，以及 BF16、梯度累积、EMA、诊断日志、在线评测、样本导出。 |
| `config_utils.py`、`resource_paths.py` | 本地配置/资源工具：分组 YAML、命令行覆盖、环境变量与共享目录解析。 |
| `evaluation/` | 本地整合的离线评测入口及第三方 evaluator，处理采样配置、NPZ、FID/IS/FDR6；并非全部从零实现。 |
| `dataset.py` | 保留 HF Arrow 读取器；当前入口用 `FlexibleImageNet256Dataset` 兼容 raw ImageNet 和 Arrow。 |
| `denoiser.py`、`vision_encoder.py` | 保留 baseline/历史通用组件；前者还有本地诊断和 FlowDPM 采样扩展。当前入口的模型和 DINO 构建走根目录文件。 |

当前 `dino_jit_denoiser.py::_jit_forward_from_tokens()` 复用 `net` 的各层，自己执行 block 循环，以接入融合 token、提取中间特征；**不是直接调用原始 `JiT.forward()`**。

## 3. 一次训练如何流过这些文件

```text
scripts/启动脚本 + configs/实验.yaml
  → main_dino_jit.py：解析参数，替换 main_jit.Denoiser 和数据集构造器
  → jit/main_jit.py：建数据集、模型、DDP、优化器，处理 checkpoint
  → jit/engine_jit.py：调用 model(x, labels)，反向、更新、EMA、日志与评测
  → dino_jit_denoiser.py / DINOJiTDenoiser：生成与对齐计算
```

以 reverse / NoDDTHead 为例，`t` 是干净图像权重：`z_t = t*x + (1-t)*noise`。

```text
带噪图像 z_t
  ├─ JiT patch embed → JiT RMSNorm → ×t ─────────────────────┐
  └─ DINOv3+LoRA → final LN → MLS → PgT(t) → DINO RMSNorm → ×(1-t)
                                                           ↓ 相加
                                               加位置编码 → JiT blocks
                                                 ├─ 第4层 → ConvProj → 对齐 loss
                                                 └─ final layer → 干净图像预测 → 去噪 loss
干净图像 x → ImageNet 标准化 → 冻结 DINO teacher → 最后层 token → 对齐目标
```

- **student 和 teacher 是独立实例。** student 看带噪图像，使用 LoRA/MLS；teacher 看干净图像、冻结、无 LoRA/MLS，不进入优化器或 EMA。采样不需要 teacher 的对齐计算。
- **teacher 的输入和输出保持原样。** `_encode_teacher_target()` 内写死 ImageNet 标准化（FP32 下 `[-1,1]` → `[0,1]` → 减均值除方差），跟随 iREPA 源码的固定预处理；独立 teacher 不重复归一化、不量化或 resize。它按本地 [iREPA 的 DINOv3Encoder](../iREPA/jit/vision_encoder.py) 调用 `forward_features(x)["x_norm_patchtokens"]`，保留预训练 final LN 的 gamma/beta。内部使用 `no_grad`，即使外部调用 `.train()` 也保持 eval。teacher 继续放在普通 list 中，旧 checkpoint 不需要转换。
- **student 输入只有两种。** `dino_preprocess_input: false`=NoDinoPre（带噪图像直接进 DINO+LoRA）；`true`=t 相关解析 norm（按 `z_t=t·x+(1-t)·ε` 的解析均值/方差标准化）。
- teacher 沿用 iREPA 的 `forward_features()` 接口，不再实现额外的 `forward()`；生成侧仍通过 `dino_student(input)` 前向并反传。
- **两个 projector 不要混淆。** `pixeldit2_projector.py` 的 PgT 处理生成分支 DINO token；`jit/projectors.py` 的 ConvProj 处理 JiT 第4层特征，只用于训练对齐。
- wrapper 的 `_spatial_normalize_target()` / `_alignment_loss()` 实现空间 z-score 与负 cosine，`total_loss = denoise_loss + 0.5 * alignment_loss`。
- B/16、256×256 输入对应空间 token `[B, 256, 768]`，这里 `B` 是每卡 batch。

## 4. 常用开关在哪生效

参数由 `main_dino_jit.py` 声明；student 开关经 `dino_encoder.build_dino_student` 和 `dino_v3_lora.py` 生效。`build_irepa_teacher` 只读取 backbone 名称和预训练权重路径，返回 `irepa_teacher.py` 的独立冻结编码器，不导入 LoRA 包装。底层加载在 `dinov3_backbone.py`，融合开关在 `dino_jit_denoiser.py` 执行。

| 参数 | 含义 |
|---|---|
| `dino_preprocess_input` | 只控制 DINO+LoRA 输入：`false` 原样输入；`true` 调用原有 t 相关解析 norm。teacher 始终使用 ImageNet 标准化。 |
| `dino_jit_input_mode` | 仅两种：`native` 只用 JiT patch 输入；`time_blend` 按时间融合 JiT 与 DINO token（默认）。两种模式都训练 JiT patch embedding。 |
| `dino_time_blend_reverse` | `false`：`t*DINO + (1-t)*JiT`；`true`：`(1-t)*DINO + t*JiT`。不改变扩散、JiT 或 PgT 使用的原始 `t`。 |
| `dino_time_blend_transform` | 默认 `linear` 保持历史行为；`sigmoid` 仅把融合系数中的 `t` 替换为 `f(t)=sigmoid(key*(t-0.1))`。 |
| `dino_time_blend_key` | sigmoid 的固定超参数，默认100，不是可学习权重；中心点固定为0.1。 |
| `dino_jit_pre_blend_rmsnorm` | 同时控制融合前的两路 RMSNorm；本版本两路各有自己的可学习 gamma。 |
| `dino_aggregation`、`dino_layers` | 生成 DINO 的取层与聚合；B/16 的 `mls/default` 使用 `[2,5,8,11]`。 |
| `dino_train_patch_embed` | 是否训练生成 DINO 的 patch embedding。 |
| `dino_generation_final_norm_affine` | 是否保留生成 DINO final LN 的 gamma/beta；`false` 是 NoAffine 消融。 |
| `dino_train_final_norm` | 默认 `false`；`true` 额外解冻生成 DINO 的 `model.norm.weight/bias`，要求 affine 保留。 |
| `dino_grounding_drop_on_label_drop` | DropRes：类别被丢弃时，融合权重改为 DINO=0、JiT=1。 |
| `use_ddt_head` | NoDDTHead 为 `false`，使用 JiT 原生输出层；历史 DDT 配置仍依赖 `jit_ddt_head.py`。 |

**DINO final LN、融合 RMSNorm、block 内部 norm 是不同位置。** final LN 解冻保留预训练 gamma/beta 初值，B/16 增加 `768+768=1,536` 个可训练参数。MLS 的多个选中层输出共用此 final LN，再聚合；teacher 和 block 内部 norm 不随开关解冻。

两路 RMSNorm 参数为 `jit_pre_blend_norm.weight`、`dino_pre_blend_norm.weight`，互不共享；学习到的 gamma 可以使归一化后的两路范数不同。

解析 norm 继续使用 `_preprocess_noisy_student_for_dino()`：令 `μ=2·ImageNet_mean−1`、`σ=2·ImageNet_std`，返回 `(z_t−t·μ) / sqrt(t²·σ²+(1−t)²·noise_scale²)`。训练和采样共用这条路径；这里使用原始扩散时间 t，不使用 sigmoid 融合权重 f(t)。当前 reverse / sigmoid 配方仍为 `dino_preprocess_input: false`。

这里的 `true` 现表示解析 norm，不再表示旧版的固定 ImageNet student 预处理。复现采用旧语义的历史实验，应使用该实验保存的源码和配置；`checkpoints/` 下的产物与队列快照不随本次修改。

sigmoid 单因素配置为 [reverse_sigmoid_blend_B80.yaml](configs/reverse_sigmoid_blend_B80.yaml)，由 reverse B80 配方复制，只改变融合权重与实验标识，从头训练。其模型部分设置：

```yaml
dino_time_blend_reverse: true
dino_time_blend_transform: sigmoid
dino_time_blend_key: 100.0
```

对应公式为 `(1-f(t))*DINO + f(t)*JiT`；在原有 RMSNorm 之后加权，PgT、JiT 时间条件、加噪、velocity/loss、采样时间步及 CFG interval 均仍使用原始 `t`。训练和采样共用 `_compose_jit_input_tokens()`，日志中的 `t_mean` 也仍是原始时间。原始 reverse YAML 保持线性权重，便于复现旧结果。

## 5. 脚本、资源和输出去哪找

方法脚本已整理为“GPU/路径设置 + 明确的 `torchrun` 参数”，默认前台运行，tmux 由你自己创建。无 GPU 的本机会经独立的 `run_on_gpu.sh` 转发；已在 GPU 节点则直接执行。完整用法见 [scripts/README.md](scripts/README.md)。

| 用途 | `scripts/` 下的入口 |
|---|---|
| 通用训练 / 资源预检 | `train_portable.sh`、`check_runtime.py` |
| 基准八卡 / 四卡 | `train_rmsnorm2_B80_8gpu.sh` / `train_rmsnorm2_B80_4gpu.sh` |
| final LN 可训练，八卡 | `train_rmsnorm2_reverse_train_final_norm_B80_8gpu.sh` |
| reverse + sigmoid 融合权重 | `train_rmsnorm2_reverse_sigmoid_blend_B80_4gpu.sh`；GPU1 八卡用 `train_rmsnorm2_reverse_sigmoid_blend_B80_8gpu.sh` |
| 四卡 reverse / NoAffine / DropRes | `train_rmsnorm2_reverse_B80_4gpu.sh`、`train_rmsnorm2_noaffine_B80_4gpu.sh`、`train_rmsnorm2_dropres_B80_4gpu.sh` |
| NoMLS / PMean0 八卡配方 | `train_rmsnorm2_reverse_nomls_B80_8gpu.sh`、`train_rmsnorm2_reverse_pmean0_B80_8gpu.sh` |
| 连接与执行公共工具 | `run_on_gpu.sh` 负责 SSH；`train_env.sh` / `check_training_launch.py` 负责环境、前台运行与记录 |
| 离线采样与评测 | `sample_portable.sh`；默认 FlowDPM100，Heun50 需传 `--sampler=heun --steps=50` |
| 训练后采样队列 / 范数分析 | `queue_posttrain_heun50.py` / `plot_reverse_token_norms.py` |

final LN 消融配方：80 epoch，8×128=global batch1024，`blr=5e-5` 对应 LR `2e-4`，warmup5 后 constant；LoRA r/alpha64，EMA .9999/.9996。每20 epoch 做 Heun50、CFG1、interval[0,1]、50K 在线评测。该实验已于2026-10-20启动，不能把已有输出当空目录重复启动。

先激活已有环境或设置 `PYTHON_BIN`。在本目录只核对配置和资源：

```bash
bash scripts/train_rmsnorm2_reverse_train_final_norm_B80_8gpu.sh --preflight
```

安装新环境用 `scripts/install_requirements.sh`，迁移见 [REMOTE_SETUP.md](../REMOTE_SETUP.md)；GPU 节点使用准备好的离线资源。

| 资源 / 输出 | 默认位置 |
|---|---|
| ImageNet | `/mnt/shared-storage-user/datafactory-share/gushishen/imagenet`；`IMAGENET_ROOT` 可覆盖 |
| 权重 / 在线评测 / 离线评测资源 | `../pretrain_models/`、`../fid_stats/`、`../nanogen-evals/`；分别支持 `PRETRAIN_MODELS_ROOT`、`FID_STATS_ROOT`、`NANOGEN_EVALS_ROOT` |
| 训练配置、命令、日志、checkpoint | `checkpoints/<实验名>/`；新启动器另存含 CLI 覆盖的 `config_resolved.yaml` |
| 离线采样结果 | `checkpoints/eval_checkpoint/<实验名>/<采样设置>/` |
| 队列计划、状态及固定源码快照 | `checkpoints/eval_queue/` |
| 汇总实验表 | [../related_tex/exp0926.tex](../related_tex/exp0926.tex) |

最终实验参数以输出目录保存的配置和命令为准。本目录与 `../iREPA__rmsnorm2_Residual_1018/` 是不同实现版本；1018 的 NoMLS/PMean0/teacher 替换实验应看它自己的脚本与输出，不能仅凭相似名称混用配置和 optimizer 状态。

## 6. 本次精简

| 对象 | 处理及原因 |
|---|---|
| `prepare_imagenet.py` | 已移出本项目并备份。仅一次性解包 ImageNet，训练/采样无调用；裁剪和增强在数据读取链路中完成。 |
| `require.txt` | 保留，`scripts/install_requirements.sh` 仍读取它。 |
| `third_party/` | 保留上游许可证、NOTICE 和来源说明，阅读模型时可略过。 |
| `tests/` | 保留融合、final LN、配置和队列的回归保障；不参与训练，简单移动还会破坏测试相对路径。 |

旧 README 和数据准备工具位于 [首次整理备份](../../work/cleanup-rmsnorm2-readability-20261020T130716/)。随后清理了源码注释中的历史编号，重写方法启动脚本，并移出6个已完成采样/队列脚本及旧自动 tmux 包装器，备份见 [脚本整理记录](../../work/refactor-rmsnorm2-scripts-20261020/)。模型计算和已有实验的超参数未改变；历史配置中的实验 ID、checkpoint 和正在使用的队列源码快照保留。

输入路径已收敛为 `native` / `time_blend`。`native` 仍保留冻结的 iREPA teacher 和对齐损失，但不构建生成侧 DINO+LoRA。旧 `replace` / `gated_residual` 的实现、adapter 和门控参数已删除；对应5份历史 YAML 移至 [旧输入模式备份](../../work/simplify-input-modes-20261021T171554/retired_configs/)，不能用于当前入口。现有 `native` / `time_blend` checkpoint 配置里的三个旧 `dino_residual_*` 字段在读取时忽略，训练产物与队列源码快照保留。

修改模型后的回归检查：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=.:jit python -m pytest -q tests
```
