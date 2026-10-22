# DINOv3 + LoRA + JiT：两路独立 RMSNorm 实验

本目录从 `Ours_BaseT_Residual_1008` 的本地版本衍生，研究**生成侧 DINO token 与 JiT 原生 patch token，分别经过 RMSNorm 后再按时间权重融合**。训练时另有冻结的 DINO teacher，为 JiT 中间层提供 iREPA 对齐目标。

**GitHub 源码版**：只同步模型、配置、依赖和第三方源码；启动脚本、测试、checkpoint、权重、日志、采样结果和实验表格留在各自机器。

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
| 融合权重 | [blend_time.py](blend_time.py) | 仅变换融合系数中的 t；可选 `sigmoid(key*(t-a))`，默认 key=100、a=0.1。 |
| 可后看 | [jit_ddt_head.py](jit_ddt_head.py) | 参考 RAEv2/JiT 的可选 DDT 输出头；当前 NoDDTHead 配置不启用。 |

“我们”包括从 1008 继承的项目扩展，不表示全部在 1016 新写。上游改编文件保留了来源及许可。

| 目录 / 文件 | 内容 |
|---|---|
| `jit/` | iREPA 的 JiT 基础代码，以及本地训练、配置和评测适配，见下一节。 |
| `SSL_code/dinov3/` | DINOv3 上游源码的本地副本；包装器通过 `torch.hub.load(..., source="local")` 加载。 |
| `configs/` | 各实验 YAML，决定实际启用的模型路径和超参数。 |
| `scripts/`（仅本地） | 各机器自己的启动与排队工具，不随源码发布。 |
| `tests/`（仅本地） | 回归测试留在开发机，不随源码发布。 |
| `third_party/` | 上游许可证和来源声明，不是另一套模型实现。 |
| `require.txt` | Python 依赖清单。 |
| `checkpoints/` | 训练输出、评测结果、队列及源码快照，不是需要阅读的模型源码。 |

## 2. `jit/` 哪些是原版，哪些已修改？

**JiT 模型主体基本沿用 iREPA，但整个 `jit/` 并非未经修改的原版。** 以下与当前本机 `iREPA/jit` 对照；“计算一致”不表示注释、格式或文件字节完全相同。

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
torchrun main_dino_jit.py + configs/实验.yaml
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
- **teacher 的输入和输出保持原样。** `_encode_teacher_target()` 内写死 ImageNet 标准化（FP32 下 `[-1,1]` → `[0,1]` → 减均值除方差），跟随 iREPA 源码的固定预处理；独立 teacher 不重复归一化、不量化或 resize。它按本地 iREPA 的 `DINOv3Encoder` 调用 `forward_features(x)["x_norm_patchtokens"]`，保留预训练 final LN 的 gamma/beta。内部使用 `no_grad`，即使外部调用 `.train()` 也保持 eval。teacher 继续放在普通 list 中，旧 checkpoint 不需要转换。
- **student 输入只有两种。** `dino_preprocess_input: false`=NoDinoPre（带噪图像直接进 DINO+LoRA）；`true`=t 相关解析 norm（按 `z_t=t·x+(1-t)·ε` 的解析均值/方差标准化）。
- teacher 沿用 iREPA 的 `forward_features()` 接口，不再实现额外的 `forward()`；生成侧仍通过 `dino_student(input)` 前向并反传。
- **两个 projector 不要混淆。** `pixeldit2_projector.py` 的 PgT 处理生成分支 DINO token；`jit/projectors.py` 的 ConvProj 处理 JiT 第4层特征，只用于训练对齐。
- wrapper 的 `_spatial_normalize_target()` / `_alignment_loss()` 实现空间 z-score 与负 cosine，`total_loss = denoise_loss + 0.5 * alignment_loss`。
- B/16、256×256 输入对应空间 token `[B, 256, 768]`，这里 `B` 是每卡 batch。

## 4. 常用开关在哪生效

参数由 `main_dino_jit.py` 声明；student 开关经 `dino_encoder.build_dino_student` 和 `dino_v3_lora.py` 生效。`build_irepa_teacher` 只读取 backbone 名称和预训练权重路径，返回 `irepa_teacher.py` 的独立冻结编码器，不导入 LoRA 包装。底层加载在 `dinov3_backbone.py`，融合开关在 `dino_jit_denoiser.py` 执行。

| 参数 | 含义 |
|---|---|
| `irepa_alignment` | 默认开启；`--no-irepa_alignment` 为 noirepa 消融：不构建 frozen teacher 与对齐 ConvProj，训练只算 velocity loss。 |
| `dino_preprocess_input` | 只控制 DINO+LoRA 输入：`false` 原样输入；`true` 调用原有 t 相关解析 norm。teacher 始终使用 ImageNet 标准化。 |
| `dino_jit_input_mode` | 仅两种：`native` 只用 JiT patch 输入；`time_blend` 按时间融合 JiT 与 DINO token（默认）。两种模式都训练 JiT patch embedding。 |
| `dino_time_blend_reverse` | `false`：`t*DINO + (1-t)*JiT`；`true`：`(1-t)*DINO + t*JiT`。不改变扩散、JiT 或 PgT 使用的原始 `t`。 |
| `dino_time_blend_transform` | 默认 `linear` 保持历史行为；`sigmoid` 仅把融合系数中的 `t` 替换为 `f(t)=sigmoid(key*(t-a))`。 |
| `dino_time_blend_key` | sigmoid 的固定斜率，默认100，不是可学习权重。 |
| `dino_time_blend_a` | sigmoid 的固定中心点，默认0.1；t=a 时两路各占0.5。旧配置未填写时仍用0.1。 |
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
dino_time_blend_a: 0.1
```

对应公式为 `(1-f(t))*DINO + f(t)*JiT`；在原有 RMSNorm 之后加权，PgT、JiT 时间条件、加噪、velocity/loss、采样时间步及 CFG interval 均仍使用原始 `t`。训练和采样共用 `_compose_jit_input_tokens()`，日志中的 `t_mean` 也仍是原始时间。原始 reverse YAML 保持线性权重，便于复现旧结果。

`a=0.3、key=20` 使用 [reverse_sigmoid_blend_k20_a0p3_B80.yaml](configs/reverse_sigmoid_blend_k20_a0p3_B80.yaml)：8×128=1024，LR 2e-4，80 epoch，从头训练。输出实验名含 `SigmoidBlendK20-A0p3`；中心点保存在配置及 checkpoint 参数中，恢复训练和采样时应沿用相同值。

## 5. 源码版的运行方式

本次发布不含开发机的 SSH、tmux、训练或排队脚本。训练入口是 `main_dino_jit.py`，参数可以在 YAML 中配置，也可以通过命令行覆盖。先安装 [require.txt](require.txt) 中的依赖，并自行准备 ImageNet、DINOv3 权重及 FID 评测资源；这些数据不会随源码上传。

在具备八卡 GPU 的机器上，从本目录启动 sigmoid 融合消融的示例（请替换资源路径）：

```bash
export IMAGENET_ROOT=/path/to/imagenet
export PRETRAIN_MODELS_ROOT=/path/to/pretrain_models
export FID_STATS_ROOT=/path/to/fid_stats

# 使用尚未存在的新输出目录；resume 为空表示从头训练。
OUTPUT_DIR=checkpoints/reverse_sigmoid_k20_a0p3_b80_new

torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 \
  main_dino_jit.py \
  --config configs/reverse_sigmoid_blend_k20_a0p3_B80.yaml \
  --batch_size 128 --global_batch_size 1024 --blr 5e-5 \
  --output_dir "${OUTPUT_DIR}" --resume '' \
  --data_path "${IMAGENET_ROOT}" --online_eval
```

该配置为 80 epoch，8×128=global batch1024，`blr=5e-5` 对应 LR `2e-4`，warmup5 后 constant；LoRA r/alpha64，EMA .9999/.9996。每20 epoch 做 Heun50、CFG1、interval[0,1]、50K 在线评测。这里不自动执行开发机脚本提供的资源和恢复状态预检，运行前请确认资源路径及输出目录；不要复用不同消融的 checkpoint/optimizer。

| 资源 / 输出 | 配置方式 |
|---|---|
| ImageNet | 设置 `IMAGENET_ROOT` 或 `--data_path`。 |
| DINOv3 权重 | 设置 `PRETRAIN_MODELS_ROOT`，目录中准备 `dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth`；也可覆盖 `--dino_weights`。 |
| 在线 FID 资源 | 设置 `FID_STATS_ROOT`。具体文件名由 `jit/engine_jit.py` 中的在线评测设置决定。 |
| 离线评测资源 | `NANOGEN_EVALS_ROOT` 指向自行准备的 nanogen-evals 目录。 |
| 训练配置、日志、checkpoint | `--output_dir` 指定的目录，不提交 Git。 |
| 离线采样与评测入口 | `jit/evaluation/prepare_sampling.py` 和 `jit/evaluation/evaluate_fdr6.py`。 |

本目录与 `../iREPA__rmsnorm2_Residual_1018/` 是不同实现版本；不要仅凭相似实验名称混用配置或 optimizer 状态。历史 YAML 的实验 ID 保留，用来对应已有结果。

## 6. 回归检查

回归测试留在开发机（`tests/`，不随源码发布）。`third_party/` 中的 LICENSE、NOTICE 和来源说明保留。
