# Our RMSNorm2 4DiT Residual

这个目录从 `Ours_BaseT_rmsnorm2_Residual_1016` 复制，只有 token 路由改成“四层双路径”结构；teacher、DINO+LoRA、PgT、RMSNorm、损失、采样和 iREPA 训练框架继续沿用原实现。

当前新路径为：

```text
带噪 z_t
  ├─ JiT patch embed → JiT RMSNorm → JiT blocks[0:4] ──────────────────┐
  └─ DINOv3+LoRA → 各选中层分别过 final LN → MLS 聚合 → PgT(t) → DINO RMSNorm
                          → 独立 blocks[0:4]
                          └─ DINO 路第4层输出 → ConvProj → 与 frozen iREPA teacher 算对齐 loss
                                                              ↓
                         按 f(t) 加权相加（两路各自过完前四层才融合）
                                  → 插入 class tokens → JiT blocks[4:]
                                  → final layer → clean image
干净图像 x → ImageNet 标准化 → 冻结 DINO teacher → 最后层 normalized token → 对齐目标
```

两条前缀路径使用完全相同的 JiTBlock 结构和初始权重，但参数不共享：JiT 路使用 `net.blocks[0:4]`，DINO 路使用 `dino_prefix_blocks` 的深拷贝。两路都先加 JiT 位置编码、共用 `c=t_embedder(t)+y_embedder(y)` 条件。第 4 层之后才按原来的 reverse 权重融合；`f(t)` 可以是线性 `t`，也可以是 `sigmoid(key*(t-a))`。MLS 的顺序与 1016 相同：每个选中层先分别过 final LN，再做均值+末层空间均值聚合，聚合后不再过 LN。

冻结 teacher 只看干净图像，保持 ImageNet normalization、最后层 normalized patch tokens 和 `no_grad` 行为不变。DINO 路的第四层输出单独进入 ConvProj，alignment loss 不使用融合后的 token。

新配置和训练入口：

- [configs/Our_rmsnorm2_4DiT_Residual_ft_1024_B80.yaml](configs/Our_rmsnorm2_4DiT_Residual_ft_1024_B80.yaml)
- [fourdit_denoiser.py](fourdit_denoiser.py)
- [scripts/train_our_rmsnorm2_4dit_residual_ft_1024_B80_4gpu.sh](scripts/train_our_rmsnorm2_4dit_residual_ft_1024_B80_4gpu.sh)

新模式由 `dino_jit_input_mode: post_block_time_blend` 和 `dual_path_depth: 4` 控制。`native`、`time_blend` 在代码层仍保留；1016 的旧实验配置不随本目录维护。

其余代码沿用 1016 版本：

- `main_dino_jit.py`：参数解析和 iREPA 训练入口；
- `dino_jit_denoiser.py`：加噪、损失、teacher、旧 token 路由和采样公共逻辑；
- `dino_encoder.py` / `irepa_teacher.py`：独立 student 与冻结 teacher；
- `dino_v3_lora.py`：DINOv3 student、LoRA、MLS 和 final LN 开关；
- `jit/`：iREPA 的 JiT、训练循环、DDP、EMA、checkpoint 和评测代码；
- `SSL_code/dinov3/`：本地 DINOv3 源码。

本目录没有复制 checkpoint、日志和缓存。训练输出默认写入 `checkpoints/Our_rmsnorm2_4DiT_Residual_ft_1024-B-80/`。

本目录从 `Ours_BaseT_rmsnorm2_Residual_1016` 的本地版本衍生，研究**生成侧 DINO token 与 JiT 原生 patch token 分别经过四个 JiT blocks 后再按时间权重融合**。训练时另有冻结的 DINO teacher，为 DINO 路第四层输出提供 iREPA 对齐目标（不对齐融合后的 token）。

先读根目录的接线和模型文件，不必从整个 `jit/` 或 `SSL_code/` 开始。当前主线配置为 `post_block_time_blend` + `dual_path_depth: 4` 的四层双路径结构（见上方数据流）；兼容的 `time_blend` reverse / NoDDTHead 配置使用 `(1-t) × DINO + t × JiT`，输出走 JiT 自带的 final layer。

## 1. 从哪些文件开始读

| 顺序 | 文件 | 职责与来源 |
|---|---|---|
| 1 | [configs/Our_rmsnorm2_4DiT_Residual_ft_1024_B80.yaml](configs/Our_rmsnorm2_4DiT_Residual_ft_1024_B80.yaml) | 当前四层双路径配方：模型开关、训练与在线评测参数。 |
| 2 | [main_dino_jit.py](main_dino_jit.py) | 我们的入口：追加参数，将模型和数据集接入 iREPA 训练框架。 |
| 3 | [fourdit_denoiser.py](fourdit_denoiser.py) | 本目录的核心模型：四层双路径 token 路由、融合与对齐、采样；其余逻辑继承 dino_jit_denoiser.py。 |
| 4 | [dino_encoder.py](dino_encoder.py) | 我们的工厂：`build_irepa_teacher` / `build_dino_student` 独立构建两条编码器路径。 |
| teacher | [irepa_teacher.py](irepa_teacher.py) | 独立冻结编码器，按 iREPA 的 `DINOv3Encoder` 取最后一层 normalized patch tokens；内部冻结并保持 eval，不使用 student 包装或开关。 |
| backbone | [dinov3_backbone.py](dinov3_backbone.py) | 只解析本地源码/权重并加载 DINOv3。两条路径各自加载独立实例，不含 LoRA 或训练策略。 |
| 5 | [dino_v3_lora.py](dino_v3_lora.py) | 生成侧 student 包装：选层、MLS 聚合、final LN、冻结/解冻；teacher 不依赖此文件。 |
| 6 | [dino_lora.py](dino_lora.py) | 我们的 LoRA 工具：冻结原 Linear，向 qkv/proj/fc1/fc2 注入可训练低秩增量。 |
| 7 | [pixeldit2_projector.py](pixeldit2_projector.py) | 改编自 PixelDiT2 的生成侧 PgT：处理 DINO token，并注入时间条件。 |
| 融合权重 | [blend_time.py](blend_time.py) | 仅变换融合系数中的 t；可选 `sigmoid(key*(t-a))`，默认 key=100、a=0.1。 |
| 可后看 | [jit_ddt_head.py](jit_ddt_head.py) | 参考 RAEv2/JiT 的可选 DDT 输出头；当前 NoDDTHead 配置不启用。 |

“我们”包括从 1016 继承的项目扩展，不表示全部代码都是本实验新写。上游改编文件保留了来源及许可。

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
- **student 输入有三种。** `dino_preprocess`：`none`=NoDinoPre（带噪图像直接进 DINO+LoRA）；`t_moment`=t 相关解析 norm（按 `z_t=t·x+(1-t)·ε` 的解析均值/方差标准化）；`imagenet`=与 teacher 相同的固定 ImageNet 标准化（PixelDiT2 grounding 方式，不使用 t）。旧布尔键 `dino_preprocess_input`（true=t_moment、false=none）仍兼容。
- teacher 沿用 iREPA 的 `forward_features()` 接口，不再实现额外的 `forward()`；生成侧仍通过 `dino_student(input)` 前向并反传。
- **两个 projector 不要混淆。** `pixeldit2_projector.py` 的 PgT 处理生成分支 DINO token；`jit/projectors.py` 的 ConvProj 处理 JiT 第4层特征，只用于训练对齐。
- wrapper 的 `_spatial_normalize_target()` / `_alignment_loss()` 实现空间 z-score 与负 cosine，`total_loss = denoise_loss + 0.5 * alignment_loss`。
- B/16、256×256 输入对应空间 token `[B, 256, 768]`，这里 `B` 是每卡 batch。

## 4. 常用开关在哪生效

参数由 `main_dino_jit.py` 声明；student 开关经 `dino_encoder.build_dino_student` 和 `dino_v3_lora.py` 生效。`build_irepa_teacher` 只读取 backbone 名称和预训练权重路径，返回 `irepa_teacher.py` 的独立冻结编码器，不导入 LoRA 包装。底层加载在 `dinov3_backbone.py`，融合开关在 `dino_jit_denoiser.py` 执行。

| 参数 | 含义 |
|---|---|
| `irepa_alignment` | 默认开启；`--no-irepa_alignment` 为 noirepa 消融：不构建 frozen teacher 与对齐 ConvProj，训练只算 velocity loss，日志无 alignment_* 指标。 |
| `dino_preprocess` | student 输入预处理三选一：`none` 原样输入；`t_moment` t 相关解析 norm；`imagenet` 与 teacher 相同的固定 ImageNet 标准化（PixelDiT2 方式）。旧布尔键 `dino_preprocess_input` 仍兼容（true=t_moment、false=none），两者都未设置时默认 `t_moment`。teacher 始终使用 ImageNet 标准化。 |
| `dino_student_time_cond` | 默认 `false`；`true` 时在 patch_embed 后给每个 patch token 加 JiT 同款 `t_embedder(t)`，让 LoRA 感知扩散时间。 |
| `dino_student_label_cond` | 默认 `false`；`true` 时同位置加 JiT 同款 `y_embedder(y)`，跟随 label drop（null 类=1000），训练与 CFG 采样一致。 |
| `dino_jit_input_mode` | `native` 只用 JiT patch 输入；`time_blend` 在 blocks 前融合；`post_block_time_blend` 让两路各过前4个独立 blocks 后再融合。三种模式都训练 JiT patch embedding。 |
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

sigmoid 等 1016 实验配方不在本目录维护；本目录只保留本方法配置 [Our_rmsnorm2_4DiT_Residual_ft_1024_B80.yaml](configs/Our_rmsnorm2_4DiT_Residual_ft_1024_B80.yaml)。融合权重语义与 1016 一致：对应公式为 `(1-f(t))*DINO + f(t)*JiT`（reverse 开启时）；PgT、JiT 时间条件、加噪、velocity/loss、采样时间步及 CFG interval 均仍使用原始 `t`，`f(t)` 可以是线性 `t` 或 `sigmoid(key*(t-a))`。1016 的历史配方与启动脚本见 `../Ours_BaseT_rmsnorm2_Residual_1016/`。

## 5. 脚本、资源和输出去哪找

方法脚本已整理为“GPU/路径设置 + 明确的 `torchrun` 参数”，默认前台运行，tmux 由你自己创建。无 GPU 的本机会经独立的 `run_on_gpu.sh` 转发；已在 GPU 节点则直接执行。完整用法见 [scripts/README.md](scripts/README.md)。

| 用途 | `scripts/` 下的入口 |
|---|---|
| 本方法训练（四卡） / 通用训练 | `train_our_rmsnorm2_4dit_residual_ft_1024_B80_4gpu.sh` / `train_portable.sh` |
| 资源预检 | `check_runtime.py`、`portable_env.sh` |
| 连接与执行公共工具 | `run_on_gpu.sh` 负责 SSH；`train_env.sh` / `check_training_launch.py` 负责环境、前台运行与记录 |
| 离线采样与评测 | `sample_ckpt_8gpu.sh`（顶部配置的模板）、`sample_portable.sh`；默认 FlowDPM100，Heun50 需传 `--sampler=heun --steps=50` |

1016 复制来的实验专用脚本（train_rmsnorm2_*、queue_sample_*、队列与分析工具）和旧实验 YAML 已移除；本目录只保留本方法配置与训练/采样模板。1016 的实验产物在它自己的目录。

先激活已有环境或设置 `PYTHON_BIN`。在本目录只核对配置和资源：

```bash
bash scripts/train_our_rmsnorm2_4dit_residual_ft_1024_B80_4gpu.sh --preflight
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

## 6. 本目录的精简

| 对象 | 处理及原因 |
|---|---|
| 1016 的实验脚本与 YAML | 已移除（`train_rmsnorm2_*`、`queue_sample_*`、`queue_posttrain_heun50.py` 等）；本目录只保留本方法配置与训练/采样模板，1016 的实验在它自己的目录维护。 |
| `require.txt` | 保留，`scripts/install_requirements.sh` 仍读取它。 |
| `third_party/` | 保留上游许可证、NOTICE 和来源说明，阅读模型时可略过。 |
| `tests/` | 只保留双路径的回归测试（`test_fourdit_dual_path.py`）；1016 的测试套件在 1016 目录运行。 |

输入路径为 `native`、`time_blend` 和本目录新增的 `post_block_time_blend`（代码层保留，旧配置不随目录复制）。`native` 仍保留冻结的 iREPA teacher 和对齐损失，但不构建生成侧 DINO+LoRA。旧 `replace` / `gated_residual` 的实现早已删除，不能用于当前入口。

修改模型后的回归检查：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=.:jit python -m pytest -q tests
```
