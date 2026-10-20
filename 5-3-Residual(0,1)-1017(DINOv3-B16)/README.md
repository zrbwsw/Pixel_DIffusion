# rmsnorm2 单因素消融：Residual(0,1)

本目录对应 `related_tex/exp0926.tex` 中的实验 **`5-3-Residual(0,1)(DINOv3-B/16)`**。
目录名为 `5-3-Residual(0,1)-1017(DINOv3-B16)`，其中 `B16` 避免 `/` 形成子目录。
2026-10-20 将目录前缀从 `4-1-4` 调整为 `5-3`，与实验表格一致；更早的目录名为
`5_rmsnorm2_Residual(0,1)_1017`。运行中的 GPU2 续训仍持有 `4-1-4` 旧绝对路径，
开发机暂留该名称的兼容符号链接，实际文件与后续操作统一使用新目录。
已有 checkpoint、实验 ID、日志和固定源码校验记录保持原样。

## GitHub 源码版

本目录替换仓库中原来的 `5_rmsnorm2_Residual(0,1)_1017/`，与实验表中的 **5-3** 对齐。
本次发布包含模型、训练/采样 Python 入口、配置、固定参考源码、校验清单及许可证；
不包含 checkpoint、预训练权重、日志、采样产物或 `scripts/`。下文涉及 `scripts/`、
tmux 和 GPU 机器的命令是开发机使用说明，相关脚本只在开发机保留。

自行准备 `require.txt` 中的依赖与资源后，可以直接从本目录启动八卡训练：

```bash
export IMAGENET_ROOT=/path/to/imagenet
export PRETRAIN_MODELS_ROOT=/path/to/pretrain_models
export FID_STATS_ROOT=/path/to/fid_stats

# 资源与配方检查；不会启动训练。
WORLD_SIZE=8 python main_rmsnorm2_jit_only.py \
  --config configs/rmsnorm2_residual_0_1_B80_gbs1024_8gpu.yaml \
  --resume '' --preflight

# 从头训练；要求配置中的输出目录尚无实验结果。
torchrun --standalone --nnodes=1 --nproc_per_node=8 \
  main_rmsnorm2_jit_only.py \
  --config configs/rmsnorm2_residual_0_1_B80_gbs1024_8gpu.yaml \
  --resume ''
```

B160 配置的 `resume` 使用本项目下的相对 checkpoint 路径，迁移目录时无需改写。
恢复已有实验时，显式传入正确的 `--resume`，不要复用不同消融的模型或 optimizer 状态。
原版 iREPA 路径和固定 rmsnorm2 路径应在独立 Python 进程中测试，避免模块名称冲突：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=.:jit python -m pytest -q tests
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. python -m pytest -q tests_rmsnorm2
```

依赖未发布启动脚本的两项测试文件留在开发机；缺少真实 checkpoint 或外部对照仓库的测试会跳过。

本目录当前实验以已完成的 `Ours_BaseT_rmsnorm2_Residual_1016` 最佳 B-80
实验为对照：**生成 DINO 路径权重为 0，并跳过其 forward；JiT 路径权重为 1，
保留原有可学习的输入 RMSNorm。** 冻结的 DINOv3 对齐教师仍执行前向，
其预训练最终 LN affine、输入处理、特征提取与对齐损失均沿用该 rmsnorm2 基线。

当前入口是 `main_rmsnorm2_jit_only.py`，配置为
`configs/rmsnorm2_residual_0_1_B80_gbs1024.yaml`，启动脚本为
`scripts/train_rmsnorm2_residual_0_1_B80_0123.sh`。本页后半部分保留导入时的
原版 iREPA 实验说明；其中 `main_basejit_irepa.py` 及相应 20/80 轮脚本
属于另一组实验，不能替代这里的单因素对照。

## 与已完成 rmsnorm2 基线的关系

`rmsnorm2_reference/` 固定仓库提交 `ebaf806ee10bd40c10688890bb4e004c5852fec1`
中的基线实现。`rmsnorm2_reference_manifest.json` 校验 47 个源码/配置/评估索引文件和
213 个共享 DINO 源码文件；启动时由 `rmsnorm2_reference_loader.py` 验证。
该入口直接复用固定版本的模型、teacher、数据处理、loss、优化器、EMA、训练循环
和采样流程，不依赖运行中的兄弟实验目录。

`rmsnorm2_jit_only.py` 的 `RMSNorm2JitOnlyDenoiser` 先按基线顺序构建模型，
使共有参数的初始化和构建阶段随机数状态保持一致；随后在迁移到 GPU、建立
DDP/优化器/EMA 之前，注销生成 DINO、生成 PgT 和 DINO 路径 RMSNorm。
保留原生 JiT patch embedding、可学习的 `jit_pre_blend_norm`、JiT 主干/输出头、
冻结对齐教师，以及用于 JiT block 4 对齐的 Conv3 projector。
代码中的 `dino_student_projector` 是该对齐 projector，仍然保留。

```text
noisy z -> JiT patch embedding -> RMSNorm(gamma_jit) --*1--> APE / JiT / output
         生成 DINO / LoRA / MLS / PgT 分支：不执行 forward，融合权重为 0

clean image -> 基线预处理 -> 冻结 DINOv3 教师 -> 基线对齐目标
JiT block 4 -> 原有 Conv3 projector -> 原有对齐 loss
```

因此模型仍会在构建时读取生成 DINO 权重，但训练和采样不执行其 forward。
原生成 DINO 的训练模式 RoPE 会抽取随机坐标缩放；适配器保留对应的独立随机
抽样调用，使删除生成计算不会额外改变后续 t、noise、label drop 的随机序列。
这只保留随机数消耗，不计算 DINO 特征。未覆盖的随机配置会显式拒绝。

## 训练与 online 评估设置

| 项目 | 当前实验与基线共同设置 |
|---|---|
| 机器 / GPU | agent-gpu，物理 GPU 0、1、2、3 |
| Batch | 每卡 256，全局 1024，accum_iter=1 |
| 学习率 | blr=5e-5，目标 LR=2e-4，warmup 5 epoch，之后 constant |
| 训练长度 | 从零训练 80 epoch，seed=0 |
| 优化器 | AdamW，betas=(0.9,0.95)，eps=1e-8，weight_decay=0 |
| 数据 | ImageNet 256，水平翻转概率 0.5，每 rank 12 workers，pin_memory |
| 模型与对齐 | JiT-B/16，无 DDTHead；冻结 DINOv3-B teacher，block 4 Conv3 对齐 |
| Loss | 基线去噪目标；spatial z-score alpha=0.8，cosine 对齐系数 0.5 |
| 扩散与 label drop | P_mean=-0.8，P_std=0.8，noise_scale=1，t_eps=0.05，label_drop=0.1 |
| EMA | 0.9999 / 0.9996；online 使用第一套 EMA |
| Online 评估 | 每 20 epoch，Heun-50，CFG=1，interval=[0,1]，50K 图，每 rank gen_bsz=64 |

输出目录独立为：

```text
checkpoints/BaseT-Residual(0,1)-NoDinoPre-PgT-blend_rmsnorm2-NoDDTHead-B-80/
```

## agent-gpu 四卡启动

脚本默认使用共享环境
`/mnt/shared-storage-user/datafactory-share/gushishen/agent-z/envs/irepa-py310-cu126-20261015/bin/python`（由项目路径自动解析），并从仓库下的 `pretrain_models`、`fid_stats`、
`nanogen-evals` 读取资源。ImageNet 默认路径为
`/mnt/shared-storage-user/datafactory-share/gushishen/imagenet`。
可通过 `PYTHON_BIN`、`PRETRAIN_MODELS_ROOT`、`FID_STATS_ROOT`、
`NANOGEN_EVALS_ROOT`、`IMAGENET_ROOT` 覆盖资源路径。

在 agent-gpu 上执行：

```bash
# 检查固定源码、训练配置和本地资源。
bash scripts/train_rmsnorm2_residual_0_1_B80_0123.sh --preflight

# 检查 GPU 0–3 空闲后，在独立 tmux 会话启动。
bash scripts/train_rmsnorm2_residual_0_1_B80_0123.sh

# 仅从本实验自己的输出目录显式续训。
bash scripts/train_rmsnorm2_residual_0_1_B80_0123.sh --resume
```

脚本固定 4 个进程及 GPU 0–3，检查目标主机、显存和占用；默认从零训练，
非空输出目录会拒绝重复启动。tmux 会话名为
`train-rmsnorm2-jitonly-B80-0123-b256-gbs1024`。输出目录保存 `train.log`、
`launcher.log`、配置、命令、源码哈希和退出状态，供核查实际运行设置。

### 八卡续训

2026-10-18 增加 agent-gpu 0–7 八卡续训入口。每卡 batch 从 256 调整为
128，保持全局 batch 1024、accum_iter=1、目标 LR2e-4、80 epoch 终点及其余
模型、loss、优化器、EMA、online 参数不变。配置为
`configs/rmsnorm2_residual_0_1_B80_gbs1024_8gpu.yaml`。

```bash
bash scripts/train_rmsnorm2_residual_0_1_B80_01234567_resume.sh --preflight
bash scripts/train_rmsnorm2_residual_0_1_B80_01234567_resume.sh
```

该脚本始终从本实验的 `checkpoint-last.pth` 恢复模型、AdamW 状态和两套 EMA，
缺少任一状态都会拒绝启动。需先停下原四卡进程并确保八卡空闲。继续写入原输出
目录，日志追加；切换前的日志及启动元数据保存在 `run_segments/before_8gpu_*`。
online 每卡 gen_bsz 仍为 64，50K 图、Heun50、CFG1、区间 [0,1] 均不变。
更换 DDP rank 数量会改变数据分配和随机数流，不保证与连续四卡训练逐位相同。

## GPU2 续训至 160 epoch（2026-10-20）

从 `checkpoints/BaseT-Residual(0,1)-NoDinoPre-PgT-blend_rmsnorm2-NoDDTHead-B-80/checkpoint-last.pth`
恢复已完成的 80 epoch，目标 160 epoch。使用 GPU2（H200-0257）全部 8 卡，
每卡 batch128、global batch1024、恒定 LR2e-4；模型、对齐损失、AdamW、
EMA0.9999/0.9996 和在线评估设置沿用原训练。恢复模型、AdamW 和两套 EMA；
配置中的 `start_epoch: 0` 会由 checkpoint 的完成轮数 80 自动覆盖。

配置：`configs/rmsnorm2_residual_0_1_B160_gbs1024_8gpu.yaml`。
入口：`scripts/resume_rmsnorm2_residual_0_1_B160_gpu2_8gpu.sh`（支持 `--preflight`）。
运行会话：`train-rmsnorm2-jitonly-B160-gpu2-8gpu-20261020`。
输出保存于独立目录：

```text
checkpoints/BaseT-Residual(0,1)-NoDinoPre-PgT-blend_rmsnorm2-NoDDTHead-B-160/
```

原 B80 目录保留。训练入口只允许本项目 checkpoints 下、超参数匹配的跨目录续训，
并拒绝覆盖已有 checkpoint 的目标目录；固定的参考模型与训练循环保持不变。

## 新消融：移除 JiT 输入 RMSNorm（2026-10-18）

以已完成的 `BaseT-Residual(0,1)-NoDinoPre-PgT-blend_rmsnorm2-NoDDTHead-B-80`
为对照，新增 `jit_input_rmsnorm: false`。默认值仍为 `true`，因此原有训练和
采样配置的行为保持不变。新实验只删除 **patch embedding 后、APE 前** 的
额外逐 token RMSNorm；JiT block、QK 和输出层内部的归一化全部保留。

```text
原对照：noisy z -> native patch embedding -> learned input RMSNorm -> APE -> JiT -> output
新消融：noisy z -> native patch embedding -------------------------> APE -> JiT -> output
```

两者都跳过生成 DINO/LoRA/MLS/PgT 的 forward，JiT 权重恒为 1；冻结 DINOv3-B
teacher、block-4 Conv3 projector、对齐目标/损失、采样器及所有训练超参数相同。
构建顺序和被跳过生成分支的随机数消耗仍与对照一致。构造后才将输入 norm 换成
`Identity`，所以共有参数初始化不变，优化器和 EMA 不含被删除的 768 个 gamma。
可训练参数由 **136,433,664** 降为 **136,432,896**。

新配置：`configs/jit_only_no_input_rmsnorm_B80_gbs1024_8gpu.yaml`。
新脚本：`scripts/train_jit_only_no_input_rmsnorm_B80_agent_gpu_8gpu.sh`。
使用 agent-gpu 的 GPU 0–7、每卡 batch128、global1024、accum1、LR2e-4、warmup5、
80 epoch、seed0；AdamW、EMA .9999/.9996 均沿用原实验。
online 为每20 epoch、EMA1、Heun50、CFG1、interval `[0,1]`、50K图、每卡 gen_bsz64。
脚本默认**从零训练**，输出独立到：

```text
checkpoints/BaseT-Residual(0,1)-NoDinoPre-PgT-NoInputRMSNorm-NoDDTHead-B-80/
```

```bash
# 只检查配置和资源；不启动训练。
bash scripts/train_jit_only_no_input_rmsnorm_B80_agent_gpu_8gpu.sh --preflight

# 正式启动时要求 agent-gpu 八卡空闲，默认创建独立 tmux 会话。
bash scripts/train_jit_only_no_input_rmsnorm_B80_agent_gpu_8gpu.sh

# 只恢复这个新实验自己的 checkpoint，不能从带 RMSNorm 的旧模型续训。
bash scripts/train_jit_only_no_input_rmsnorm_B80_agent_gpu_8gpu.sh --resume
```

### 与原版 JiT+iREPA forward 的核对

移除额外输入 RMSNorm 后，生成主干的数学结构与原生 JiT 一致。原生 `model_jit.py`
与固定基线版本在去除注释/docstring 后 AST 相同，但本实验沿用的
`_jit_forward_from_tokens` 是另一条执行路径，**不能把完整训练实现称为原版 iREPA
逐位等价复现**：

- APE 和 class tokens：当前 wrapper 显式转换到 patch-token dtype；原版采用
  自身的原位 APE 加法及 class-token 拼接，BF16 下有既有数值差异。
- 对齐分支：仍使用对照的外置 block-4 Conv3 projector、显式 FP32 spatial
  z-score/cosine；原版 projector 在 JiT 内，对齐处理由原版 engine/loss 负责。
- 教师：保留对照的 DINOv3 wrapper（从 clean `[-1,1]` 还原预处理、
  `get_intermediate_layers`）；原版 iREPA 从 uint8 图像经 encoder 的 preprocess /
  `forward_features` 取得目标。构建顺序和 RNG 消耗也沿用对照。
- 文档中的历史 JiT+iREPA DINOv1、bsz512、LR1e-4 实验另有 teacher/训练设置差异。

这些已有差异没有在本次消融中同时修改，以便单独检验输入 RMSNorm 的作用。
相同已训练 backbone 权重下，真实 CUDA 单图测试得到：原生 JiT 与去输入 norm
后的 wrapper 在 **FP32 输出逐位一致**；在 **BF16 最大绝对误差为 0.0291748046875**，
平均绝对误差为 0.00391383795067668。这是指定输入/权重的诊断结果，不是整个
训练流程的误差上界。

验证：64 项 CPU 测试通过；真实 DINOv3 权重和 epoch-80 权重探针的 CUDA
eager 检查通过 944 项断言，覆盖共有初始化、教师、仅删除 input gamma、
原始 tokens、完整训练 loss/梯度、CPU/CUDA RNG、条件与 null-label 采样预测。
对照 oracle 是旧 JiT-only 模型仅将输入 RMSNorm 换成 `Identity`，其 BF16 loss、
全部梯度和预测与新开关实现逐位一致。epoch-80 权重仅用于非零输出头验证，
新训练不会加载它。配置与实际旧 checkpoint 参数也已比对。
原对照前61轮使用4卡、随后8卡续训；本次按要求从第1轮使用8卡，虽保持全局
batch/LR和模型外的训练超参数一致，DDP 样本划分与随机数轨迹仍不保证与旧运行相同。

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest -q tests_rmsnorm2
CUDA_VISIBLE_DEVICES=0 python scripts/verify_no_input_rmsnorm.py \
  --device cuda:0 --output work/no_input_rmsnorm_forward_audit.json
```

本次报告保存在共享目录 `../work/jitonly-no-input-rmsnorm-20261018/forward_audit_cuda.json`
（相对仓库根目录）。agent-gpu 八卡资源预检已通过；本次只准备配置和脚本，
尚未启动新实验，也未进行八卡 DDP 长时间训练验证。

## 单因素验证

```bash
# CPU：共有初始化、RNG、保留的 RMSNorm、生成模块删除等。
CUDA_VISIBLE_DEVICES='' python -m pytest -q tests_rmsnorm2

# 在空闲的单张 CUDA 卡上：真实权重、真实图像的数值对照。
CUDA_VISIBLE_DEVICES=0 python scripts/verify_rmsnorm2_single_factor.py \
  --optimizer-steps 2 --sampling-steps 50 \
  --output work/rmsnorm2_single_factor_parity.json
```

GPU 对照的 oracle 仍执行固定基线的真实生成 DINO/PgT，随后用原有 grounding
分支将融合权重强制为 `(0,1)`；候选模型直接跳过生成分支。验证覆盖共有参数与
teacher、CPU/CUDA RNG、输入 tokens、预测、loss、梯度、AdamW、双 EMA、
CFG 分支和完整 Heun-50，另用非零输出头探针避免零初始化掩盖主干差异。
脚本将实际误差、容差和失败项写入 JSON，不能以“脚本存在”作为验证通过的依据。

该数值对照是单卡小 batch 的确定性 eager 验证；正式四卡训练中的编译、DDP、
显存和连续更新需要另外观察训练日志。单次对照也不证明 80 epoch 的最终 FID
或跨硬件长期轨迹相同。以下历史测试结果只对应导入时的旧入口。

---

# 历史导入说明：原版 iREPA 配对实验

以下内容记录导入版本的结构、脚本和历史验证，不描述上方当前 rmsnorm2
单因素实验的启动状态。旧实验的路径、参数和结果应按各自入口独立理解。

## 原版 JiT+iREPA 基座与可选 rmsnorm2 输入

本项目基于 `Ours_BaseT_rmsnorm2_Residual_1016`（仓库提交 `ebaf806`）创建，
用于先确认 JiT+iREPA baseline，再在同一基座上加入生成空间的 idea。
**默认 baseline 完全不构建、不执行生成 DINO、LoRA、PgT 或额外输入 RMSNorm。**
它与之前保留原封装和生成分支计算的 `(0,1)` 消融目的不同。

### 原版逻辑的保留方式

`jit/` 内 21 个原版 Python 文件直接来自 iREPA 提交
`99ad4ac234efe8de52ce157120f72856e836d09f`，逐字节保留，包括：

- `main_jit.py`：数据变换、初始化顺序、DDP、AdamW、EMA、训练和在线评估调度。
- `engine_jit.py`：uint8 输入的 DINO 预处理、目标空间归一化、BF16 训练步骤。
- `denoiser.py`：label drop、采样 t/noise、flow loss、对齐 loss、CFG 和 Heun。
- `model_jit.py`：原生 patchify、APE 加法、类别 token dtype、JiT blocks、内部 ConvProj。
- `vision_encoder.py`、`projection_loss.py`、`spnorm.py` 及其原版依赖。

`upstream_manifest.json` 固定源码 SHA256；每次启动和 preflight 都校验。
入口 `main_basejit_irepa.py` 只负责配置、资源路径、输出目录保护和模型工厂选择，
随后直接调用原版 `main_jit.main(args)`。原版先构造正则化 DINO，再构造 JiT
及内部 projector 的顺序不变。baseline 模型没有新增参数或 forward hooks。

删除的是**融合前额外添加的输入 RMSNorm**；原版 JiT block、QK 和 final layer
内部的 RMSNorm 必须保留。

### 三种输入模式

```text
baseline:
  noisy z -> 原版 patchify -> 原版 APE / blocks / final layer -> x_pred

jit_rmsnorm:
  noisy z -> 原版 patchify -> RMSNorm(gamma_jit) -> 原版 APE / blocks / final layer
  (DINO, JiT) = (0, 1)，不构建生成 DINO / LoRA / MLS / PgT

rmsnorm2:
  noisy z -> 原版 patchify --------------------> RMSNorm(gamma_jit) --*(1-t)--+
          -> DINOv3+LoRA -> MLS -> PgT(t) ------> RMSNorm(gamma_dino) --*t----+-> 原版 APE / blocks / final layer

各模式共同的 iREPA 正则化（教师由配置选择）:
  clean uint8 -> DINOv1-B/16 或 DINOv3-B/16 -> 原版 spatial z-score(alpha=.8) -> 对齐目标
  JiT block 4 -> JiT 内部 Conv3 projector -> 原版 negative cosine，系数 .5
```

idea 只通过 patch embedder 的输出 hook 接入；双路模式的 t 从同一次 JiT.forward 获取，
不重写 JiT trunk、Denoiser.forward、loss 或 sampler。实现见 `generation_input.py`。
生成分支保持 rmsnorm2 的设置：DINOv3-B/16、LoRA r/alpha=64、dropout=0、
训练 patch embedding、MLS 层 `[2,5,8,11]`、raw noisy z 无图像预处理、
最终 LN 预训练 affine 保留并冻结、PgT、两个独立可学习 RMSNorm gamma。

**生成 DINO 和 iREPA 正则化 DINO 是两个职责不同的编码器。**
原有三套配置的正则化侧均为原版 DINOv1，保留其最终 LN gamma/beta；
不采用生成 DINO 的 LoRA、MLS 或输出处理。
旧最佳实验使用 DINOv3 正则化，本项目的 idea 配置是受控的新实验，
不能直接等同于已有 FID 7.44 的 checkpoint。

### 配置与对照顺序

| 配置（configs/ 下） | 输入 | Epoch | 全局 batch / LR | 在线评估 |
|---|---|---:|---|---|
| `basejit_irepa_dinov1_b20_gbs512.yaml` | baseline | 20 | 512 / 1e-4 | Heun50，CFG2.9，区间(.1,1)，gen_bsz256 |
| `basejit_irepa_dinov1_b80_gbs1024.yaml` | baseline | 80 | 1024 / 2e-4 | Heun50，CFG1，gen_bsz64 |
| `rmsnorm2_basejit_irepa_dinov1_b80_gbs1024.yaml` | rmsnorm2 | 80 | 1024 / 2e-4 | 与上一行相同 |
| `basejit_irepa_dinov3_b20_gbs512.yaml` | baseline | 20 | 512 / 1e-4 | Heun50，CFG2.9，区间(.1,1)，gen_bsz256 |
| `jit_rmsnorm_basejit_irepa_dinov3_b20_gbs512.yaml` | jit_rmsnorm | 20 | 512 / 1e-4 | 与上一行相同 |
| `basejit_irepa_dinov3_b80_gbs1024.yaml` | baseline | 80 | 1024 / 2e-4 | Heun50，CFG1，gen_bsz64 |
| `jit_rmsnorm_basejit_irepa_dinov3_b80_gbs1024.yaml` | jit_rmsnorm | 80 | 1024 / 2e-4 | 与上一行相同 |

均为 ImageNet256、JiT-B/16、AdamW(.9,.95)、weight decay=0、warmup5、
blr5e-5、常数 LR、无梯度累积、EMA .9999/.9996、每20 epoch在线评估50K图。
在线评估使用第一套 EMA。历史512配置对应 `../jit_ckpts/logs/jit_dino.txt`。
两套1024配置仅模式、实验名、输出目录不同，以隔离生成输入模块的影响。

新增 DINOv3 配对实验优先排除生成分支：两组都完全不构建生成 DINO，
只比较是否在 JiT patchify 后、APE 前加一个带可学习 gamma 的逐 token RMSNorm。
复用 `jit/util/model_util.py` 的实现，eps=1e-6，gamma 初始为1，JiT-B仅新增768个参数。
JiT 分支权重恒为1，不乘 `(1-t)`，不在融合后或 APE 后归一化。
正则化教师为原版加载路径的 DINOv3-B/16，使用干净图像、ImageNet 预处理、
单层最终输出、预训练最终 LN affine，不使用 LoRA 或 MLS；由原版 engine
在 eval/no_grad 下抽取目标，不加入优化器。对齐仍在 block4，Conv3、cosine系数.5、
spatial z-score alpha=.8、cls_token_weight=.2，其他训练超参数与对应历史配置一致。
不能直接将 DINOv3 的新结果与 DINOv1 历史结果之差归因于输入 RMSNorm。

先比较历史512配置与旧 baseline，再比较1024 baseline与1024 idea。
同样 epoch 下，1024 batch 的更新步数约为512的一半；不能将这个变化归因于模型实现。

### 使用

使用 Python3.10 和 `require.txt` 中依赖。资源与权重不上传 GitHub。
本机可显式设置：

```bash
export PYTHON_BIN=/data0/zhangruobei/anaconda3/envs/irepa/bin/python
export PRETRAIN_MODELS_ROOT=/data0/pretrain_models
export IMAGENET_ROOT=/data0/datasets/imagenet
```

H200 可将上述路径替换成对应环境；也支持 `FID_STATS_ROOT`。
原版 iREPA 的 `data_path` 是包含 `imagenet` 的父目录，入口从 `IMAGENET_ROOT`
自动转换；显式传 `--data_path` 时遵循原版含义。

```bash
# 检查历史baseline的配置和本地资源（不构建模型、不训练）。
GPUS=0,1,2,3 bash scripts/train_basejit_irepa.sh --preflight

# 历史512 baseline；请在计划使用的GPU有容量时执行。
GPUS=0,1,2,3 bash scripts/train_basejit_irepa.sh

# 1024 baseline，四卡每卡256；也可八卡每卡128。
GPUS=0,1,2,3 bash scripts/train_basejit_irepa.sh \
  configs/basejit_irepa_dinov1_b80_gbs1024.yaml --batch_size 256

# 在同一基座上启用生成输入idea。
GPUS=0,1,2,3,4,5,6,7 bash scripts/train_basejit_irepa.sh \
  configs/rmsnorm2_basejit_irepa_dinov1_b80_gbs1024.yaml

# 新消融：只加 JiT 输入 RMSNorm，DINOv3-B/16 正则化；默认20轮、global512。
GPUS=0,1,2,3 bash scripts/train_jit_rmsnorm_irepa.sh --preflight
GPUS=0,1,2,3 bash scripts/train_jit_rmsnorm_irepa.sh

# 本机八卡：global512 = 8x64，LR1e-4；其余沿用上面的20轮配置。
bash scripts/train_jit_rmsnorm_irepa_8gpu_bsz512.sh --preflight
bash scripts/train_jit_rmsnorm_irepa_8gpu_bsz512.sh

# 对照组：同一 DINOv3 教师，但不加输入 RMSNorm。
GPUS=0,1,2,3 bash scripts/train_basejit_irepa.sh \
  configs/basejit_irepa_dinov3_b20_gbs512.yaml

# 可选80轮配对设置：八卡每卡128，global1024、LR2e-4。
GPUS=0,1,2,3,4,5,6,7 bash scripts/train_jit_rmsnorm_irepa.sh \
  configs/jit_rmsnorm_basejit_irepa_dinov3_b80_gbs1024.yaml
```

启动前检查 `每卡batch × WORLD_SIZE == expected_global_batch_size`，不自动缩放实验。
配置与 CLI 的明确覆盖均生效；相对 output 路径固定相对于本项目。
默认从零开始，非空输出目录会拒绝启动；续训须显式指定同目录 `--resume`，
并校验关键模型/训练设置。旧 rmsnorm2 或 `(0,1)` 项目的 checkpoint 不兼容。

本机八卡脚本默认使用 `irepa` 环境的 Python，沿用20轮配置的warmup5、
AdamW(.9,.95)、weight decay=0、EMA .9999/.9996、DINOv3-B/16正则化和
Heun50/CFG2.9/[.1,1]/50K评估，不切换到global1024/LR2e-4的80轮配置。
输出目录为 `checkpoints/jit-rmsnorm-basejit-irepa-dinov3-B-20-gbs512-gpus8/`。
`jit_rmsnorm` 不构建生成DINO、LoRA、MLS、PgT及其RMSNorm，因此训练和采样
都不会执行生成DINO分支的forward；仅干净图像的正则化教师仍需前向。
八卡与四卡保持有效batch/LR相同，但分布式样本划分和随机数序列不保证逐步一致。

官方 DINOv1 源码固定在 `SSL_code/dino/`，入口为原版 torch.hub 建立项目内缓存链接，
使用现有 `dino_vitbase16_pretrain.pth`，不改 encoder 加载/预处理逻辑。
DINOv3 教师使用项目内 `SSL_code/dinov3/` 和 `PRETRAIN_MODELS_ROOT` 下的
`dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth`；可用 `DINOV3_REPO_DIR`、
`DINOV3_CKPT_DIR` 分别覆盖源码、权重目录。DINOv3 配置不依赖 DINOv1 权重。
运行缓存、输出、数据集、权重均被 gitignore 排除。

### 验证与边界

```bash
python scripts/verify_upstream.py
CUDA_VISIBLE_DEVICES='' PYTHONPATH=.:jit python -m pytest -q tests
CUDA_VISIBLE_DEVICES=4 python scripts/verify_baseline_parity.py \
  --mode cuda --execution compiled --hub-dir .cache/torch/hub \
  --sampling-steps 50 --output tests_results/baseline_parity_cuda
```

2026-10-17 的完整 GPU 对照已通过：真实预训练 DINOv1 + JiT-B/16、256px、
BF16、torch.compile、batch1，33个采集张量逐位一致，最大绝对误差0。
初始化、随机数状态、native及非零head探针的梯度、AdamW、双EMA、CFG边界
和完整 Heun50 采样均一致。非零head探针仅在测试进程使用，避免零初始化输出
掩盖主干差异。详细记录见 [验证说明](readme/baseline_contract_and_verification.md)。
idea模式也通过真实双DINO、单rank DDP、原版训练循环的两步forward/backward检查，
全部可训练参数梯度存在且有限；CPU测试10通过、2项GPU专属测试跳过。

数值验证对照的是上述当前原版源码，使用相同设备/软件和确定性测试设置。
历史日志没有记录源码 revision，不能仅凭日志证明历史训练源码逐字相同；
也没有由单次 forward 证明长期训练、不同GPU或FID完全重现。
上述基座验证完成时尚未启动20/80 epoch正式训练；JiT-only实验的启动状态见下文。

新增 JiT-only RMSNorm 的 CPU 回归：20项通过、2项GPU专属测试跳过。
检查包含原版参数初始化/RNG保持不变、唯一新增gamma、FP32/BF16归一化、
APE前插入、固定权重不受t或null label影响、原版去噪/对齐loss反传、AdamW、
双EMA、权重重载、CFG/Heun采样，以及DINOv3配对配置仅改变模式与输出标识。
20轮四卡与80轮八卡配置均通过preflight；24项上游文件哈希校验保持通过。
本机20轮八卡脚本也通过preflight，并验证其只改变每卡batch和输出标识，
有效batch/LR及其余训练配置不变；训练/采样测试禁止调用生成分支构建或forward。
另用本机真实DINOv3-B/16预训练权重完成CPU教师前向：256px输出为
`[1,256,768]` patch tokens，eval模式、特征有限、最终LN gamma/beta与权重逐位一致。
JiT-only模式已于2026-10-17在本机GPU 0--7启动正式训练，global512=8x64、
目标LR1e-4、20轮，已通过CUDA编译及首批forward/backward并进入第二轮。
这不代表完整20轮训练或最终生成质量已验证。日志在本机
`checkpoints/logs/jit_rmsnorm_dinov3_b20_gbs512_8gpu.train.log`，不上传GitHub。
