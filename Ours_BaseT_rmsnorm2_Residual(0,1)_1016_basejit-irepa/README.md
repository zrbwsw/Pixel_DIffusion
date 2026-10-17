# 原版 JiT+iREPA 基座与可选 rmsnorm2 输入

本项目基于 `Ours_BaseT_rmsnorm2_Residual_1016`（仓库提交 `ebaf806`）创建，
用于先确认 JiT+iREPA baseline，再在同一基座上加入生成空间的 idea。
**默认 baseline 完全不构建、不执行生成 DINO、LoRA、PgT 或额外输入 RMSNorm。**
它与之前保留原封装和生成分支计算的 `(0,1)` 消融目的不同。

## 原版逻辑的保留方式

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

## 三种输入模式

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

## 配置与对照顺序

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

## 使用

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

官方 DINOv1 源码固定在 `SSL_code/dino/`，入口为原版 torch.hub 建立项目内缓存链接，
使用现有 `dino_vitbase16_pretrain.pth`，不改 encoder 加载/预处理逻辑。
DINOv3 教师使用项目内 `SSL_code/dinov3/` 和 `PRETRAIN_MODELS_ROOT` 下的
`dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth`；可用 `DINOV3_REPO_DIR`、
`DINOV3_CKPT_DIR` 分别覆盖源码、权重目录。DINOv3 配置不依赖 DINOv1 权重。
运行缓存、输出、数据集、权重均被 gitignore 排除。

## 验证与边界

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
本次完成代码和验证，尚未启动20/80 epoch正式训练。

新增 JiT-only RMSNorm 的 CPU 回归：19项通过、2项GPU专属测试跳过。
检查包含原版参数初始化/RNG保持不变、唯一新增gamma、FP32/BF16归一化、
APE前插入、固定权重不受t或null label影响、原版去噪/对齐loss反传、AdamW、
双EMA、权重重载、CFG/Heun采样，以及DINOv3配对配置仅改变模式与输出标识。
20轮四卡与80轮八卡配置均通过preflight；24项上游文件哈希校验保持通过。
另用本机真实DINOv3-B/16预训练权重完成CPU教师前向：256px输出为
`[1,256,768]` patch tokens，eval模式、特征有限、最终LN gamma/beta与权重逐位一致。
新增模式尚未进行CUDA训练/编译验证，未启动正式训练，也未打断本机已有任务。
