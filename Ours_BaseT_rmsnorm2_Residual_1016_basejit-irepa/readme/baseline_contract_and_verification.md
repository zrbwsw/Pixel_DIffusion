# baseline 契约及验证记录（2026-10-17）

## 目标与来源

本项目要消除旧封装引入的 JiT+iREPA 差异。它并非之前的 fixed `(0,1)` 实验：
后者保留生成 DINO/PgT 计算、旧 trunk dtype、外部 projector 和旧 loss。
本项目 baseline 则直接构造原版 Denoiser，不构造生成路径，甚至不导入其模块。

- rmsnorm2 源版本：Pixel_Diffusion `ebaf806`。
- iREPA：`99ad4ac234efe8de52ce157120f72856e836d09f`，原版21个 Python 文件逐字节相同。
- DINOv1 官方代码：`7c446df5b9f45747937fb0d72314eb9f7b66930a`。
- DINOv1 权重 SHA256：`bf34ad0f424b9029b593e8dc3ed553bf26e88bcba0d32bf3e62a6209cb64c85e`。
- 历史超参数：`../jit_ckpts/logs/jit_dino.txt`。日志没有源码 revision 字段。

`upstream_manifest.json` 包含21个原版文件和3个官方DINO文件的hash。
两个额外辅助文件 `jit/config_utils.py`、`jit/resource_paths.py` 继承自 rmsnorm2，
只用于新入口配置和资源路径，不改变原版 main/engine 的训练数学。

## 明确保留的计算顺序

| 环节 | 新 baseline 的行为 |
|---|---|
| 初始化 | 原版 main 设 seed+rank；先构造 DINOv1，再构造 JiT 及内置 ConvProj |
| 输入与增强 | 原版 ImageFolder、center crop、RandomHorizontalFlip(.5)、PILToTensor |
| 正则化目标 | uint8 /255 → ImageNet Normalize → bicubic256 → DINO最后LN → patch tokens →原版 spatial z-score |
| 生成输入 | uint8 → FP32 /255 → `x*2-1`；Denoiser内依次drop labels、sample t、sample noise |
| patch/APE | 原版 patchify，原样 `x += self.pos_embed` |
| 类别 token | 原版 FP32 class token，无额外BF16强转；concat后残差流为FP32 |
| projector | JiT构造时建立内置Conv3，在第4个block后执行 |
| loss | 原版分别F.normalize的negative cosine；系数.5；不额外cast FP32或替换API |
| 优化 | 原版weight-decay分组与参数顺序、AdamW(.9,.95)、无累积、原版双EMA |
| 采样 | 原版batched CFG、区间边界、Heun运算顺序及末步Euler |

去掉的是新增的融合前RMSNorm。原版 block、attention QK、final layer 的RMSNorm保留。
新增idea的两路输入RMSNorm在 `rmsnorm2` 模式才创建，gamma相互独立。

## GPU 数值对照：通过

验证脚本：`scripts/verify_baseline_parity.py`。
原版与新baseline在两个独立子进程顺序执行，避免Python同名模块或模型状态串用。

- GPU4，NVIDIA GeForce RTX4090；PyTorch2.7.0+cu126，CUDA12.6。
- 完整JiT-B/16、256×256、真实预训练DINOv1、BF16 autocast、torch.compile。
- 每侧batch1，候选峰值分配显存约4.00GiB。
- 确定性cuDNN设置，benchmark=False，用于同设备逐位比较。
- 两个优化步骤：原生零head初始化；随后测试专用非零head/adaLN探针。
- 非零探针确保首/末block、patchify和projector真正得到非零梯度，避免零输出掩盖错误。
- 优化测试使用非零历史峰值LR1e-4；warmup学习率另行逐点比较。
- CFG边界、单步Heun及完整50步轨迹全部比较。

结果：**零差异，33个采集张量逐位一致，最大绝对误差0**。
所有初始化参数、DINO权重、CPU/CUDA RNG、两步骤梯度、AdamW状态、更新参数及双EMA
均按完整张量SHA256比较，一致。

| 探针 | Denoise loss | Negative cosine | Total loss |
|---|---:|---:|---:|
| 原生初始化 | 0.7754241824 | 0.0052801077 | 0.7780642509 |
| 非零head探针 | 0.5007874966 | -0.1691669226 | 0.4162040353 |

以上为固定合成输入的测试值，两侧相同；不是训练集或历史run的loss结果。

| 激活位置 | 实际dtype |
|---|---|
| patchify输出 | BF16 |
| block4输出 | BF16 |
| block5输入（加入类别token后） | FP32 |

本地完整证据保留在 `tests_results/baseline_parity_cuda/`：
`reference.json`、`candidate.json`、`comparison.json`、激活张量和执行日志。
这些运行产物不上传，验证脚本与本说明上传GitHub。

## idea 与原版训练循环连通：通过

完整模型另通过 `scripts/smoke_rmsnorm2.py`：GPU4、单rank DDP、BF16、
torch.compile、真实DINOv1目标与真实DINOv3+LoRA+MLS+PgT输入，
直接使用原版 `engine_jit.train_one_epoch` 处理两个合成uint8 batch（每批1张）。
优化器执行2步，所有可训练参数梯度存在且有限，无缺失梯度；
生成LoRA、生成patch embedding、PgT、双RMSNorm、JiT和内部对齐projector
各自均有非零梯度，hook的临时t引用正确清空，AdamW与双EMA检查通过。
这项检查验证新idea的训练链路，不是对旧best模型的等价声明。
报告在本地 `tests_results/rmsnorm2_smoke.json`。

CPU测试集结果为 **10 passed、2 skipped**；两个skip为CPU模式下的GPU专属测试。
真实GPU验证结果如上单独记录。

## 配置对齐与后续实验

历史复现配置：4×128=512、20epoch、blr5e-5→LR1e-4、warmup5、
DINOv1-B/16、block4、Conv3、cosine系数.5、z-score alpha.8；
在线Heun50、CFG2.9、区间(.1,1)、gen_bsz256、50K图、EMA.9999。

1024 baseline与1024 idea配置只在输入模式、实验名、输出目录上不同。
二者均保持DINOv1正则化侧，80epoch、blr5e-5→LR2e-4、在线Heun50/CFG1。
旧FID7.44实验使用DINOv3正则化和旧封装；不应将本项目idea结果视作直接续训。

正式20/80epoch训练尚未启动。源码和短程数值一致不等于已经复现历史FID：
历史源码未标revision，软件/GPU/数据与增强状态也可能不同；独立进程数值对照
不覆盖完整DataLoader随机序列、多卡归约或长期训练。

## 可复核命令

先运行README中的preflight建立资源缓存，再执行：

```bash
python scripts/verify_upstream.py
CUDA_VISIBLE_DEVICES='' PYTHONPATH=.:jit python -m pytest -q tests
CUDA_VISIBLE_DEVICES=4 python scripts/verify_baseline_parity.py \
  --mode cuda --execution compiled --hub-dir .cache/torch/hub \
  --sampling-steps 50 --output tests_results/baseline_parity_cuda

# 可选idea真实模型两步烟测；不会训练ImageNet或写正式checkpoint。
CUDA_VISIBLE_DEVICES=4 torchrun --standalone --nproc-per-node=1 \
  scripts/smoke_rmsnorm2.py --output tests_results/rmsnorm2_smoke.json
```
