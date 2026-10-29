# 24层 JiT-XL/16 + DDTHead + DINOv3-L/16

本目录基于 `Ours_BaseT_rmsnorm2_Residual_1016_XL_1027`，保留运行源码、公共工具及一份 a=0.1 配置/训练脚本，不复制 checkpoint、测试或另一份旧实验配置。

## 模型改动

- `jit/model_jit.py` 的 `JiT-XL/16` 为 `hidden_size=1152、depth=24、num_heads=16、patch_size=16、bottleneck_dim=256`；`in_context_start=8` 与 JiT-L 一致。
- 启用 `JiTDDTHead`：`hidden_size=2048、depth=2、num_heads=16、bottleneck_dim=128`。DDTHead 负责像素预测，JiT 原生 final layer 保留在 state_dict 中但冻结。
- iREPA 对齐使用 JiT 第8层，与 JiT-L 默认 `encoder_depth=8` 一致；ConvProj 仍将 1152 维投影到 DINO teacher 的 1024 维。
- DINO student/teacher、DinoYCond、LoRA rank64、PgT、两路 RMSNorm 和其余训练设置不变。

## 时间融合函数

`blend_time.py` 与 `normft_1028` 一致，支持 `linear/smooth/smoother/sigmoid`。当前默认实验使用 `dino_time_blend_transform: sigmoid`，即端点归一化的 **SigmoidNormBlend**：`f(0)=0、f(1)=1`，`K=20,a=0.1`。实验名已从 `SigmoidBlendK20` 改为 `SigmoidNormBlendK20`，避免与旧的未归一化 sigmoid 混淆。多项式函数不需要 key/a；归一化 sigmoid 的 `a≠0.5` 时，`f(a)` 不严格等于 0.5。

## 配置与训练

配置为 `configs/reverse_sigmoid_blend_k20_a0p1_dino_ycond_bottle256_XL_B80.yaml`。训练脚本默认 GPU1 八卡，每卡 batch64、累积2步、全局 batch1024、有效 LR `1e-4`、80 epoch；文件名、实验名和输出目录均包含 `bottle256`。

```bash
bash scripts/train_rmsnorm2_reverse_sigmoidnorm_blend_k20_a0p1_dino_ycond_bottle256_XL_B80_8gpu.sh --dry-run
bash scripts/train_rmsnorm2_reverse_sigmoidnorm_blend_k20_a0p1_dino_ycond_bottle256_XL_B80_8gpu.sh --preflight
```

## 参数精确计数

在 GPU 上按当前源码真实实例化计数，不使用估算公式：

| 口径 | 参数量 |
|---|---:|
| JiT-XL/16（24层，bottleneck=256） | 580,964,352 |
| DDTHead（2048维、2层） | 163,758,676 |
| **JiT-XL + DDTHead 总参数** | **744,723,028** |
| **JiT-XL + DDTHead 可训练参数** | **740,884,948** |
| 完整方法可训练参数（含 LoRA/PgT/ConvProj 等） | 806,515,172 |
| 完整方法总参数（不含 frozen teacher） | 1,112,719,972 |
| frozen DINOv3-L teacher | 303,154,176 |
| 训练时总参数（含 teacher） | 1,415,874,148 |
| 采样侧有效参数（不含 teacher、对齐投影和原生 final layer） | 1,098,558,948 |

“可训练”已排除被 DDTHead 替代的 JiT 原生 final layer（3,543,168 参数）及其他冻结参数。真实 DDTHead 前向/反向、完整方法的第8层对齐与训练损失反向均已验证；尚未启动正式训练。
