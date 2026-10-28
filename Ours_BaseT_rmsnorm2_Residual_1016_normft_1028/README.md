# DINOv3 + LoRA + JiT：归一化时间融合函数

本目录从 `Ours_BaseT_rmsnorm2_Residual_1016` 复制运行所需源码，不复制历史实验配置、checkpoint 或采样产物。当前改动集中在融合权重函数 `f(t)`，并保留归一化 sigmoid 与 smoother 两套实验配置。

## 四种可选函数

通过 `--dino_time_blend_transform` 选择：

| 选项 | 公式 |
|---|---|
| `linear` | `f(t)=t` |
| `smooth` | `f(t)=3t²-2t³` |
| `smoother` | `f(t)=6t⁵-15t⁴+10t³` |
| `sigmoid` | `f(t)=(σ(k(t-a))-σ(-ka)) / (σ(k(1-a))-σ(-ka))` |

本版本的 `sigmoid` 是端点归一化版本，保证 `f(0)=0`、`f(1)=1`。`--dino_time_blend_key` 和 `--dino_time_blend_a` 只给 sigmoid 使用，要求 `k>0`、`0<a<1`；两个多项式函数不需要额外参数。

注意：`a` 是归一化前 sigmoid 的对称中心。若 `a≠0.5`，端点归一化会让 `f(a)` 相对 0.5 有很小偏移；当前常用 `a=0.3,k=20` 时约为 0.4975。

## 生效位置

训练和采样共用 `dino_jit_denoiser.py::_compose_jit_input_tokens()`。`f(t)` 只改变 DINO/JiT token 的融合权重；加噪、velocity 目标、PgT/JiT 时间条件、CFG interval 和采样步仍使用原始 `t`。

`dino_time_blend_reverse` 语义不变：

- `false`：`f(t)*DINO + (1-f(t))*JiT`
- `true`：`(1-f(t))*DINO + f(t)*JiT`

核心文件：

- `blend_time.py`：四种函数、低精度计算和参数校验。
- `main_dino_jit.py`：命令行超参数与合法选项。
- `dino_jit_denoiser.py`：训练/采样共用融合入口。
- 其余 DINO、JiT、iREPA、PgT 和损失逻辑保持原 1016 实现。

## 验证

```bash
PYTHONPATH=.:jit python -m pytest tests/test_blend_time.py -q
```

当前实验配置：

- 归一化 sigmoid：`configs/reverse_sigmoid_blend_k20_a0p1_dino_ycond_norm_B80.yaml` + `scripts/train_rmsnorm2_reverse_sigmoid_blend_k20_a0p1_dino_ycond_norm_B80_8gpu.sh`；K20、a=0.1。
- smoother：`configs/reverse_smoother_blend_dino_ycond_B80.yaml` + `scripts/train_rmsnorm2_reverse_smoother_blend_dino_ycond_B80_8gpu.sh`；不需要 key/a。

两者都沿用 DinoYCond、全局 batch1024、有效 LR `2e-4` 和 GPU2 八卡。本目录不包含历史实验配置、checkpoint 或采样产物。选择 `sigmoid` 时仍应明确写出 `dino_time_blend_key` 和 `dino_time_blend_a`。
