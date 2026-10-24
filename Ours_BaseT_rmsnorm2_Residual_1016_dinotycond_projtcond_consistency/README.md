# DinoTYCond + PgT / JiT 条件一致性

本目录从 `Ours_BaseT_rmsnorm2_Residual_1016` 复制，只维护 **a0.3/k20 DinoTYCond + PgT 条件一致性**实验。原始 1016 目录及其历史配置保持不变。

## 模型改动

开启 `dino_jit_condition_consistency` 后：

- DINO patch embedding 后加入 `c=t_emb+y_emb`，与 JiT blocks/final layer 使用同一个 `net.t_embedder/net.y_embedder` 和相同的原始 t、drop 后 y。
- PgT 的 adaLN 直接使用 JiT 的 `t_emb`，不再使用独立的 `TimestepConditioner`。
- DINO 原来的 `cond_t_embedder/cond_y_embedder` 和 PgT 原来的 `t_embedder` 置为 `None`。只有 `net` 保存并训练这套参数，避免 checkpoint/EMA 中出现模块别名。
- 其余模型与训练配方不变：NoDinoPre、LoRA、MLS、PgT 主体、两路独立 RMSNorm、reverse sigmoid 融合、JiT 第4层 ConvProj、frozen iREPA teacher 和去噪损失。

```text
原始 t、drop 后 y → net.t_embedder / net.y_embedder → t_emb、y_emb，c=t_emb+y_emb
带噪 z_t
  ├─ JiT patch embed → JiT RMSNorm → ×t ───────────────────────────────┐
  └─ DINO patch_embed + c → DINO blocks+LoRA → 各选中层 final LN → MLS
                         → PgT(adaLN=t_emb) → DINO RMSNorm → ×(1-t) ──┤
                                                                  ↓ 相加
                                              加位置编码 → JiT blocks(c)
                                                ├─ 第4层 → ConvProj → 对齐 loss
                                                └─ final layer(c) → 干净图像预测
干净 x → ImageNet 标准化 → frozen DINO teacher → normalized patch tokens → 对齐目标
```

MLS 仍是每个选中层分别过 final LN 后再聚合。采样/CFG 与训练共用相同的条件来源；sigmoid 只改变融合权重，不改变传入 DINO、PgT 或 JiT 的原始 t。

## 配置与入口

| 用途 | 文件 |
|---|---|
| 唯一实验配置 | `configs/reverse_sigmoid_blend_k20_a0p3_dino_tycond_projtcond_consistency_B80.yaml` |
| 当前八卡训练入口 | `scripts/train_rmsnorm2_reverse_sigmoid_blend_k20_a0p3_dino_tycond_projtcond_consistency_B80_8gpu.sh` |
| 通用训练 / 采样入口 | `scripts/train_portable.sh`、`scripts/sample_portable.sh`、`scripts/sample_ckpt_8gpu.sh` |
| 公共环境与预检 | `scripts/train_env.sh`、`scripts/portable_env.sh`、`scripts/check_runtime.py`、`scripts/check_training_launch.py` |

当前配方为 GPU2 八卡，每卡 batch 128、全局 1024、有效学习率 `2e-4`、80 epoch；融合为 reverse `sigmoid(20*(t-0.3))`，在线评测为 Heun50 / CFG1。详细用法见 `scripts/README.md`。

## 源码阅读顺序

1. `main_dino_jit.py`：声明 `dino_jit_condition_consistency` 和 `dino_projector_condition`（本目录为 `t`）。
2. `dino_jit_denoiser.py`：计算并传递共享 `c` 和 `t_emb`，执行融合、训练与采样。
3. `dino_v3_lora.py`：接收 `shared_cond [B,D]`，在 DINO patch embedding 后广播注入。
4. `pixeldit2_projector.py`：接收外部 `condition [B,D]` 作为 PgT adaLN 条件。
5. `irepa_teacher.py`：独立 frozen teacher，不读取 student 的条件开关。

`jit/`、`SSL_code/dinov3/` 和 `third_party/` 保留运行所需源码与许可证。`tests/` 只保留当前模型、条件一致性及保留入口的回归测试。

## Checkpoint 与输出

训练写入 `checkpoints/<实验名>/`，其中 `config_resolved.yaml` 记录最终 CLI/YAML 参数。后续采样优先使用该文件。

本实验移除了 DINO/PgT 的独立条件嵌入参数，参数集合与旧 DinoTYCond checkpoint 不同，不应跨配置直接续训；本实验自己的 checkpoint 可正常续训和进行 EMA 采样。清理旧脚本和配置不影响已有输出、日志或 checkpoint。

## 验证

```bash
PYTHONPATH=.:jit python -m pytest tests -q
bash scripts/train_rmsnorm2_reverse_sigmoid_blend_k20_a0p3_dino_tycond_projtcond_consistency_B80_8gpu.sh --dry-run
bash scripts/train_rmsnorm2_reverse_sigmoid_blend_k20_a0p3_dino_tycond_projtcond_consistency_B80_8gpu.sh --preflight
```
