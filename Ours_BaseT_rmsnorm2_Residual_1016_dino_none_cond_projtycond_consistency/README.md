# DinoNoneCond + ProjTYCond / JiT 条件一致性

本目录只维护 **a0.3/k20 DinoNoneCond + ProjTYCond 条件一致性**实验。它从首版 `Ours_BaseT_rmsnorm2_Residual_1016_dinotycond_projtcond_consistency` 复制，只改变 DINO 与 PgT 的 t/y 注入方式，其余训练配方保持不变。

## 模型改动

- **DINO patch 后不加任何条件**：`dino_student_time_cond: false`、`dino_student_label_cond: false`。DINO blocks+LoRA、final LN 和 MLS 仍正常执行。
- **PgT 使用完整 JiT 条件**：`dino_projector_condition: ty`，adaLN 接收 `c=t_emb+y_emb`，与 JiT blocks/final layer 使用同一个 `net.t_embedder/net.y_embedder`、原始 t 和 drop 后 y。
- PgT 原来的独立 `TimestepConditioner` 置为 `None`；只有 `net` 保存并训练 t/y 嵌入参数，checkpoint/EMA 中没有共享模块别名。
- 其他设置不变：NoDinoPre、PgT 主体、两路独立 RMSNorm、reverse sigmoid 融合、JiT 第4层 ConvProj、frozen iREPA teacher 和去噪损失。

```text
原始 t、drop 后 y → net.t_embedder / net.y_embedder → c=t_emb+y_emb
带噪 z_t
  ├─ JiT patch embed → JiT RMSNorm → ×t ───────────────────────────────┐
  └─ DINO patch_embed（不加 t/y）→ DINO blocks+LoRA → final LN / MLS
                         → PgT(adaLN=c) → DINO RMSNorm → ×(1-t) ───────┤
                                                                  ↓ 相加
                                              加位置编码 → JiT blocks(c)
                                                ├─ 第4层 → ConvProj → 对齐 loss
                                                └─ final layer(c) → 干净图像预测
干净 x → ImageNet 标准化 → frozen DINO teacher → normalized patch tokens → 对齐目标
```

训练和 CFG 采样都使用对应的标签；sigmoid 只改变融合权重，不改变传入 PgT 或 JiT 的原始 t。

## 配置与入口

| 用途 | 文件 |
|---|---|
| 唯一实验配置 | `configs/reverse_sigmoid_blend_k20_a0p3_dino_none_cond_projtycond_consistency_B80.yaml` |
| 当前八卡训练入口 | `scripts/train_rmsnorm2_reverse_sigmoid_blend_k20_a0p3_dino_none_cond_projtycond_consistency_B80_8gpu.sh` |
| 通用训练 / 采样入口 | `scripts/train_portable.sh`、`scripts/sample_portable.sh`、`scripts/sample_ckpt_8gpu.sh` |
| 公共环境与预检 | `scripts/train_env.sh`、`scripts/portable_env.sh`、`scripts/check_runtime.py`、`scripts/check_training_launch.py` |

当前配方为 GPU2 八卡，每卡 batch 128、全局 1024、有效学习率 `2e-4`、80 epoch；融合为 reverse `sigmoid(20*(t-0.3))`，在线评测为 Heun50 / CFG1。详细用法见 `scripts/README.md`。

## 源码阅读顺序

1. `main_dino_jit.py`：`dino_jit_condition_consistency` 和 `dino_projector_condition` 参数。
2. `dino_jit_denoiser.py`：DINO 零条件注入，向 PgT 传递共享 t 或 t+y 条件。
3. `pixeldit2_projector.py`：接收外部 `condition [B,D]`，形成 `[B,1,D]` adaLN 条件。
4. `dino_v3_lora.py`：保留可选条件接口；本配置两个 student 条件开关均关闭。
5. `irepa_teacher.py`：独立 frozen teacher，不读取生成侧条件开关。

## Checkpoint 与输出

训练写入 `checkpoints/<实验名>/`，采样优先读取其中的 `config_resolved.yaml`。不要跨不同条件配置直接续训；使用本实验自己的 checkpoint 续训和 EMA 采样。`jit/`、`SSL_code/dinov3/` 和 `third_party/` 保留运行所需源码与许可证。

## 验证

```bash
PYTHONPATH=.:jit python -m pytest tests -q
bash scripts/train_rmsnorm2_reverse_sigmoid_blend_k20_a0p3_dino_none_cond_projtycond_consistency_B80_8gpu.sh --dry-run
bash scripts/train_rmsnorm2_reverse_sigmoid_blend_k20_a0p3_dino_none_cond_projtycond_consistency_B80_8gpu.sh --preflight
```
