# DinoYCondConsistency：DINO 与 JiT 共享类别 embedding

> GitHub 仅发布源码、配置、依赖清单和许可证。下文的 `scripts/`、`tests/`、tmux 与 GPU 连接说明对应 H200 本地工作区，不随本次代码同步；其他机器请自行准备启动脚本与资源路径。

本目录基于 `Ours_BaseT_rmsnorm2_Residual_1016`，只增加 **DINO patch 后的类别条件是否复用 JiT `net.y_embedder`** 的开关。默认实验使用同一个 embedding 模块和权重，不是复制两份初值。

参考 `dinotycond_projtcond_consistency` 的共享方式，但本版本默认 **DINO 只加 y，不加 t；PgT 仍使用自己的原始时间 embedding**。iREPA teacher 继续独立冻结、保持 eval，输入为 ImageNet 标准化后的干净图像。没有增加 consistency loss。

## 默认配置与数据流

- 配置：[reverse_sigmoid_blend_k20_a0p3_dino_ycond_consistency_B80.yaml](configs/reverse_sigmoid_blend_k20_a0p3_dino_ycond_consistency_B80.yaml)
- 脚本：[八卡训练入口](scripts/train_rmsnorm2_reverse_sigmoid_blend_k20_a0p3_dino_ycond_consistency_B80_8gpu.sh)
- 参考配方：[原 DinoYCond 配置](../Ours_BaseT_rmsnorm2_Residual_1016/configs/reverse_sigmoid_blend_k20_a0p3_dino_ycond_B80.yaml)

```text
labels（训练时已做 label drop，采样时按 CFG 分支给标签）
  → 同一个 JiT net.y_embedder
      ├─ 加到 DINO patch tokens，再经过 DINO+LoRA
      └─ 与 JiT net.t_embedder(t) 相加，调制 JiT blocks 和输出层

带噪图像 z_t
  ├─ JiT patch embed → JiT RMSNorm → ×f(t) ────────────────────────┐
  └─ DINO patch embed + 共享 y_emb → DINO+LoRA → MLS → PgT(t)      │
                                          → DINO RMSNorm → ×(1-f(t))
                                                                ↓ 相加
                                                 加位置编码 → JiT blocks
                                                    ├─ 第4层 → ConvProj → iREPA loss
                                                    └─ final layer → 干净图像预测
干净图像 → ImageNet 标准化 → frozen DINO teacher → 对齐目标
```

`f(t)=sigmoid(20*(t-0.3))` 只控制融合权重；加噪、JiT 时间条件和 PgT 时间条件仍使用原始 `t`。本版本不添加独立的4层分支。

## 共享开关

默认实验在 `model.dino_student` 设置：

```yaml
dino_student_time_cond: false
dino_student_label_cond: true
dino_jit_label_consistency: true
```

`dino_jit_label_consistency` 的代码默认值为 `false`，保留旧配方的独立 DINO label embedding；本实验 YAML 显式开启。`true` 时，DINO 路径通过 JiT 的同一个 `net.y_embedder` 取类别向量，两条生成路径的梯度共同更新这份参数。训练 label drop 和 CFG 无条件分支都使用相同的 null 类（ImageNet 1000 类时为索引1000）。

末尾追加 `--no-dino_jit_label_consistency` 可关闭共享。切换开关属于不同实验，应使用新的实验名和输出目录；不将旧独立 label embedding 的 optimizer 状态直接续训到共享实验。

共享会自然改变 DINO label 条件的初始幅度：原独立 `nn.Embedding` 默认 std 约1，共享后沿用 JiT 的 std=0.02。我们不额外缩放共享向量，也不改变其余模块的初始化规则。PgT 的时间 MLP、JiT 的时间 MLP仍各自独立，不受此开关控制。

其余训练超参数与参考配方一致：8卡×每卡128，全局1024，`blr=5e-5` 对应 LR `2e-4`，80 epoch，warmup 5，`P_mean=-0.8`、`P_std=0.8`，EMA `0.9999/0.9996`。生成 DINO 使用 LoRA r64、MLS，训练 patch embed，final LN 冻结；NoDDTHead。每20 epoch 在线评测 Heun50、CFG1、interval `[0,1]`、50K，`gen_bsz=64`。损失仍是去噪损失加 `0.5 × iREPA` 对齐损失。

## 从哪些文件读起

| 文件 | 职责 |
|---|---|
| [main_dino_jit.py](main_dino_jit.py) | 声明开关，接入 iREPA 训练入口。 |
| [dino_jit_denoiser.py](dino_jit_denoiser.py) | 构建共享关系，传入 JiT label embedding，完成融合与损失。 |
| [dino_v3_lora.py](dino_v3_lora.py) | DINO+LoRA 包装；在 patch tokens 上加类别条件，控制可训练参数。 |
| [dino_encoder.py](dino_encoder.py) | 分别构建生成 student 和独立 frozen teacher。 |
| [irepa_teacher.py](irepa_teacher.py) | iREPA 的 frozen DINO 编码行为，不读取 student 条件开关。 |
| [pixeldit2_projector.py](pixeldit2_projector.py) | 生成侧 PgT；时间编码和前向保持基座实现。 |
| [blend_time.py](blend_time.py) | 只变换融合系数。 |
| [jit/model_jit.py](jit/model_jit.py) | JiT 模型、共享的 `y_embedder`，及原有 `t_embedder`。 |
| `jit/` 其余文件 | 沿用基座的 iREPA/JiT 训练、对齐、评测和本地资源适配。 |
| `configs/`、`scripts/` | 仅保留本实验的配置、八卡训练入口和公共运行工具。 |
| `tests/` | 配置、共享参数、梯度与基座行为的回归检查；历史测试配置单独放在 `tests/fixtures/configs/`。 |

## 在开发机 tmux 中运行

脚本默认连接 GPU1（`agent-gpu`），GPU 0–7；`GPU_HOST=agent-gpu2` 可切换 GPU2。已在 GPU 节点执行时直接使用当前节点。下面的 `--dry-run` 只打印命令，不连接 GPU、不训练。

```bash
cd /mnt/shared-storage-user/datafactory-share/gushishen/agent-z/Pixel_DIffusion/Ours_BaseT_rmsnorm2_Residual_1016_dinoycond_consistency
bash scripts/train_rmsnorm2_reverse_sigmoid_blend_k20_a0p3_dino_ycond_consistency_B80_8gpu.sh --dry-run

tmux new-session -s 'BaseT-Residual(1-t,t)-NoDinoPre-PgT-blend_rmsnorm2-SigmoidBlendK20-A0p3-DinoYCondConsistency-NoDDTHead-B-80' -c "$PWD"
# 进入开发机 tmux 后执行：
bash scripts/train_rmsnorm2_reverse_sigmoid_blend_k20_a0p3_dino_ycond_consistency_B80_8gpu.sh
```

`Ctrl-b d` 分离会话，之后在开发机执行 `tmux attach-session -t '=BaseT-Residual(1-t,t)-NoDinoPre-PgT-blend_rmsnorm2-SigmoidBlendK20-A0p3-DinoYCondConsistency-NoDDTHead-B-80'` 查看进度。

默认从头训练，日志和 checkpoint 位于本目录 `checkpoints/<实验名>/`，不复用基座输出。数据与预训练权重通过 `IMAGENET_ROOT`、`PRETRAIN_MODELS_ROOT` 指向现有资源；本目录不复制训练产物或模型权重。预检、换节点和续训见 [scripts/README.md](scripts/README.md)。
