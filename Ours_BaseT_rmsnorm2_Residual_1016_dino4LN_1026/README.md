# DINO MLS 四层可学习 LN

基于 `Ours_BaseT_rmsnorm2_Residual_1016` 的 DinoYCond、sigmoid K20 / A0.3 配方。只改变生成侧 DINO+LoRA 的 MLS 归一化：层 2、5、8 各用一个新 LN，层 11 继续用预训练 final LN，四者均可学习。层索引从 0 开始。

```text
带噪图像 z_t（不做额外输入标准化）
  ├─ JiT patch embed ──────────────────────── JiT RMSNorm × f(t)
  └─ DINO patch embed + 独立 y embedding → DINO+LoRA blocks
         ├─ 层  2 → norm_2 ┐
         ├─ 层  5 → norm_5 ├─ 四层均值 + 最后层空间均值 → PgT(t)
         ├─ 层  8 → norm_8 │                              ↓
         └─ 层 11 → norm ──┘                 DINO RMSNorm × (1-f(t))
                                                            ↓
                                  两路相加 → 位置编码 → JiT blocks
                                           ├─ 第 4 层 → ConvProj → iREPA loss
                                           └─ 最后层 → 图像预测 → 去噪 loss

干净图像 x → ImageNet 标准化 → 独立冻结 DINO teacher → iREPA 对齐目标
```

其中 `f(t) = sigmoid(20 * (t - 0.3))`。DINO 仅在 patch 后注入 y，不注入 t；DINO 的 y embedding 与 JiT 结构相同，但参数各自独立。PgT 保持原有的 t 条件。

## LN 与可学习参数

| 层索引 | 生成侧模块名 | 初始化 | 默认是否训练 |
|---|---|---|---|
| 2 | `dino_student.model.norm_2` | γ=1，β=0 | 是 |
| 5 | `dino_student.model.norm_5` | γ=1，β=0 | 是 |
| 8 | `dino_student.model.norm_8` | γ=1，β=0 | 是 |
| 11 | `dino_student.model.norm` | 保留 DINOv3 预训练 γ/β | 是 |

三个新 LN 与 final LN 的维度和 epsilon 相同。每个选中层先单独归一化，再做 MLS：

B/16 中均为 `LayerNorm(768, eps=1e-5)`；三个新 LN 增加 4608 个参数，加上解除冻结的末层 LN，四个输出 LN 共训练 6144 个参数。默认初始化不消耗随机数，因此不会改变后续 y embedding 的初始化。

```python
mls = (tokens_2 + tokens_5 + tokens_8 + tokens_11) / 4
mls = mls + tokens_11.mean(dim=1, keepdim=True)
```

`lastmean` 使用层 11 归一化后的 tokens，求空间均值后不再做 LN。

默认配置显式启用两个开关：

- `dino_mls_layer_norms: true`：层 2/5/8 使用各自新建的可学习 LN。
- `dino_train_final_norm: true`：开放层 11 的预训练 final LN 的 γ/β。

原来的 LoRA、patch embed、DINO y embedding、PgT、RMSNorm、JiT 与对齐投影层继续训练；DINO backbone 其余参数冻结。iREPA teacher 的预处理、输出和冻结行为保持不变，不使用生成侧这四个可学习 LN。

## 文件入口

- `dino_v3_lora.py`：DINO+LoRA、各层 LN 与 MLS 聚合。
- `dino_encoder.py`：构建生成侧 student 和独立的冻结 teacher。
- `irepa_teacher.py`：干净图像的冻结 DINO teacher。
- `dino_jit_denoiser.py`：两路输入融合、JiT 前向与损失。
- `main_dino_jit.py`：训练入口和超参数。
- `blend_time.py`：只用于融合权重的 sigmoid 时间变换。
- `jit/`：沿用基项目的 JiT/iREPA 训练组件。
- `SSL_code/dinov3/`：DINOv3 backbone 源码。
- `configs/`：仅保留本实验默认配方。
- GitHub 只同步代码、配置、依赖和说明；各机器单独维护启动脚本、tmux 会话和训练产物。

## 默认配方与启动

配置：`configs/reverse_sigmoid_blend_k20_a0p3_dino_ycond_dino4ln_B80.yaml`。

默认八卡 × 每卡 128，全局 batch 1024，`blr=5e-5`，warmup 后 LR 为 `2e-4`，80 epochs；每 20 epochs 用 Heun50 / CFG1 在线评测 50K 图像。除四层 LN 和实验输出目录外，训练超参数沿用原 DinoYCond 配方。

配置默认从头训练，输出到本项目独立的 `checkpoints/<实验名>`。新增了 LN 参数，不能直接把旧结构 checkpoint 当作同一实验续训；本实验中断后可用 `--resume /path/to/output` 恢复自身输出目录。

在仓库根目录激活本机 Python 环境，并按 [部署说明](../REMOTE_SETUP.md) 准备依赖、预训练权重和在线评测资源。下面路径请替换为实际本地路径；在线评测默认开启，FID 统计与 Inception 权重必须已准备好。

```bash
export IMAGENET_ROOT=/path/to/imagenet
export PRETRAIN_MODELS_ROOT=/path/to/pretrain_models
export FID_STATS_ROOT=/path/to/fid_stats
export NANOGEN_EVALS_ROOT=/path/to/nanogen-evals
cd Ours_BaseT_rmsnorm2_Residual_1016_dino4LN_1026
export DINOV3_REPO="$PWD/SSL_code/dinov3"
export PYTHONPATH="$PWD:$PWD/jit${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 \
    main_dino_jit.py \
    --config configs/reverse_sigmoid_blend_k20_a0p3_dino_ycond_dino4ln_B80.yaml
```

可以在执行机器上创建以实验命名的 tmux 会话，再运行上述命令。默认从头训练；恢复本方法的同结构 checkpoint 时，传入 `--resume /path/to/this_experiment_output`。训练、在线评测和采样使用相同四层 LN 配置；离线采样入口见 [评测说明](jit/evaluation/README.md)。

## 验证

- 9 项针对性测试通过：独立初始化、逐层 LN / MLS 数值、梯度和优化器更新、teacher 隔离、权重保存恢复，以及关闭新开关后的兼容行为。
- 真实 DINOv3-B/16 权重、256×256 输入的 CPU 前向/反向通过，输出 `[1,256,768]`；四个 LN 的 gamma/beta 均有有限非零梯度，且完成优化器更新。
- 手工按四层 LN 重建的 MLS 与实际 forward 输出完全一致；teacher 保持冻结和 eval。
- 启动脚本语法、dry-run 和资源预检通过；采样配置保留四层 LN 和末层可学开关。
