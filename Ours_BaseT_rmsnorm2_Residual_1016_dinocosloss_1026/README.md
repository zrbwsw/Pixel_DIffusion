# DINO+LoRA → MLS → PgT：增加逐 patch cosine 对齐

本目录基于 `Ours_BaseT_rmsnorm2_Residual_1016` 的 DinoYCond / sigmoid K20、a0.3 配方，新增一项 DINO 分支对齐损失。原 JiT 第4层 iREPA 对齐和去噪损失保留；原项目代码、训练输出及正在运行的实验不变。

## 数据流和损失

```text
带噪图像 z_t = t*x + (1-t)*noise
  ├─ JiT patch embed → JiT RMSNorm → ×f(t) ─────────────────────────────┐
  └─ DINOv3+LoRA（patch 后加 y）→ MLS → PgT(t)
                                       ├─ 新增 patch cosine loss（对齐 teacher）
                                       └─ DINO RMSNorm → ×(1-f(t)) ───┤
                                                                    ↓ 相加
                                               位置编码 → JiT blocks
                                                 ├─ 第4层 → ConvProj → 原 iREPA loss
                                                 └─ final layer → 干净图像预测 → 去噪 loss

干净图像 x → ImageNet 标准化 → frozen DINOv3 teacher → 最后层 normalized patch tokens
                                               └─ 两项对齐损失共用同一次 teacher 输出
```

`f(t)=sigmoid(20*(t-0.3))` 只控制融合权重。加噪、JiT 时间条件和 PgT 时间条件继续使用原始 `t`。

令 `u` 为 **MLS → PgT(t) 输出、RMSNorm 之前**的 token，`z` 为 teacher 最后一层的 patch token。新损失采用标准 REPA 的负余弦形式：

\[
L_{\mathrm{dino\_cos}}=-\frac{1}{BN}\sum_{b,p}
\left\langle\frac{u_{b,p}}{\|u_{b,p}\|_2},
\frac{\operatorname{stopgrad}(z_{b,p})}{\|z_{b,p}\|_2}\right\rangle.
\]

每个空间位置一一对应，归一化沿通道维度进行。新损失不做 iREPA 的空间 z-score，也不新增投影头；默认 B/16 下，PgT 和 teacher 均为 `[B,256,768]`。负余弦可以小于零；改写成 `1-cos` 只增加常数，梯度相同。

```text
total_loss = denoise_loss + 0.5 * alignment_loss
             + dino_cos_loss_weight * dino_cos_loss
```

`dino_cos_loss_weight` 默认 `0.5`，设为 `0` 可关闭新增项。原 `alignment_loss` 继续使用 JiT 第4层 → ConvProj 的 iREPA 计算，包括原来的空间标准化。

`--no-irepa_alignment` 仅关闭原 JiT 对齐，新 DINO 对齐仍可单独工作；两项都关闭时才跳过 teacher。`native` 模式没有 DINO 生成分支，因此不启用新增项。

### 与原仓库 REPA 默认设置的对应关系

对照原 iREPA 仓库的 `jit/configs/repa.yaml` 及 `main_jit.py` 默认参数，**新增项的损失设置一致，对齐位置和投影结构沿用本方法**：

| 设置 | 原 REPA 默认 | 本方法新增 DINO 对齐 |
|---|---|---|
| 损失 | 逐 patch 负 cosine，再对 batch / patch 取均值 | 相同，直接复用 `jit/projection_loss.py` 的 `make_projection_loss("cosine")` |
| 特征归一化 | 沿通道 L2 归一化 | 相同 |
| 空间标准化 | `spnorm_method: none` | 不做 z-score，不添加 CLS 目标 |
| 权重 | `proj_coeff: "0.5"` | `dino_cos_loss_weight: 0.5` |
| 对齐输入 | JiT 第8层 → 三层 Linear / SiLU MLP，隐藏维2048 | DINO MLS → 现有 PgT(t)，不额外添加 MLP |
| teacher | 默认 DINOv2-B/14，ImageNet 标准化后 resize224 | 沿用实验的 DINOv3-B/16，ImageNet 标准化后保持256 |

`dino_cosine_loss.py` 只负责显式转为 FP32、teacher detach，以及用 `-loss` 返回平均相似度；归一化、点积与归约交给原 loss 实现。调用的是本项目已有源码，不依赖旁边的 `iREPA` 目录来训练。

原 JiT 第4层 ConvProj + z-score 的 iREPA 项仍单独保留；配置中的 `dino_zscore_alpha: 0.8` 只用于它，不作用于新增 DINO loss。

- **student** 直接接收带噪图像，不做输入标准化；训练 LoRA、patch embedding 和独立的 y embedding。默认不注入 student t embedding，原 final LN 继续冻结。
- **teacher** 接收干净图像，固定执行 `[-1,1] → [0,1] → ImageNet 标准化`，内部冻结并保持 eval；不使用 LoRA、MLS 或 student 开关。
- 新增损失反传到 PgT 及其上游 DINO student，teacher 不接收梯度。采样的数据流不变，不计算对齐损失。

训练日志中的新增指标：

| 指标 | 含义 |
|---|---|
| `dino_cos_similarity` | 所有样本、所有 patch 的平均余弦相似度，越高越对齐 |
| `dino_cos_loss` | 未加权的负平均余弦，即 `-dino_cos_similarity` |
| `dino_cos_loss_weighted` | 乘 `dino_cos_loss_weight` 后加入总损失的值 |

## 文件入口

| 文件 / 目录 | 用途 |
|---|---|
| `main_dino_jit.py` | 配置入口，声明新增损失权重，接入 iREPA 训练框架 |
| `dino_cosine_loss.py` | 调用原 `jit/projection_loss.py`，补充 FP32、teacher 停止梯度及相似度统计 |
| `dino_jit_denoiser.py` | 构建两路 token、保存 PgT 输出、计算新增损失及原有损失 |
| `dino_encoder.py`、`irepa_teacher.py` | 独立构建 student / frozen teacher |
| `dino_v3_lora.py`、`dino_lora.py` | student 的 LoRA、MLS 和参数训练控制 |
| `pixeldit2_projector.py` | 生成侧 PgT(t)，与 JiT 第4层的对齐 ConvProj 不同 |
| `blend_time.py` | sigmoid 融合系数 |
| `jit/` | iREPA/JiT 模型、训练框架及已有本地配置、日志和评测适配 |
| `SSL_code/`、`third_party/` | 必需的上游源码、许可证与来源说明 |
| `configs/` | 仅本方法默认配方 |

GitHub 仅发布代码、配置、依赖和说明；启动脚本、测试目录、训练和采样产物由各机器本地维护，不随本方法发布。

默认配方：[reverse_sigmoid_blend_k20_a0p3_dino_ycond_dinocosloss_B80.yaml](configs/reverse_sigmoid_blend_k20_a0p3_dino_ycond_dinocosloss_B80.yaml)。除新增损失权重及实验、日志名称外，超参数沿用原 DinoYCond K20 A0.3 配方：8×128=1024、LR 2e-4、80 epoch、LoRA rank 64、在线评测 Heun50 / CFG1 / 50K。

## 在运行机器上训练

在仓库根目录激活本机 Python 环境，并按 [部署说明](../REMOTE_SETUP.md) 准备依赖、预训练权重和评测资源。以下资源路径请替换为实际本地路径；在线评测默认开启，需要 FID 统计和 Inception 权重。

```bash
export IMAGENET_ROOT=/path/to/imagenet
export PRETRAIN_MODELS_ROOT=/path/to/pretrain_models
export FID_STATS_ROOT=/path/to/fid_stats
export NANOGEN_EVALS_ROOT=/path/to/nanogen-evals
cd Ours_BaseT_rmsnorm2_Residual_1016_dinocosloss_1026
export DINOV3_REPO="$PWD/SSL_code/dinov3"
export PYTHONPATH="$PWD:$PWD/jit${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 \
    main_dino_jit.py \
    --config configs/reverse_sigmoid_blend_k20_a0p3_dino_ycond_dinocosloss_B80.yaml
```

可以在运行机器上创建实验同名的 tmux 会话，再执行上述命令；从开发机启动时，由本机维护的 SSH 脚本负责连接 GPU。默认从头训练，写入本项目独立的 `checkpoints/<实验名>/`。恢复本实验时传入 `--resume /path/to/this_experiment_output`，并保持训练配置一致。离线采样入口见 [评测说明](jit/evaluation/README.md)。

## 验证

- 18 项 CPU 测试通过：逐 patch 计算、teacher 停止梯度、BF16 输入下以 FP32 计算、RMSNorm 前对齐、损失叠加及 CSV/JSONL 指标记录；FP32 / BF16 输入的 loss 与 student 梯度均与原 `iREPA/jit/projection_loss.py` 在相同 FP32 计算条件下逐值一致。
- 新增权重为 0 时，loss、日志和各参数梯度与原项目逐值一致；采样结果不受此损失权重影响。
- 真实 DINOv3-B 预训练权重 + 一张 ImageNet 256×256 图像完成 CPU 全模型前向和反向；teacher / PgT 各前向一次，对齐形状为 `[1,256,768]`。新增项向 LoRA、patch embedding、y embedding 和 PgT 传回有限且非零梯度，teacher 无梯度。
- 本地八卡启动脚本的语法、dry-run 和资源预检通过：8×128=1024，LR=2e-4，从头训练；启动脚本和测试代码未包含在此次发布中。

真实模型 CPU 验证仅在测试进程中将 JiT 的历史 `.cuda()` 分配留在 CPU，并禁用编译；训练代码不变。本次没有启动 GPU 训练。
