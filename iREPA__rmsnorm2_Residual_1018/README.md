# iREPA + rmsnorm2 Residual（1018）

以当前 `iREPA/` 为基座，只增加生成侧的 DINOv3-LoRA、PgT 和双路 RMSNorm
时间融合。**JiT+iREPA 的 teacher、数据预处理、对齐 projector、loss、训练循环、
AdamW、EMA、checkpoint 和 Heun/CFG 采样全部执行原版实现。**

`jit/` 的 23 个 Python 文件与 2026-10-18 本地 `iREPA/jit/` 快照逐字节一致，
SHA256 固定在 `upstream_manifest.json`，启动时校验。这里的“原版”指用户指定的
本地 iREPA，已含此前 ImageNet/DINOv1/FID 路径、可选 wandb 和 CLI 优先级适配；
不将这些本地运行适配宣称为未经修改的公开上游提交。

## 代码边界

```text
main.py                       参数/资源/输出检查；替换原 trainer 的模型工厂
generation_input.py           唯一新增的 JiT 接入逻辑，约 180 行
dino_v3_lora.py + dino_lora.py  复用 1016 的生成 DINO/LoRA/MLS 实现
pixeldit2_projector.py         复用 1016 的 PgT
resources.py                  给原版 teacher loader 提供本地资源路径
jit/                          原版 iREPA Python 文件，未修改
SSL_code/dinov3/               本地 DINOv3 依赖与许可证
configs/、scripts/、tests/     配置、启动和验证
```

`generation_input.py` 构造原版 `Denoiser`，在原生 patch embedder 的输出处
安装 hook。它捕获本次 `JiT.forward(x,t,y)` 的 t，融合后交回原版 JiT，位置
编码仍由原版只加一次。原版原生 patch embedding 及对齐 projector 的名称保持
原样；新增参数集中在 `net.generation_input.*`。

```text
noisy z -> 原生 JiT patchify ------> RMSNorm(gamma_jit) --*t-------+
                                                               +-> 原版 APE / JiT / 输出
noisy z -> DINOv3-LoRA -> PgT(t) -> RMSNorm(gamma_dino) --*(1-t)--+

clean uint8 -> 原版 DINOv3 teacher preprocess / final patch tokens
            -> 原版 spatial z-score(alpha=.8)
JiT block 4 -> 原版内部 Conv3 projector -> 原版 cosine loss，权重 .5
```

生成 DINO 使用 raw noisy z，不做图像标准化；LoRA r/alpha64，dropout0，
训练 patch embedding，最终 LN 的预训练 affine 保留并冻结。
两路 RMSNorm 均为逐 token 沿通道归一化，具有各自独立的可学习 gamma。
默认保留最佳实验的 `(1-t)*DINO + t*JiT`，PgT/JiT 接收原始 t。
关闭生成分支时返回原版 `Denoiser`，不建立生成模块或 hook。

这里不再使用旧 `DINOJiTDenoiser`，也不引入其 DDTHead、DropRes、独立对齐
projector、teacher 预处理或 flow-DPM sampler。`dino_v3_lora.py` 相对 1016
仅修改资源导入路径，LoRA、PgT 数值代码直接复用并保留来源和许可证。

## 配置

| configs/ 下的文件 | 生成分支 | P_mean | 用途 |
|---|---|---:|---|
| `irepa_b80.yaml` | 无 | -0.8 | 原版 JiT-B + iREPA / DINOv3-B 对照 |
| `reverse_b80.yaml` | MLS `[2,5,8,11]` + 反向融合 | -0.8 | 1018 实现的参考实验 |
| `reverse_nomls_b80.yaml` | 最后一层最终 LN 后 token + 反向融合 | -0.8 | agent-gpu 八卡消融 |
| `reverse_pmean0_b80.yaml` | MLS + 反向融合 | 0.0 | gpu-2 八卡消融 |

NoMLS 设置 `dino_aggregation=none,dino_layers=null`，不会在最后层 token 上
再做 MLS 或重复 LN；送入现有 PgT。PMean0 只修改时间采样参数，仍为
`t=sigmoid(N(0,.8²))`，总体期望 `E[t]=E[1-t]=.5`，有限 batch 存在波动。
这两个消融分别相对 `reverse_b80.yaml` 改一个因素。

共同设置：JiT-B/16，ImageNet256，冻结 DINOv3-B/16 teacher，第4层 Conv3
对齐，cosine .5、z-score alpha .8；**8×128=1024，LR2e-4（blr5e-5），
无梯度累积，80 epoch，warmup5 后 constant**；AdamW(.9,.95)，weight decay0，
EMA .9999/.9996，seed0，label drop.1，P_std.8、noise_scale1、t_eps.05。
每10 epoch 保存检查点、每epoch更新last；每20 epoch在线评测
Heun50/CFG1/50K，gen_bsz64，interval[0,1]。

本次固定的是原版 iREPA 训练语义，并非旧 Ours 训练轨迹的逐位等价重现。
初始化顺序、teacher 路径和 checkpoint 命名空间都以新基座为准，不能将历史
Ours checkpoint 直接续训到本目录。两组新实验均从头训练；比较时应同时保留
`reverse_b80.yaml` 的新基座参考。原目录与已运行实验不受影响。

## 运行

共享环境默认 `../../envs/irepa-py310-cu126-20261015/bin/python`，可由
`PYTHON_BIN` 覆盖。新环境先按 `require.txt` 准备依赖及匹配的 CUDA wheels；
GPU 运行阶段不安装或下载资源。

资源默认复用上级 `pretrain_models/`、`fid_stats/`，以及
`/mnt/shared-storage-user/datafactory-share/gushishen/imagenet`。
可用 `PRETRAIN_MODELS_ROOT`、`FID_STATS_ROOT`、`IMAGENET_ROOT` 覆盖。
权重需要 `dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth`，在线评测需要
`jit_in256_stats.npz` 和 `weights-inception-2015-12-05-6726825d.pth`。
teacher 和生成 DINO 都读取本目录的 `SSL_code/dinov3/`；不依赖兄弟方法源码。

在对应机器进入本目录后：

```bash
# agent-gpu / GPU 0–7：NoMLS
bash scripts/train_rmsnorm2_reverse_nomls_B80_agent_gpu_8gpu.sh --preflight
bash scripts/train_rmsnorm2_reverse_nomls_B80_agent_gpu_8gpu.sh

# gpu-2 / GPU 0–7：μ=0
bash scripts/train_rmsnorm2_reverse_pmean0_B80_gpu2_8gpu.sh --preflight
bash scripts/train_rmsnorm2_reverse_pmean0_B80_gpu2_8gpu.sh
```

脚本检查指定主机和八卡空闲状态，默认在独立 tmux 中启动；`--foreground`
前台运行，`--resume` 只恢复本实验的新格式 checkpoint。不会停止其他任务或
自动排队。GPU 主机映射沿用现有实验：agent-gpu=`gpu-lg-cmc-h-h200-0029`，
gpu-2=`gpu-lg-cmc-h-h200-0257`。准备时 gpu-2 的已有 SSH 配置认证失败，
远端连通性需在实际启动前恢复。

Checkpoint、TensorBoard、解析配置和 `eval_metrics.jsonl` 存在本目录
`checkpoints/<experiment>/`；文本 `train.log`、命令及源码哈希存在共享
`../../work/train-irepa1018-reverse-{nomls,pmean0}-B80-8gpu-b128-gbs1024/`。
两份配置都保留原实验命名，但目录和 tmux 会话独立于 1016。

其他机器或参考配置可直接使用原版 torchrun 方式，无需依赖主机专用脚本：

```bash
WORLD_SIZE=8 python main.py --config configs/reverse_b80.yaml --preflight
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.run \
  --standalone --nproc_per_node=8 main.py --config configs/reverse_b80.yaml
```

## 验证

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 TORCH_COMPILE_DISABLE=1 PYTHONPATH=.:jit \
  python -m pytest -q tests
OMP_NUM_THREADS=2 python scripts/verify_real_model.py \
  --config configs/reverse_nomls_b80.yaml --device cpu --output /tmp/nomls_check.json
```

测试覆盖原版源码哈希、baseline 的初始化/RNG/loss/梯度/AdamW/EMA/CFG/Heun
一致性、融合位置与两种权重方向、独立 RMSNorm 梯度及异常后的 t 清理。
真实模型检查加载 ImageNet 和真实 DINOv3 teacher/student，执行完整 JiT-B
前向/反向。CPU 验证仅将原版硬编码 `.cuda()` 的 Tensor 转移重定向到 CPU，
没有修改生产计算代码。它不证明八卡 BF16/DDP、生产 batch 的显存或完整
50K 采样已通过；准备阶段不启动训练。

许可证：iREPA 见 `LICENSE`；DINOv3 见 `SSL_code/dinov3/LICENSE.md`；
PgT 和 MLS 来源见 `third_party/pixeldit2/`、`third_party/raev2/`。
