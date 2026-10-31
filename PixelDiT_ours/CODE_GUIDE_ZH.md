# PixelDiT_ours 中文代码导读

本文说明 PixelDiT v1（class-to-image）及我们新增的 DINO+LoRA 双路 patchify。PixelDiT2、T2I 和 `third_party/` 内的官方实现不属于本次方法改动。算法改动集中在 patch-level 输入；其他改动主要是离线资源、实验配置和中文注释。

## 1. 推荐阅读顺序

1. **先看整体结构**：根目录 `README.md` 和本文。先弄清两个 patch 级输入分支，以及 PixelDiT 原有的 patch/pixel 双尺度结构。
2. **读主模型 `pixdit_core/pixeldit_c2i.py`**：重点看 `AugmentedDiTBlock`、`PiTBlock`、`PixDiT.forward()`。理解 patch 级语义处理与 patch 内像素解码，以及两路 token 在哪里相加。
3. **读基础组件 `pixdit_core/modules.py`**：RMSNorm、RoPE 注意力、时间/类别嵌入、SwiGLU、原生 `PatchTokenEmbedder`。注意绝对 sin-cos 位置编码与注意力 RoPE 是两套机制。
4. **读我们的 DINO 支路**：
   - `pixdit_core/dino_patch_embedder.py`：DINO token 如何经 PgT 和 RMSNorm 变成 PixelDiT 输入。
   - `pixdit_core/dino_encoder.py`：LoRA 冻结策略、y cond 注入位置、final LN、选层与 MLS 聚合。
   - `pixdit_core/dino_lora.py`：`frozen Linear + (alpha/r)BA` 的增量形式。
   - `pixdit_core/dinov3_backbone.py`：本地 DINOv3 源码和权重如何加载。
5. **读 PgT 和时间融合**：`pixdit_core/grounding.py` 的 `TimestepProjection`，以及 `pixdit_core/blend_time.py`。PgT 使用原始 `t`；`f(t)` 只控制融合权重。
6. **读训练与 REPA**：`c2i/src/diffusion.py`。先看 `LinearScheduler` 和 `REPATrainer._impl_trainstep()`，再看 frozen `DINOv2` 和 `FlowDPMSolverSampler`。
7. **读数据与运行生命周期**：`c2i/src/data.py`、`c2i/src/lightning.py`、`c2i/src/utils.py`。理解 normalized/raw 两种图像、EMA、在线验证和 `output.npz`。
8. **最后看配置与启动**：`c2i/configs/`、`c2i/main.py`、`scripts/`。配置决定具体模型和实验参数，脚本只负责环境、GPU 连接、预检和前台启动。

## 2. 项目组织结构

```text
PixelDiT_ours/
├─ pixdit_core/                       # 模型源码
│  ├─ pixeldit_c2i.py                # [修改] PixelDiT v1 主干 + 我们的双路 patchify
│  ├─ modules.py                     # [仅补注释] 上游通用模块
│  ├─ grounding.py                   # [仅补注释] 上游 PgT/TimestepProjection，直接复用
│  ├─ pixeldit2_c2i.py               # [上游] PixelDiT2，未修改
│  ├─ pixeldit_t2i.py                # [上游] T2I，未修改
│  ├─ blend_time.py                  # [新增] linear/smooth/smoother/归一化 sigmoid
│  ├─ dino_encoder.py                # [新增] DINOv3+LoRA、y cond、选层/MLS
│  ├─ dino_lora.py                   # [新增] LoRA 包装与注入
│  ├─ dino_patch_embedder.py         # [新增] DINO → PgT → RMSNorm
│  └─ dinov3_backbone.py             # [新增] 本地 DINOv3 加载器
├─ c2i/
│  ├─ main.py                        # [上游] LightningCLI 训练/采样入口
│  ├─ configs/
│  │  ├─ pix256_xl.yaml              # [修改] 原 XL 配置 + DINO 参数；smoother、无 y cond
│  │  ├─ pix256_xl_dinov3b_ycond_sigmoid_k20_a0p1_norm.yaml  # XL a=0.1
│  │  ├─ pix256_xl_dinov3b_ycond_sigmoid_k20_a0p1_norm_{80,160,320}.yaml  # XL 分阶段
│  │  ├─ pix256_b_dinov3b_ycond_sigmoid_k20_a0.1_norm.yaml   # B 80ep a=0.1
│  │  ├─ pix256_b_dinov3b_ycond_sigmoid_k20_a0.1_norm_160.yaml  # B 160ep
│  │  ├─ pix256_b_dinov3b_ycond_linear.yaml             # B 80ep 线性融合
│  │  └─ 其他 YAML                   # [上游] 其他模型配置
│  └─ src/
│     ├─ diffusion.py                # [修改] REPA/采样不变；DINOv2 支持本地加载并补注释
│     ├─ data.py                     # [仅补注释] 上游 HDF5/ImageFolder/采样数据集
│     ├─ lightning.py                # [仅补注释] 上游训练、EMA、预测生命周期
│     └─ utils.py                    # [仅补注释] 上游 EMA、checkpoint、图片保存
├─ scripts/                          # [新增] 本地训练、预检、SSH 启动脚本
├─ third_party/
│  ├─ dinov2/                        # [新增] 离线 DINOv2 源码、LICENSE、source_manifest
│  └─ dinov3/                        # [新增] DINOv3 运行源码与 LICENSE.md
├─ pretrain_models/                  # [本地资源] PixelDiT v1 权重和 torch-hub 缓存，非源码
├─ tools/
│  ├─ download.py                    # [修改] 已发布 checkpoint 优先从本地解析
│  └─ 其他工具                       # [上游]
├─ t2i/                              # [上游] T2I 代码，未修改
├─ assets/                           # [上游] 图片资源
├─ train_logs/                       # [运行产物] 训练日志、checkpoint、W&B offline run
├─ requirements*.txt、pyproject.toml  # [上游] 依赖与打包配置
└─ README.md、LICENSE、NOTICE、SECURITY.md
```

## 3. 模型主流程

### 3.1 两路 patchify 与融合

当前 `PixDiT.forward()` 的输入为带噪图像 `x [B,3,H,W]`、时间 `t [B]`、已 drop 的类别 `y [B]`。可选 `patch_condition [B,N,C]` 是外部已完成 patch 级处理的条件；`dino_patchify_mode=dino_only` 时只走 DINO 路，`dino_time_blend_reverse=true` 时交换下述两路权重：

```text
x_t
├─ DINOv3+LoRA → final LN → 选层/MLS → PgT(原始t) → RMSNorm × (1-f(t))
└─ raw unfold → s_embedder → RMSNorm ─────────────────────── × f(t)
                                                        ↓ 相加
                                         融合后的 patch tokens [B,N,C]
```

- **raw 分支**：`s_embedder` 把每个 patch 的 `3P²` 像素向量映射到 `C=hidden_size`，是 PixelDiT 原有 patchify。
- **DINO 分支**：提取预训练语义特征。原 backbone 冻结，训练 LoRA、可配置的 patch embedding 和 y embedding。
- **两路 RMSNorm**：先统一两路 token 的通道尺度，再加权，避免直接相加时某一路尺度占优。
- **权重方向**：默认 `(DINO,raw)=(1-f(t),f(t))`，reverse 后为 `(f(t),1-f(t))`。因此默认 `t=0`（噪声）全部使用 DINO，`t=1`（干净图）全部使用 raw patchify。
- **四种函数**：`linear=t`、`smooth=3t²-2t³`、`smoother=6t⁵-15t⁴+10t³`，以及端点归一化 sigmoid。当前 DinoYCond 配方使用后者，`k=20`，`a=0.1` 或 `0.3`，保证 `f(0)=0,f(1)=1`。

`f(t)` 只用于融合。扩散加噪、PgT 的时间条件、主 PixelDiT 的 `t_embedder` 和采样步进都继续使用原始 `t`。

### 3.2 DINO 支路内部

`dino_encoder.py` 的顺序是：

1. 加载本地 DINOv3 权重。
2. 冻结 backbone，向 `qkv/proj/fc1/fc2` 注入 LoRA。
3. 可选地训练 DINO patch embedding；final LN 默认冻结并保留 affine。
4. `dino_label_cond=True` 时，在 **patch embedding 之后、DINO blocks 之前**，给每个 patch token 加独立的 `y_embedder(y)`。
5. 各选中层先经过 final LN，再去掉 CLS/register token。
6. 按 `dino_layers` 取层并聚合：`mls = 选层均值 + 最后选层的空间均值`。

当前配方下，student 直接使用带噪 `x_t`，不做额外的 ImageNet 或 t 相关归一化；frozen teacher 的固定 ImageNet 标准化与 student 预处理相互独立。

`ConditionalPatchEmbed` 是注册的 `nn.Module` 包装，不用闭包替换 forward，确保 Lightning 深拷贝出的训练模型和 EMA 不共享条件状态。

`dino_patch_embedder.py` 随后执行：

```text
DINO tokens [B,Nd,D]
  → 必要时按二维网格插值到 PixelDiT 的 N 个 patch
  → PgT：D 维投影到 C 维 + 原始 t 的 adaLN + RoPE 注意力
  → RMSNorm
  → 输出 [B,N,C]
```

PgT 直接复用上游 `grounding.py` 的 `TimestepProjection`，不是我们重新实现的另一套 projector。

### 3.3 PixelDiT 双尺度主干

融合后的 token 进入 `patch_depth=26` 个 `AugmentedDiTBlock`：

- RMSNorm + RoPE 全局注意力负责不同 patch 间的信息交换。
- `c = SiLU(t_emb + y_emb)` 是 PixelDiT 原有的全局条件，供 adaLN 使用。
- DINO 支路的 y embedding 与这里的 PixelDiT 主 `y_embedder` 是独立参数。

随后 `s = SiLU(t_emb + s)` 作为 pixel 级条件 `s_cond [B*N,C]`。`PixDiT` 仍保留原来的 pixel 分支：

```text
原带噪图像 x_t
  → PixelTokenEmbedder：逐像素投影 + 绝对位置编码
  → [B*N,P²,pixel_C] 的 patch 内像素序列
  → pixel_depth=4 个 PiTBlock
  → FinalLayer：每个像素 token → RGB
  → fold 回 [B,3,H,W]
```

`PiTBlock` 的关键过程是：**把一个 patch 的 P² 个像素压成一个注意力 token，在全图 patch 间做注意力，再展开回 P² 个像素**。这样既能利用全局语义，又能恢复 patch 内细节。

若调用方显式传入 `s`，则把它视为已计算的 patch 条件，跳过 patchify 和 patch blocks；通常训练和采样都走 `s=None` 的上述完整路径。

## 4. REPA、扩散损失与采样

### 4.1 REPA 对齐

REPA 位于 `c2i/src/diffusion.py` 的 `REPATrainer`，与 DINO 生成分支是独立的两套编码器：

```text
干净 raw_image [0,1]
  → frozen DINOv2-B/14 teacher
  → 目标 tokens [B,N,768]

x_t、t、y 经过 PixelDiT
  → 第 8 个 patch block 的输出（26 层中的第 8 层，索引7）
  → 可训练 MLP：1152 → 1152 → 768
  → 与 frozen teacher 做逐 token cosine
```

- `align_layer: 8` 在 `c2i/configs/pix256_xl.yaml` 中指定。
- `net.patch_blocks[align_layer-1]` 注册 forward hook，读取中间特征后立即移除 hook。
- DINOv2 使用 `[0,1]` 干净原图，不是带噪 `x_t`；256×256 输入默认缩到 224×224。
- token 数不同时按二维网格池化或插值；默认 256×256 时两边均为 256 token。
- `cos_loss = 1 - cosine_similarity`，最终 `loss = fm_loss.mean() + 0.5 * cos_loss.mean()`。
- frozen teacher 不更新，MLP 投影可训练并进入 optimizer。

因此，DINO student 不是 REPA teacher。DINO student 先参与 patchify 融合，再由前 8 个 patch blocks 处理后的特征与 frozen DINOv2 对齐。

### 4.2 扩散目标

`LinearScheduler` 使用：

```text
x_t = t*x + (1-t)*noise
v_t = x - noise
```

`t=0` 为噪声、`t=1` 为干净图像。`fm_loss` 是网络输出与解析 velocity 的 MSE。`BaseTrainer` 默认以 0.1 概率把类别替换为 null 类 1000，训练无条件生成能力。

### 4.3 EMA 与采样

- `LightningModel` 保存训练中的 `denoiser` 和独立的 `ema_denoiser`。
- `SimpleEMA` 只对 `requires_grad=True` 的参数执行 `ema = decay*ema + (1-decay)*current`，默认 `decay=0.9999`；frozen teacher/backbone 不做 EMA。
- 默认验证和正式采样使用 EMA，除非设置 `eval_original_model=True`。
- `FlowDPMSolverSampler` 首步 Euler、后续 AB2。每个采样步将 `[uncond, cond]` 拼成 `2B` 一次前向，在 CFG 区间内执行 `uncond + scale*(cond-uncond)`。
- 在线 validation 每 100000 training steps 生成 50000 张图，保存 `output.npz`；它只保存样本，不在线计算 FID/IS。

## 5. 我们新增的文件

### 模型与算法源码

- `pixdit_core/blend_time.py`：四种融合权重函数及归一化 sigmoid。
- `pixdit_core/dino_encoder.py`：DINOv3+LoRA、可选 y cond、选层/MLS、final LN 和可训练参数控制。
- `pixdit_core/dino_lora.py`：LoRA 线性层和按名称注入。
- `pixdit_core/dino_patch_embedder.py`：DINO token 的网格适配、PgT 和融合前 RMSNorm。
- `pixdit_core/dinov3_backbone.py`：本地 DINOv3 结构和权重加载。

### 配置与运行脚本

- `c2i/configs/pix256_xl_dinov3b_ycond_sigmoid_k20_a0p1_norm.yaml` 及 `_80/_160/_320` 阶段配置。
- `c2i/configs/pix256_b_dinov3b_ycond_sigmoid_k20_a0.1_norm.yaml`、`_160` 和线性融合配置。
- `scripts/run_on_gpu.sh`：含机器连接信息，仅在本机维护；远端可直接在 GPU 上运行其余脚本。
- `scripts/train_env.sh`：环境、资源预检、日志和前台运行工具。
- `scripts/train_pixeldit_xl_dinov3b_ycond_sigmoid_k20_a0p1_norm_8gpu.sh` 及 `_80/_160/_320` 阶段脚本。
- `scripts/train_pixeldit_b_dinov3b_ycond_sigmoid_k20_a0.1_norm_4gpu.sh`、`_160` 和线性融合脚本。
- `scripts/sample_ckpt_8gpu.sh`：通用 checkpoint 采样，支持 FlowDPM、Heun、Euler 及 CFG 区间覆盖。
- `scripts/evaluate_npz_fid_is_fdr6.sh`：使用 `nanogen-evals` 离线计算 NPZ 的 FID、IS、FDR6。
- `scripts/evaluate_npz_jit_fid_is.sh`：按 JiT/torch-fidelity 协议直接计算 NPZ 的 FID、IS。

### 第三方运行源码

- `third_party/dinov2/`：离线 DINOv2 源码、许可证和来源清单。
- `third_party/dinov3/`：DINOv3 运行所需源码与 `LICENSE.md`。
- `tools/evaluation/`：离线 NPZ 评测入口、缓存解析及 vendored `fd_evaluator`；FDR6 协议与其他方法一致。
- `tools/evaluate_jit_fid_is.py`：复用 JiT 的 torch-fidelity 特征与统计函数，免 PNG 转换。

这些目录只保留运行所需代码和许可证，不包含训练 checkpoint、日志或采样产物。

## 6. 基于上游修改的文件

- `pixdit_core/pixeldit_c2i.py`：接入双路 patchify、DINO 参数和完整中文流程注释；PiT/DiT 主体保持原结构。
- `c2i/src/diffusion.py`：DINOv2 优先用 `source="local"` 离线加载，避免 torch-hub 缓存命中后仍访问 GitHub；REPA 和采样算法不变。
- `tools/download.py`：已发布 checkpoint 优先从 `pretrain_models/` 返回，缺失时才下载。
- `c2i/configs/pix256_xl.yaml`：增加 DINO 配置项；其余 XL、REPA、优化器和采样默认值不变。
- `pixdit_core/modules.py`、`pixdit_core/grounding.py`、`c2i/src/data.py`、`c2i/src/lightning.py`、`c2i/src/utils.py`：只补充中文模块说明、维度与生命周期注释，没有改变算法。
- `README.md`、`c2i/README.md`、`.gitignore`：补充方法说明和本地资源/产物忽略规则。

`c2i/configs/pixeldit2_h16_in256.yaml` 的既有本地注释修改早于本次工作，不属于本次新增或算法改动。PixelDiT2 和 T2I 源文件均未修改。

## 7. 配置、资源与运行输出

- **基础配置 `pix256_xl.yaml`**：默认 `smoother`，`dino_label_cond` 未开启；不是 DinoYCond 实验。
- **DinoYCond 配置**：当前发布 A0.1 归一化 sigmoid `k=20`、y cond、LoRA rank/alpha=64、MLS/default 选层、`qkv,proj,fc1,fc2`，以及对应的 XL/B 阶段配置。
- **数据**：新配方使用上游已有的 `ImageFolderDataset`，读取原始 ImageNet，采用 ADM 中心裁剪，`random_flip: false`。已用真实样本验证它与 REPA-E HDF5 的像素和归一化输入一致。
- **PixelDiT v1 权重**：统一使用主仓库 `Pixel_DIffusion/pretrain_models/` 下的四个 checkpoint。
- **DINOv3 权重**：默认从 `PRETRAIN_MODELS_ROOT` 或主仓库 `Pixel_DIffusion/pretrain_models/` 读取，也可在 YAML 中显式指定 `dino_weights`。
- **DINOv2 teacher**：从统一目录 `pretrain_models/torch_hub/` 的本地源码和权重缓存加载。
- **日志**：`train.log` 实时写入对应 `RUN_DIR/train.log`；W&B 固定 `offline`，指标写入 `RUN_DIR/wandb/` 下的 offline run，不自动在线上传。
- **频率**：XL 指标每500 steps，B 指标每50 steps；XL checkpoint 和在线采样每100000 steps，B 按 epoch 配置执行。
- **同步纪律**：`scripts/`、checkpoint、训练日志、采样结果和缓存只作为本地资源，不按算法源码上传。
