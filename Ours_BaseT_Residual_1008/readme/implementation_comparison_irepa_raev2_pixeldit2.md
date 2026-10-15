# Residual-NoDinoPre-PgT-BlendT 实现审计

本文记录当前实验
`BaseT-Residual(t,1-t)-NoDinoPre-PgT-B-80`
与 `iREPA`、`RAEv2` 和 `PixelDiT2` 的代码关系。检查基于本地源码和配置，日期为
2026-10-07。

本地参考仓库的 HEAD 分别为 iREPA `99ad4ac234ef`、RAEv2 `6ea37982de9f`、
PixelDiT2 `8f83fa88a7b9`。本文核对的是磁盘上的实际源码，不假定这些 checkout
没有本地改动；结论限于当前实验配置，不代表所有历史分支。

## 结论

当前实现已经按预期组合了三类上游代码，但没有把它们完整地原样拼接到一起。

| 部分 | 当前结论 |
| --- | --- |
| JiT 主干和基础投影层 | 与本地 `iREPA/jit` 的可执行代码一致，当前 wrapper 改变了调用方式 |
| DINOv3 MLS | 聚合公式与 `RAEv2` 一致；最终 LayerNorm 设置不同，student/teacher、LoRA 和输入域也有适配 |
| DDTHead | 沿用 `RAEv2` 的逐 token AdaLN 和双流宽 head；patch embed、attention 实现、初始化等已有改动 |
| PixelDiT2 projector | `TimestepProjection` 已按官方源码复制，B 模型初始化和前向/反向对照测试通过 |
| Residual 与时间权重 | 这是本项目的核心改动；当前这个配置实际使用 `time_blend`，不是旧的 `gated_residual` adapter |
| 评测 | 加入项目内离线 `FID/IS/FDR6` 流程；online eval 仍只计算 FID/IS |

当前配置中的实际输入是：

```text
z_t -> JiT native patch embed --------------------------+
                                                       +-> (1-t) * native + t * Pg(DINO(z_t)) -> JiT -> DDTHead
z_t -> DINOv3-B LoRA -> MLS -> PixelDiT2 projector ----+
```

因此，当前实验名称中的 `Residual` 反映项目分支和实验家族；当前 YAML 中
`dino_jit_input_mode: time_blend`，公式是

```text
h_native = JiT_PatchEmbed(z_t)
h_dino   = Pg(DINOv3-LoRA-MLS(z_t), t)
h_0      = (1 - t) * h_native + t * h_dino + pos_embed
```

旧的残差 adapter 路径则是

```text
h_0 = h_native + g(t) * ResidualAdapter(DINOv3-LoRA-MLS(z_t)) + pos_embed
```

两者不能混称。`g(t)=t` 的 `gated_residual` 是另一个配置分支；当前实验使用的
是两路 token 的凸组合。虽然两路系数之和为 1，但这不保证两路 token 的范数相等。
旧残差 adapter 的末层置零，初始化时退化为完整 native JiT 输入；当前 PgT 的
输出层没有置零，因此 BlendT 不具有同样的 native-only 初始化性质。

## 当前配置和数据流

配置文件为
[`BaseT-Residual(t,1-t)-NoDinoPre-PgT-B-80.yaml`](../configs/BaseT-Residual(t,1-t)-NoDinoPre-PgT-B-80.yaml)。关键设置如下：

- `dino_preprocess_input: false`：student 直接接收未经映射和标准化的 noisy
  `z_t`，不做 clamp。干净图像在 `[-1,1]`，加高斯噪声后的 `z_t` 可以超出此范围。
- frozen teacher 仍通过 `(x+1)/2` 和 ImageNet mean/std 标准化 clean image。
- student 使用 DINOv3-B/16、LoRA-r64/alpha64、`mls`，默认层为 `[2,5,8,11]`
  （从 0 编号）；训练 qkv/proj/fc1/fc2 的 LoRA 和 patch embed，冻结其余原始参数。
- `dino_input_projector: pixeldit2`，当前 B 配置为 1 个 projector block、768 维、12 个 head。
- `dino_grounding_drop_on_label_drop: false`：CFG null-label 分支继续 forward DINO path。
- JiT alignment 在第 4 个 block 后进行，使用独立 `3x3 ConvProj` 与 teacher 对齐。
- 对齐损失固定乘以 `0.5`，不使用额外的时间加权。
- DDTHead 为 2048 维、16 heads、2 blocks，输入 RGB noisy image，输出 RGB patch。
- 训练为 80 epoch、全局 batch 256、每卡 batch 64、累积 1 次、5 epoch warmup、
  基础学习率 `1e-4`；这些是当前实验配置，不表示与各上游所有训练超参数相同。

核心连接逻辑位于 [`dino_jit_denoiser.py`](../dino_jit_denoiser.py)：
`_encode_noisy_student` 负责 DINO 输入，`_compose_jit_input_tokens` 负责融合，
`_jit_forward_from_tokens` 负责 JiT block 和对齐输出，`forward` 负责训练损失。

当前有三个不同用途的 projector，不能混用：

| 模块字段 | 输入和输出 | 用途/来源 |
| --- | --- | --- |
| `dino_input_projector` | DINO MLS 768 维 -> 768 维 | PixelDiT2 PgT，参与 JiT 输入融合 |
| `dino_student_projector` | JiT 第 4 个 block 768 维 -> 768 维 | iREPA ConvProj，只用于 teacher alignment；字段名不代表直接处理 DINO 输出 |
| `ddt_head.z_proj` | JiT 最后空间 token 加时间向量并 SiLU，768 维 -> 2048 维 | DDT base/head 边界投影 |

## 与 iREPA 的关系

### 原样复用的部分

去掉注释和 docstring 后，下面三个文件与本地 `iREPA/jit` 的可执行 AST 一致：

- [`jit/model_jit.py`](../jit/model_jit.py) 对应 [`iREPA/jit/model_jit.py`](../../iREPA/jit/model_jit.py)
- [`jit/projectors.py`](../jit/projectors.py) 对应 [`iREPA/jit/projectors.py`](../../iREPA/jit/projectors.py)
- [`jit/util/model_util.py`](../jit/util/model_util.py) 与 [`iREPA/jit/util/model_util.py`](../../iREPA/jit/util/model_util.py) 内容一致

这意味着 JiT 的 attention、SwiGLU、RMSNorm、RoPE 和 `BottleneckPatchEmbed`
等基础实现保持复用。这些文件的差异主要是注释和 docstring；当前模型如何调用
它们、如何维护 EMA，则由 denoiser/engine 负责，不包含在上述 AST 一致结论中。

### 保留的训练定义

当前扩散路径仍采用 iREPA/JiT 的 clean-weight 时间约定：

```text
z_t = t * x + (1 - t) * e
v   = (x - z_t) / clamp(1 - t, t_eps)
```

`P_mean=-0.8`、`P_std=0.8`、`noise_scale=1`、`t_eps=0.05`、label dropout、
EMA 和 rectified-flow 采样接口也沿用这套定义。

空间归一化仍是 iREPA 的 z-score 形式：

```text
mean = target.mean(dim=patch)
std  = target.std(dim=patch)
target' = (target - 0.8 * mean) / (std + 1e-6)
```

当前默认对齐损失是 `-cosine(projected_student, target')`，总损失为
`denoise_loss + 0.5 * alignment_loss`。这与 iREPA 的 cosine projection loss
和 `proj_coeff=0.5` 的数学意义一致，但不是完全相同的数值实现：

- iREPA 的 [`projection_loss.py`](../../iREPA/jit/projection_loss.py) 使用
  `F.normalize`（默认 `eps=1e-12`）后点积；当前使用 `F.cosine_similarity`
  （默认 `eps=1e-8`），接近零向量时两者的数值行为可能不同。
- 当前 z-score 和 cosine 前显式转换为 FP32；iREPA 的这两个 helper 没有同样的
  显式转换。两边 z-score 的公式、`alpha=0.8`、`std` 的默认 correction=1 和
  `1e-6` 平滑项一致，但混合精度计算路径不能宣称完全相同。

iREPA 的 parser 默认 `enc_type=dinov2-vit-b`、`encoder_depth=8`；当前使用
DINOv3-B teacher 和 JiT 第 4 个 block。实验表中的历史 `JiT+iREPA(DINOv1)`
是另一个具体运行配置，不能用仓库 parser 默认值代替其实际实验设置。

### 必要的 wrapper 改动

当前没有直接调用原始 `iREPA/jit/denoiser.py::Denoiser.forward`，原因是输入和
监督路径已经扩展：

1. DINOv3 student 先处理带噪图像，DINOv3 teacher 处理 clean image。
2. JiT 的 patch token 在进入 block 前由 native token 和 DINO token 融合。
3. 需要截取 JiT 第 4 个 block 的空间 token 做 alignment。
4. 原 JiT 的 final layer 被保留用于 checkpoint 兼容，但当前像素预测由 DDTHead 完成。
5. `z_dims=[]` 关闭 native iREPA model 内的注册式 projection head，避免与当前
   独立的 alignment `ConvProj` 重复。

所以，这部分是接口适配和模型结构扩展，并非改变 JiT 基础算子。

## 与 RAEv2 的关系

### DINOv3 MLS

[`RAEv2/src/encoders/vision_encoder.py`](../../RAEv2/src/encoders/vision_encoder.py)
中的 `DINOv3MultiLayerSimpleAddEncoder` 先取得选中层经最终 norm 处理的 token，
逐元素平均，再把最后一个选中层的 patch 空间均值广播加回去。
当前 [`dino_v3_lora.py`](../dino_v3_lora.py) 在 `aggregation="mls"` 时保持相同公式：

```text
layer_mean = mean(output[layer_i], dim=layer)
last_mean  = mean(output[last_layer], dim=patch, keepdim=True)
MLS        = layer_mean + last_mean
```

两边在 MLS 输出中都只保留 patch token，不把 CLS/register token 送入空间对齐；B/16 默认层也都
是 `[2,5,8,11]`。因此“多层聚合”这一点是直接跟随 RAEv2 的。

但完整 encoder 并不相同：

| 项目 | RAEv2 | 当前实验 | 是否有意改变 |
| --- | --- | --- | --- |
| DINO 模型 | 根据 RAEv2 encoder 配置加载 | DINOv3-B/16 本地权重 | 为 B-scale class-to-image 固定模型 |
| encoder 角色 | RAE 编码器在 `RAE.encode` 中通过 no-grad 提取 latent，不是当前 noisy student/clean teacher 双路 | student 使用 LoRA-r64，且 patch embed 可训练；teacher 冻结 | 是，student 需要适应 noisy image |
| teacher 输出 | MLS 可直接作为 RAE representation | teacher 只取最后一层 patch token，不做 MLS | 是，当前 alignment target 与 student 聚合分开 |
| noisy student 输入 | RAEv2 encoder 的标准预处理 | 当前配置关闭预处理，直接输入 `z_t` | 是，`NoDinoPre` 实验定义 |
| clean teacher 输入 | ImageNet 标准化 | ImageNet 标准化 | 保持预训练 target 的输入域 |
| 最终 LayerNorm | 默认替换成 `elementwise_affine=False`、`eps=1e-5`；指定 `[norm]` 才保留原 norm | student/teacher 都保留预训练 affine 参数并冻结，原 norm 的 `eps=1e-6` | 聚合前的数值差异，不是 residual 公式必需的改动 |

因此可以说“MLS 聚合跟随 RAEv2”，不能说整个 DINO encoder 完全相同。
当前 DINO 构建、LoRA 注入和角色区分分别见
[`dino_encoder.py`](../dino_encoder.py) 与 [`dino_v3_lora.py`](../dino_v3_lora.py)。

### DDTHead

参考文件为 [`RAEv2/src/stage2/models/DDT.py`](../../RAEv2/src/stage2/models/DDT.py)。
当前 [`jit_ddt_head.py`](../jit_ddt_head.py) 复用了以下设计：

- head 是 base representation 的浅层宽 head；
- condition 通过逐 token AdaLN 的 shift/scale/gate 注入；
- head 使用独立的 patch stream 和 head-local 2D RoPE；
- final layer 继续使用 AdaLN 后逐 patch 输出。

但它不是 RAEv2 DDT 的逐行复制，主要区别如下：

| 项目 | RAEv2 DDT | 当前 JiT DDTHead |
| --- | --- | --- |
| base stream | `s_embedder(x_t)` 加 time/context token，再过 RAEv2 base blocks | JiT-B 主干已经负责 patch、时间和类别条件 |
| head 输入 | RAE latent，由 `timm.PatchEmbed` 产生 head token | RGB noisy image，由 JiT `BottleneckPatchEmbed` 的 `3->128->2048` 产生 |
| condition | `s_projector(SiLU(spatial_token+t_emb_base))` | `z_proj(SiLU(jit_output+t_emb))`，`Linear(768->2048)`；运算顺序跟随 RAEv2 |
| head 宽度/深度 | 由 RAEv2 配置决定，例如 2048/2 | 当前固定为 2048/2/16 heads |
| attention | `NormAttention`：独立 Q/K/V Linear，均带 bias；PyTorch SDPA | JiT `Attention`：合并 QKV Linear，同样带 bias；显式 FP32 QK/softmax 计算 |
| FFN | SwiGLU 的输入两路分别为 `w1/w2` | JiT SwiGLU 合并为 `w12`；两者内部宽度均为 `int(2048*4*2/3)=5461` |
| RMSNorm/RoPE | RMSNorm 的归一化结果先转回输入 dtype 再乘 weight；RAEv2 2D RoPE | RMSNorm 先在 FP32 乘 weight 再转回输入 dtype；当前 16x16 网格 RoPE 公式等价，实现不同 |
| 初始化 | 普通 Linear（Q/K/V、FFN、`s_projector`）保留 PyTorch 默认初始化；显式处理 patch/time/class embedding、AdaLN 和 final | 所有 Linear 先 Xavier、bias 置零，patch conv Xavier，随后 AdaLN/final 置零 |
| 输出 | latent patch，随后恢复 latent | RGB patch `16*16*3=768`，随后用 JiT unpatchify 恢复图像 |
| 时间条件 | Gaussian Fourier embedding，base stream 有显式 time tokens | JiT sinusoidal embedding/global condition，加 head 的 `time_reinject`；没有额外 time tokens |

其中 RGB 通道、patch 尺寸和 JiT condition 接口属于像素模型适配；换用 bottleneck、
不同 attention 数值实现和整体 Xavier 初始化，则不是这种适配唯一必需的选择。
因此应描述为“RAEv2-style DDT head”，不能描述成“RAEv2 head 完全原样复用”。
QKV/FFN 合并参数是实现组织方式的差异，本身不代表新增了另一种 attention/FFN 公式。

当前构造函数明确传入 `time_reinject=True`；`jit_ddt_head.py` 中仍有一段描述
旧标准配置关闭该选项的历史 docstring，但实际运行配置以构造函数为准。这个
陈旧描述不改变当前 forward 行为。

## 与 PixelDiT2 的关系

### projector 已与官方实现对齐

当前 [`pixeldit2_projector.py`](../pixeldit2_projector.py) 来源于
[`PixelDiT2/pixdit_core/grounding.py`](../../PixelDiT2/pixdit_core/grounding.py)
和其依赖模块，使用本地版本
`8f83fa88a7b9dab575e5a45dafe3bb3c308685e0`。当前保留了：

- `input_proj -> TimestepConditioner -> ModulatedDiTBlock -> output_proj`；
- `max_period=10` 的时间 embedding；
- RMSNorm、Q/K norm、无 QKV bias、SwiGLU、PyTorch SDPA 和二维 complex RoPE；
- 官方的 linear 默认初始化和 adaLN modulation 末层置零。

当前 B 模型从 [`dino_jit_denoiser.py`](../dino_jit_denoiser.py) 传入
`hidden_size=768,num_heads=12,num_blocks=1`。这与 PixelDiT2-B/16 的主体配置
一致；projector 类本身的默认 head 数是 16，但当前调用会覆盖为 B 模型的 12。

对照测试见 [`tests/test_pixeldit2_projector.py`](../tests/test_pixeldit2_projector.py)。
本次命令为 `CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 python -m pytest -q
tests/test_pixeldit2_projector.py -k 'not cuda'`（使用 `irepa` 环境）。结果为
**3 passed, 2 deselected**：B 规格初始化与官方逐 tensor 完全相同；另两组 CPU
用例检查不同网格/深度的前向和梯度。2 个 CUDA 用例本次未运行，不据此宣称已
完成 GPU 或整个训练系统的等价验证。

### 当前使用方式与官方完整模型不同

PixelDiT2 官方完整模型在
[`PixelDiT2/pixdit_core/pixeldit2_c2i.py`](../../PixelDiT2/pixdit_core/pixeldit2_c2i.py)
中把 grounding token 与 timestep/class embedding 合成为每个主干 block 的
condition：

```text
c = DINO_grounding(x_t, t) + t_embedding + y_embedding
```

官方 `GroundingConditioner` 使用 frozen encoder 和 ImageNet 预处理，并把 projector
输出作为所有 block 的 per-token AdaLN condition。当前项目只复用 projector 本身：

```text
dino_tokens = DINOv3-LoRA-MLS(z_t)
projected   = PixelDiT2Projector(dino_tokens, t)
h_0         = (1-t) * native_jit_tokens + t * projected
```

因此以下行为是当前项目自己的结构选择：DINO student 的 LoRA/MLS、`NoDinoPre`、
projector 输出用于 JiT 初始 patch token、alignment teacher 以及 null-label 仍
forward DINO path。可选 `dino_grounding_drop_on_label_drop=true` 会令 null-label
样本改用完整 native token，而不是仅保留 `(1-t)*native`；当前开关为 false。
即使打开，该实现也是计算后掩码，不等于跳过 DINO forward 节省算力。
不能把当前模型描述为完整 PixelDiT2 reproduction。

## 我们真正新增的逻辑

1. **DINO representation 接入 JiT 输入。** 保留 native JiT token，同时引入 DINO
   token；当前配置用 PixelDiT2 projector 做 timestep-conditioned 映射。
2. **时间加权。** `t` 是 clean-image weight，当前 BlendT 对 native 和 projected
   DINO 两路使用 `1-t` 与 `t`；Residual 基线则使用 `h_native+t*r`。
3. **残差实验分支。** `gated_residual` 支持 `g(t)=t`、`g(t)=1` 和 null-label
   清零，用来验证时间门控和 unconditional representation 的作用。
4. **DINO 输入域消融。** `dino_preprocess_input` 可控制 noisy student 是否进行
   ImageNet 标准化；当前 NoDinoPre 配置关闭，teacher 不受影响。
5. **诊断指标。** 训练日志记录两路 token、加权 token、融合 token、alignment
   projector 输入/输出的范数和比值，用于检查 projector 是否改变表示尺度。
6. **离线评测。** 采样后保留 NPZ，并通过项目内
   [`jit/evaluation`](../jit/evaluation/README.md) 使用统一
   [`nanogen-evals`](../../nanogen-evals) 资产计算 FID、IS、FDR6。

## 评测路径的边界

- online evaluation：[`jit/engine_jit.py`](../jit/engine_jit.py) 的
  `online_evaluation_assets` 默认从
  `/data0/zhangruobei/Work/Pixel_Diffusion/fid_stats` 读取
  `jit_in256_stats.npz` 和 `weights-inception-2015-12-05-6726825d.pth`，只计算
  FID、IS。路径允许通过 `FID_STATS_ROOT` 覆盖。当前配置 Heun-50、`cfg=1`、
  interval `[0,1]`，每 20 epoch 和最终 epoch 用 EMA1（0.9999）评测 50K 图片。
- standalone offline evaluation：[`scripts/sample_and_evaluate_fdr6.sh`](../scripts/sample_and_evaluate_fdr6.sh)
  先从 `nanogen-evals` 检查资源，再采样生成 `samples.npz`，最后运行
  `evaluation.evaluate_fdr6` 计算 `fid,inception_score,fdr6`。独立 CFG 采样默认
  FlowDPMSolver-100、`cfg=2.5`、interval `[0,1]`、EMA1、50K 图片，具体参数写入
  sampling YAML。脚本的 `PYTHONPATH` 使模块解析到项目内 `jit/evaluation`。
  图片生成是多卡 torchrun，生成完后指标计算只使用指定卡组的第一张卡。
- offline 的网络权重从 `nanogen-evals/torch/hub/checkpoints` 和
  `nanogen-evals/huggingface/hub` 解析；参考统计值从 `nanogen-evals/stats` 下
  Hugging Face 数据集缓存的 `refs/main` 所指 snapshot 解析，亦支持显式覆盖。
- online/offline 的 FID 默认参考统计文件、IS 实现不完全相同，需标注评测协议，
  不应只因都叫 FID/IS 就把结果视为同一协议。

离线 [`distributional.py`](../jit/evaluation/distributional.py) 调用 vendored
`fd_evaluator`：FDR6 是 Inception、ConvNeXt、DINOv2、MAE、SigLIP、CLIP 六路
FD 各除以对应 `norm_const` 后取平均。wrapper 要求六路结果齐全且有限，避免
部分 extractor 缺失时仍报告一个不完整的 FDR6。它不进入训练 loss。
本次核实了这段计算流程，未对 vendored 包与外部上游做逐字节等价认证。

评测接入新增了资产路径解析、缓存配置、NPZ 导出和结果校验。该 shell 发现
`samples.npz` 已存在就跳过生成，因此每个 checkpoint、配置和 CFG scale 应使用
独立输出目录，避免把旧 NPZ 用于另一组配置；不能把“文件存在”理解成完整性认证。

采样器本身也有扩展：`_flow_dpm_solver_sample` 是首步 Euler、后续 AB2
`dt*(1.5*v-0.5*v_prev)`，100 步对应 100 次速度场计算；online 使用的 Heun-50
采用预测/校正，最后一步 Euler。这是独立采样实现的差异，不只是新增 FDR6 指标。

## 最小复现入口

训练配置和单文件启动脚本分别为：

- [`configs/BaseT-Residual(t,1-t)-NoDinoPre-PgT-B-80.yaml`](../configs/BaseT-Residual(t,1-t)-NoDinoPre-PgT-B-80.yaml)
- [`scripts/train_BaseT-Residual(t,1-t)-NoDinoPre-PgT-B-80_0123.sh`](../scripts/train_BaseT-Residual(t,1-t)-NoDinoPre-PgT-B-80_0123.sh)

脚本固定使用 `CUDA_VISIBLE_DEVICES=0,1,2,3`，不依赖 `runtime_env.sh`，也不设置
`NCCL_DEBUG`、`NCCL_IB_DISABLE`、`NCCL_P2P_DISABLE` 或
`TORCH_COMPILE_DISABLE`。

## 后续维护原则

- JiT 基础算子优先从 `iREPA/jit` 对齐，不在当前项目复制第二套实现。
- 若目标是尽量原样 follow，优先核对 MLS 最终 LayerNorm、DDT 初始化和 iREPA
  cosine/精度路径。这些差异都不由 residual 或 `t` 融合公式强制要求，当前保留
  不代表已通过消融证明有益；调整它们应作为明确的新配置，避免混入既有轨迹。
- DDT 后续可复用 RAEv2 的 `DDTDecoderBlock`、`DDTFinalLayer` 和基础算子，
  只在 RGB patch embed、JiT 输出接口、unpatchify 处保留适配。当前 QKV/FFN
  的参数组织与上游不同，复用时需要显式转换 checkpoint key，不能直接替换后
  宣称旧 checkpoint/optimizer 兼容。
- MLS 的 mean-plus-last-mean 公式保持复用；LoRA student 与 clean teacher 的
  职责应保留在外层 wrapper，不重复维护另一套 DINO 主干。
- projector 继续以 `PixelDiT2` 官方源码为准，更新时同步 commit、NOTICE 和
  对照测试。
- residual、`t` 权重、NoDinoPre 和 null-label 规则属于本项目实验变量，应在
  YAML、实验名和日志中同时体现。
- 评测时明确区分 `fid_stats` 的 online FID/IS 与 `nanogen-evals` 的离线
  FID/IS/FDR6，避免不同协议的指标直接比较。

本次只建立实现审计文档，没有调整模型、训练超参数或正在运行的实验。
