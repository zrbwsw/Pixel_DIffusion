# DINO-LoRA 前端的数值坐标与生成化改造

本文只讨论两个直接影响当前结构的问题：

1. pixel flow 加噪后，送入 DINO 前执行 `+1`、`/2` 和 ImageNet mean/std 到底表示什么；
2. 如何把 DINO-LoRA 改造成真正适用于生成的模块：作为 gated representation residual，以及在 DINO 内部注入时间和类别条件。

当前代码位置：

- `jit/engine_jit.py`：DataLoader 图像转为 `[-1,1]`；
- `dino_jit_denoiser.py::forward`：pixel flow 加噪；
- `dino_jit_denoiser.py::_preprocess_for_dino`：`[-1,1] -> [0,1]` 坐标变换与 ImageNet mean/std；
- `dino_v3_lora.py::DINOv3PatchTokenEncoder`：DINOv3-LoRA normalized patch token 与 MLS 输出。

## 1. Pixel flow 到 DINO 输入的完整数值变换

### 1.1 当前 pixel flow

设 DataLoader 解码后的干净图像为

$$
x_{01}\in[0,1].
$$

训练循环先做线性范围映射：

$$
x=2x_{01}-1,
\qquad x\in[-1,1].
$$

当前时间变量 `t` 是 clean-image weight：

- $t=0$：纯噪声；
- $t=1$：干净图像。

采样标准高斯噪声

$$
\epsilon\sim\mathcal N(0,I),
$$

并在 `[-1,1]` pixel 坐标中构造直线路径：

$$
z_t=t x+(1-t)\epsilon.
$$

对应代码为：

```python
x = image.float().div(255.0) * 2.0 - 1.0
epsilon = torch.randn_like(x)
z_t = t * x + (1.0 - t) * epsilon
```

`z_t` 是 DDT head 使用的 diffusion state。由于 $\epsilon$ 无界，`z_t` 通常也无界；所谓 `[-1,1] pixel 坐标` 描述的是干净图像端点的坐标约定，不表示所有 noisy state 都落在 `[-1,1]` 内。

### 1.2 DINO 前的 `+1`、`/2` 等价于什么

DINO student 不直接接收 `z_t`，而是先执行：

$$
z_{01,t}=\frac{z_t+1}{2}.
$$

代入 $z_t=t x+(1-t)\epsilon$ 和 $x=2x_{01}-1$：

$$
\begin{aligned}
z_{01,t}
&=\frac{t x+(1-t)\epsilon+1}{2}\\
&=\frac{t(2x_{01}-1)+(1-t)\epsilon+1}{2}\\
&=\frac{2t x_{01}-t+(1-t)\epsilon+1}{2}\\
&=t x_{01}+(1-t)\frac{\epsilon+1}{2}.
\end{aligned}
$$

定义经过相同坐标变换的噪声端点：

$$
\epsilon_{01}=\frac{\epsilon+1}{2},
$$

则有

$$
\boxed{z_{01,t}=t x_{01}+(1-t)\epsilon_{01}}.
$$

因此，当前操作

```text
clean image [0,1]
-> 映射到 [-1,1]
-> 与标准高斯噪声插值
-> DINO 前映射回 [0,1] 坐标
```

严格等价于

```text
clean image [0,1]
-> 与经过同一仿射坐标变换的 Gaussian endpoint 插值
```

这是一般的仿射等价性。对于 $A(u)=au+b$，因为 $t+(1-t)=1$：

$$
A\!\left(tu+(1-t)v\right)=tA(u)+(1-t)A(v).
$$

### 1.3 `[0,1]` 坐标下的噪声分布

由于

$$
\epsilon\sim\mathcal N(0,1),
\qquad
\epsilon_{01}=\frac{\epsilon+1}{2},
$$

所以

$$
\boxed{\epsilon_{01}\sim\mathcal N(0.5,0.5^2)}.
$$

即均值为 `0.5`、标准差为 `0.5`、方差为 `0.25`。它仍然是无界 Gaussian，不是被限制在 `[0,1]` 的随机变量。纯噪声时：

$$
P(\epsilon_{01}<0\ \text{or}\ \epsilon_{01}>1)
=P(|\epsilon|>1)
\approx31.7\%.
$$

这解释了为什么默认不 clamp 时，noisy DINO student 输入可能明显超出自然图像范围。`[0,1]` 在这里应称为**像素坐标约定**，不能称为有界取值范围。

不能把等价路径误写成

$$
z'_{01,t}=t x_{01}+(1-t)\epsilon,
\qquad \epsilon\sim\mathcal N(0,1).
$$

后者把纯噪声端点从 $\mathcal N(0.5,0.5^2)$ 改成 $\mathcal N(0,1)$，会同时改变 flow path 的均值、方差、velocity 尺度和采样初值；这不是简单地交换操作顺序。

### 1.4 ImageNet mean/std 后的等价形式

送入 DINO 前还会逐 RGB 通道执行：

$$
d_t=\frac{z_{01,t}-\mu}{\sigma},
$$

其中

$$
\mu=(0.485,0.456,0.406),
\qquad
\sigma=(0.229,0.224,0.225).
$$

代入前面的插值：

$$
\begin{aligned}
d_t
&=\frac{t x_{01}+(1-t)\epsilon_{01}-\mu}{\sigma}\\
&=t\frac{x_{01}-\mu}{\sigma}
 +(1-t)\frac{\epsilon_{01}-\mu}{\sigma}.
\end{aligned}
$$

所以 ImageNet mean/std 也是一个逐通道仿射变换，它与 linear flow interpolation 交换。DINO 实际看到的是：

```text
ImageNet-standardized clean image
与
ImageNet-standardized transformed Gaussian endpoint
之间的线性插值
```

纯噪声端点经过 ImageNet mean/std 后，各通道分布为：

$$
d_{\epsilon,c}\sim
\mathcal N\!\left(
\frac{0.5-\mu_c}{\sigma_c},
\left(\frac{0.5}{\sigma_c}\right)^2
\right).
$$

它的标准差约为 `2.18--2.23`，明显宽于 DINO 预训练时的自然图像输入分布。当前数值变换在代数上正确，但不能消除 noisy image 对 DINO 的 OOD 问题。

### 1.5 统一术语

后续文档和代码应按下表区分这些操作：

| 操作 | 建议名称 | 数学形式 | 是否产生固定范围 |
| --- | --- | --- | --- |
| `[0,1] -> [-1,1]` 或逆变换 | **线性范围映射 / affine rescaling** | `2x-1`、`(x+1)/2` | 只对原本有界的 clean image 保持相应范围 |
| ImageNet mean/std | **逐通道标准化 / ImageNet standardization** | `(x-mu)/sigma` | 否 |
| 任意变量转均值 0、方差 1 | **统计标准化 / standardization** | `(x-E[x])/Std[x]` | 否 |
| `torch.randn_like` | **标准高斯采样** | `epsilon ~ N(0,I)` | 否；这里没有额外执行“高斯标准化” |
| DINO `norm=True` | **feature LayerNorm** | 对 token 的 channel hidden state 做 LayerNorm | 否 |
| cosine 前的 `F.normalize` | **L2 normalization / 单位范数化** | `h / ||h||_2` | 只把向量范数变成 1 |
| 将 standardized latent 恢复原统计量 | **反标准化 / de-standardization** | `z*sigma+mu` | 否 |

工程中常把 ImageNet standardization 叫作 `normalize`，但讨论设计时应明确它是 mean/std **标准化**，而不是 `[0,1]` 范围**归一化**。

### 1.6 对当前实现的判断

当前

```text
clean image -> [-1,1] -> pixel flow 加噪
-> DINO 前转回 [0,1] 坐标 -> ImageNet mean/std
```

在数学上自洽，不需要改成“先在 `[0,1]` 中直接混合标准高斯”。真正的问题是：

1. 低 `t` 时 DINO 接收的仍是大幅偏离自然图像分布的输入；
2. 当前 DINO blocks 内部不知道 `t`，也不知道条件/空标签 `y`；
3. 历史结构用深层 DINO token 硬替换 native JiT PatchEmbed token，丢失了稳定的 noisy-pixel 主路径；
4. DINO MLS token 与 native JiT token 的尺度和统计分布不同。

DINO 输出是 `[B,N,D]` feature token，不能对其执行 RGB ImageNet mean/std 的“反归一化”。RAEv2 只有在固定 Stage-1 decoder 要求 raw latent 分布时才需要对 latent de-standardize；我们的 JiT/DDTHead 不存在这一接口约束。当前需要的是 feature scale adaptation，而不是反标准化。

## 2. 应更新的两项核心设计

### 2.1 DINO-with-LoRA 作为 gated representation residual

不再让 DINO token 替换 native JiT token。主生成路径始终保留 JiT 对 noisy pixels 的原生 PatchEmbed：

```text
z_t -> native JiT PatchEmbed + position -----------------------+
                                                               +-> conditioned JiT -> DDTHead
z_t -> DINOv3-LoRA -> MLS -> Norm -> residual adapter -> gate -+
```

建议形式为：

$$
h_{\mathrm{native}}=E_{\mathrm{JiT}}(z_t)+p,
$$

$$
f_{\mathrm{DINO}}=F_{\mathrm{DINO}}(d_t),
$$

$$
r=W_2\!\left(\operatorname{SiLU}
  \left(W_1(\operatorname{LN}(f_{\mathrm{DINO}}))\right)\right),
$$

$$
\boxed{h_0=h_{\mathrm{native}}+g(t,y)\,r}.
$$

实现约束：

- `residual adapter` 必须与 iREPA/alignment projector 分开；cosine alignment 不约束 projector 输出模长，不能把 alignment ConvProj 输出直接加到生成主路径；
- residual adapter 的最后一层零初始化，或将 gate 初始化为接近 0，使训练起点退化为稳定的 native JiT；
- adapter 前使用 LayerNorm/RMSNorm，解决 DINO MLS 与 native JiT token 的尺度差异；这里不做 de-standardization；
- 当前 `t` 是 clean weight，第一版可使用 `g(t)=t` 或 `g(t)=t^2`，在纯噪声端抑制不可靠的 DINO residual；
- 后续可改为由 `t/y` 预测的 scalar、channel-wise 或 token-wise learned gate，但仍保持接近零的初始化；
- native noisy-pixel stream 必须始终存在，DINO 只提供可关闭的增量表征；
- clean frozen DINO teacher 和 alignment loss 继续作为独立训练支路，不能与 residual adapter 共享输出层。

这项修改直接解决历史结构的不可逆瓶颈：即使 DINO 在高噪声处提取失败，JiT/DDTHead 仍然可以从原始 `z_t` 完成生成。

### 2.2 在 DINO 内部从 block 0 注入 `t` 和 `y`

门控只能降低错误 DINO feature 的影响，不能让 DINO 自身理解当前生成状态。若要让 DINO-LoRA 真正产生有用的生成表征，需要在第一个 DINO block 前注入时间和类别条件。

建议构造：

$$
\Delta_t=W_t(M_t(t)),
\qquad
\Delta_y=W_y(E_y(y_{\mathrm{drop}})).
$$

其中 `y_drop` 必须与 JiT 使用同一份 label-dropout 结果，保证 conditional/null 两条 CFG 分支在 DINO 与 JiT 中一致。

首版 token 方案：

```text
noisy z_t
-> DINO PatchEmbed
-> pretrained CLS + projected class condition
-> storage/register tokens + projected time condition
-> DINO block 0 ... block L-1 with LoRA
-> discard all non-spatial tokens
-> keep [B,256,D] spatial patch tokens
-> MLS / residual adapter
```

#### 2.2.1 原版 DINOv3 的 CLS 与 register/storage token

这里必须区分三类对象：预训练输入参数、逐图像产生的 token activation，以及把条件映射成 token 的可训练层。原版 DINOv3-B/16 在 $256\times256$ 输入下先通过 `patch_embed.proj`（`Conv2d(3,768,kernel_size=16,stride=16)`）得到 $16\times16=256$ 个 patch token：

$$
p(x)=E_{\mathrm{patch}}(x)\in\mathbb{R}^{B\times256\times768}.
$$

模型还直接注册了两组与图像无关的可学习输入参数：

$$
e_{\mathrm{cls}}\in\mathbb{R}^{1\times1\times D},
\qquad
e_{\mathrm{reg}}\in\mathbb{R}^{1\times4\times D}.
$$

源码中的名字分别是 `model.cls_token` 和 `model.storage_tokens`；MODEL_CARD 将后者称为 4 个 register tokens，因此本文把 register/storage 视为同一组 token。它们在模型初始化时由 $\mathcal N(0,0.02^2)$ 初始化，随后随 DINOv3 自监督预训练一起学习，并最终保存在预训练 checkpoint 中。forward 时只是沿 batch 维复制，再与 patch token 拼接：

$$
H^{(0)}=
[e_{\mathrm{cls}},e_{\mathrm{reg},1},\ldots,e_{\mathrm{reg},4},p_1(x),\ldots,p_{256}(x)]
\in\mathbb{R}^{B\times261\times D}.
$$

- **CLS input parameter**：所有图像共享同一个初始向量 $e_{\mathrm{cls}}$，没有空间位置。这里的 CLS 是用于聚合全局表征的 classification token，并不是 ImageNet 类别标签 $y$ 的 embedding。它经过 self-attention 后形成依赖当前图像的 $h_{\mathrm{cls}}^{(l)}(x)$；DINOv3 预训练的 DINO head 作用于最终 CLS activation。
- **Register/storage input parameters**：4 个共享的非空间参数向量，为 Transformer 提供额外的全局信息存储位置。它们不对应任何图像 patch，也不应进入最终的 $16\times16$ spatial grid；其具体功能由预训练目标和 self-attention 自发学习，不能把 4 个位置解释成预先指定的 4 种语义。
- **Patch tokens**：`patch_embed.proj` 含有卷积权重和 bias；$p_i(x)$ 是该映射对当前图像产生的 activation，而不是独立参数。
- **Block outputs**：$h_{\mathrm{cls}}^{(l)}(x)$、$h_{\mathrm{reg}}^{(l)}(x)$ 和 $h_{\mathrm{patch}}^{(l)}(x)$ 都是经过第 $l$ 个 Transformer block 后的逐样本 activation，本身不是 `nn.Parameter`；真正的参数位于初始 CLS/storage embedding、PatchEmbed、attention、MLP、normalization 和 LoRA 等模块中。

因此，“真正有参数的只有映射层，token 只是映射结果”只适用于我们准备新增的动态条件 token，不适用于原版 DINOv3 的 CLS/register 输入。上式中的 $E_y$、$M_t$、$W_y$ 和 $W_t$ 持有可训练参数，而 $\Delta_y$、$\Delta_t$ 以及 $e_{\mathrm{cls}}+\Delta_y$ 是随样本变化的 activation。原版 $e_{\mathrm{cls}}$ 和 $e_{\mathrm{reg}}$ 仍然是 checkpoint 中已有的预训练参数。

当前 `DINOv3PatchTokenEncoder` 会先冻结完整 DINOv3，再打开 LoRA 和 `patch_embed`；默认 `dino_train_cls_token=false`，而 storage tokens 没有重新解冻。因此在当前 J18/J29 系列中，PatchEmbed Conv 和 LoRA 是可训练的，预训练 CLS/register 参数保持冻结。当前 wrapper 最终调用 `return_class_token=False`，只把 $[B,256,D]$ patch activations交给 JiT/alignment；CLS/register 虽参与 DINO block 内部计算，但不会直接进入 JiT。

更具体地：

- 不建议完全覆盖预训练 CLS 参数，优先使用 `cls_pretrained + Delta_y`，保留预训练初始化；
- 可将 `Delta_t` 加到已有 storage/register tokens，或新增独立 time token；
- time/class/storage/CLS 都属于 non-spatial prefix，不应混入最终 16x16 patch grid；
- DINOv3 spatial RoPE 只对应 256 个 patch token，新增 non-spatial token 必须使用 identity/no-rotation 位置处理；
- student 输出时严格切掉 prefix，只返回 `[B,256,D]`，保证与 native JiT grid 和 clean teacher patch target 一一对应；
- conditional 与 null-label 采样必须都执行 conditioned DINO forward，不能在 DINO 前共享同一份 feature，否则 CFG 的类别差分仍只能在后续 JiT 中形成；
- LoRA 继续作用于 qkv/proj/MLP，使预训练 DINO blocks 能在 `t/y` 条件下适配 noisy-image 分布。

建议按单因素顺序验证：

1. `native JiT + gated DINO residual`，先检验 DINO 是否能作为受控增量；
2. 在相同 residual 结构上加入 DINO `time-only`，检验噪声级别条件；
3. 再加入 `time+class`，检验 CFG 条件方向；
4. 最终推荐结构同时采用 gated residual 与 DINO `t/y` 注入，而不是重新回到 DINO token 硬替换 JiT token。
