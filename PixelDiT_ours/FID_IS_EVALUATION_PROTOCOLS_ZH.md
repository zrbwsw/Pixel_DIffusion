# 三种 FID / IS 评测协议对比

本文只比较 **FID** 和 **Inception Score（IS）**，不讨论 sFID、Precision、Recall、FDR6 或其他指标。

## 结论

- **JiT 与 nanogen 的 FID 使用完全相同的参考统计和 Inception 权重，评测流程最接近，但不保证逐位一致。**
- **ADM 使用另一套参考统计和 TensorFlow Inception 图，属于 PixelDiT v1 官方协议。**
- **IS 不使用任何参考均值或协方差。** 三者的 IS 区别来自 Inception 实现、数值精度、样本是否打乱以及分组方式。
- 比较数值时必须使用同一份生成图像，并标明评测协议；不能把不同协议的 FID/IS 当作完全相同的指标复现。

## 总览

| 项目 | JiT / torch-fidelity | ADM 官方 evaluator | nanogen / fd_evaluator |
|---|---|---|---|
| FID 参考统计 | `fid_stats/jit_in256_stats.npz` | `fid_stats/VIRTUAL_imagenet256_labeled.npz` 的 `mu/sigma` | `nanogen-evals/.../imagenet_256_fid_stats.npz` |
| FID 参考统计关系 | 与 nanogen 完全相同 | 与 JiT/nanogen FID 统计不同 | 与 JiT 完全相同 |
| Inception 实现 | torch-fidelity、PyTorch `.pth` | TensorFlow `classify_image_graph_def.pb` | 与 JiT 相同的 torch-fidelity `.pth` |
| 生成样本输入 | PNG 目录 | NPZ 中的 `arr_0` | NPZ 中的 `arr_0` 或 `images` |
| FID 特征 | Inception pool 2048 维 | Inception `pool_3`，2048 维 | Inception pool 2048 维 |
| FID 数值实现 | 协方差乘积的特征值 | `scipy.linalg.sqrtm` | `scipy.linalg.sqrtm` |
| IS 是否使用参考统计 | 否 | 否 | 否 |
| IS 样本顺序 | 默认打乱，种子 2020 | 不打乱 | 打乱，种子 0 |
| IS 数值实现 | PyTorch float64 softmax/log_softmax | TensorFlow softmax + NumPy | PyTorch float32 softmax + NumPy |

## 参考 NPZ 的实际关系

已直接读取并逐项比较数组：

```text
fid_stats/jit_in256_stats.npz
    == nanogen-evals 的 imagenet_256_fid_stats.npz
    mu/sigma 完全相同

fid_stats/VIRTUAL_imagenet256_labeled.npz 的 mu/sigma
    != JiT / nanogen FID 使用的 mu/sigma
    差异不只是 float32/float64 类型差异
```

nanogen 目录还包含 `guided_diffusion_stats.npz`，其 `mu/sigma` 与 ADM 参考文件完全相同；但 **nanogen 输出的 `fid` 字段使用的是 `imagenet_256_fid_stats.npz`**，不是 `guided_diffusion_stats.npz`。因此比较三者的 FID 时，应以 `imagenet_256_fid_stats.npz` 为准。

`fid_stats` 和 nanogen 中的 `weights-inception-2015-12-05-6726825d.pth` 文件大小及 SHA256 完全相同。ADM 不使用这个 `.pth`，而是使用官方 TensorFlow 图文件。

### JiT 与 ADM 参考统计的实测差异

两份文件的 `mu/sigma` 确实不同，并非仅由 float32/float64 类型转换造成：

| 比较项 | JiT vs ADM |
|---|---:|
| `mu` 最大绝对差 | `0.0308858280` |
| `mu` 平均绝对差 | `0.0027063469` |
| `mu` 相对 L2 差异 | 约 `1.77%` |
| `sigma` 最大绝对差 | `0.0152054295` |
| `sigma` 平均绝对差 | `0.0001435457` |
| `sigma` 相对 L2 差异 | 约 `3.37%` |

将 ADM 的 `mu` 转为 float32 后，仍不与 JiT 的 `mu` 相等。文件结构也不同：

- JiT 的 `jit_in256_stats.npz` 只有 `mu[2048]`（float32）和 `sigma[2048,2048]`（float64），评估时直接读取，不重新统计参考数据集。
- ADM 的 `VIRTUAL_imagenet256_labeled.npz` 还包含 `mu_s/sigma_s`、`mu_clip/sigma_clip` 和 10000 张 `arr_0` 参考图；其 `mu/sigma` 均为 float64，评估时同样直接读取。

两者的目标统计量相同，均为参考 Inception 特征的均值和协方差：

\[
\mu=\frac1N\sum_i f(x_i),\qquad \Sigma=\operatorname{Cov}(f(x_i))
\]

区别在统计流水线：JiT 使用 torch-fidelity 的 PyTorch InceptionV3，ADM 使用 TensorFlow `classify_image_graph_def.pb` 的 `pool_3` 特征。ADM 官方文档说明参考 NPZ 包含整个参考数据集的预计算统计，附带的 10000 张 `arr_0` 用于 Precision/Recall，因此不能认为其 `mu/sigma` 只由这 10000 张图计算。

**本地没有生成 `jit_in256_stats.npz` 的脚本**，所以无法仅凭当前文件确认两份统计是否使用完全相同的参考图像集合，也不能把差异全部归因于 TensorFlow/PyTorch 实现。能够确认的是：参考统计数值、Inception 实现和存储精度均不同。该差异只影响 FID；IS 不读取这些参考 `mu/sigma`。

## FID 计算流程

三者都计算：

$$
\mathrm{FID}=\|\mu_g-\mu_r\|_2^2+
\operatorname{Tr}\left(\Sigma_g+\Sigma_r-2(\Sigma_g\Sigma_r)^{1/2}\right)
$$

其中 \(g\) 是生成样本，\(r\) 是参考数据。

### `pool` 与 `pool_3` 特征的区别

本文中的 `pool` 是 torch-fidelity 2048 维池化特征的简写，`pool_3` 是 ADM TensorFlow 图中的张量名称。两者在 FID 中的作用相同，并不是两种不同语义的特征：

- **torch-fidelity 的 `2048` / pool 特征**：Inception 最后的卷积特征经过 `AdaptiveAvgPool2d(output_size=(1,1))`，在空间维做全局平均池化，再展平为 `[N,2048]`。
- **ADM 的 `pool_3:0`**：TensorFlow Inception 最终全局平均池化层的输出，ADM 将其展平为 `[N,2048]`。名称后的 `:0` 表示该操作的第一个输出张量。

因此，这里不是“平均池化与最大池化”的区别，也不是一个用于 FID、另一个用于其他任务的区别：**两者都是分类层之前的全局平均池化表征，均用于 FID 的均值和协方差统计。** 名称差异来自 PyTorch/torch-fidelity 与 TensorFlow 的接口。IS 也会在该池化特征后接分类权重得到类别概率；它不使用 FID 的参考均值或协方差。

### JiT / torch-fidelity

代码：`iREPA/jit/engine_jit.py:198`。

1. 将生成图像保存为 PNG。
2. torch-fidelity 读取 PNG，以 uint8 RGB 图像为输入。
3. 使用 `inception-v3-compat`，按 TensorFlow 兼容方式缩放到 299×299。
4. 提取生成图像的 2048 维特征。
5. 使用 `np.mean` 和 `np.cov(..., rowvar=False)` 计算生成样本的均值、协方差。
6. 从 `jit_in256_stats.npz` 直接读取参考 `mu/sigma`，不重新提取参考图像特征。
7. 使用 `np.linalg.eigvals(sigma1.dot(sigma2))` 的平方根计算 Fréchet Distance，见 torch-fidelity 的 `metric_fid.py:32`。

### ADM 官方 evaluator

代码：`guided-diffusion/evaluations/evaluator.py:44`。

1. 从生成 NPZ 的 `arr_0` 逐批读取 uint8 NHWC RGB 图像。
2. 使用 TensorFlow `classify_image_graph_def.pb` 提取 `pool_3` 的 2048 维特征。
3. 若生成 NPZ 没有预计算统计，则计算生成样本的 `mu/sigma`。
4. 从 `VIRTUAL_imagenet256_labeled.npz` 读取参考 `mu/sigma`。
5. 使用 `scipy.linalg.sqrtm(sigma1.dot(sigma2))` 计算 FID；出现非有限值时加入 `1e-6` 对角偏移，见 `evaluator.py:72`。

### nanogen / fd_evaluator

代码：`PixelDiT_ours/tools/evaluation/vendor/fd_evaluator/api.py:43`。

1. 从生成 NPZ 读取 uint8 NHWC RGB 图像。
2. 使用与 JiT 相同的 torch-fidelity InceptionV3 提取 2048 维特征。
3. 使用 float64 累加特征和与外积，再计算生成样本均值、协方差，见 `frechet.py:10`。
4. 从 `imagenet_256_fid_stats.npz` 读取参考 `mu/sigma`；该文件与 JiT 统计完全相同。
5. 使用 `scipy.linalg.sqrtm` 计算 FID，见 `frechet.py:43`。

**FID 与样本排列顺序无关。** 在生成像素完全相同的前提下，JiT 与 nanogen 使用相同参考统计和 Inception 权重，结果应当接近；差异主要来自协方差累计和矩阵平方根的数值实现。ADM 的参考统计和 Inception 实现均不同，因此是另一种协议。

## IS 计算流程

IS 只使用生成图像的分类概率：

$$
\mathrm{IS}=\exp\left(\mathbb E_x\left[D_{\mathrm{KL}}(p(y|x)\|p(y))\right]\right)
$$

参考 `mu/sigma` 不参与 IS。实际实现通常将样本分组，分别计算后取平均。

### JiT / torch-fidelity

代码：torch-fidelity 的 `metric_isc.py:16`，由 `iREPA/jit/engine_jit.py:198` 调用。

1. 使用 Inception 的 `logits_unbiased`。
2. 默认以随机种子 **2020** 打乱生成样本。
3. 转为 float64，计算 softmax 和 log_softmax。
4. 分成 10 组，使用 `i*N//splits` 确定边界，因此所有样本都会保留。
5. 计算每组 IS 后取均值；引擎记录均值。

### ADM 官方 evaluator

代码：`guided-diffusion/evaluations/evaluator.py:179`。

1. 从 Inception pool 特征和分类权重计算 softmax，见 `evaluator.py:603`。
2. **不打乱样本**，保留 NPZ 原始顺序。
3. 默认每 5000 张分为一组；50K 样本对应 10 组。
4. 使用 NumPy 计算每组 KL/IS，再取均值。

因此 ADM 的 IS 可能受到生成 NPZ 中样本排列方式影响，尤其是按类别连续排列的 NPZ。

### nanogen / fd_evaluator

代码：`PixelDiT_ours/tools/evaluation/vendor/fd_evaluator/api.py:332`。

1. 使用与 JiT 相同的 `logits_unbiased`。
2. 以随机种子 **0** 打乱样本。
3. 使用 float32 softmax，再转为 NumPy 计算。
4. `chunk = N // 10`，每组取 `chunk` 张；50K 时每组同样是 5000 张。
5. 对数计算加入 `1e-10`，最后取 10 组均值。

所以即使 JiT 和 nanogen 使用相同 Inception 权重，**IS 也不保证完全相同**：两者的打乱种子和数值精度不同。ADM 还不打乱样本，协议差异更明显。

## 对齐建议

- 对齐 **PixelDiT v1 官方结果**：使用 ADM 的 FID/IS，并保留 ADM 参考 NPZ、TensorFlow Inception 和生成 NPZ 的原始样本顺序。
- 对齐之前 **JiT 实验**：使用 `iREPA/jit/engine_jit.py` 的 torch-fidelity FID/IS 及 `jit_in256_stats.npz`。
- 使用 **nanogen FID/IS**：可以视为与 JiT 最接近的离线实现，但仍应标注为 nanogen 协议，不宣称与 JiT 或 ADM 逐位一致。
- 正式比较时固定为同一批 **50K、uint8、RGB、NHWC** 生成图像，同时记录样本数量、随机种子、样本顺序和评测代码版本。

## 项目内的三个入口

在 `PixelDiT_ours` 根目录执行。

### JiT / torch-fidelity

```bash
NPZ=/path/to/output.npz bash scripts/evaluate_npz_jit_fid_is.sh
```

实现为 `tools/evaluate_jit_fid_is.py`，直接读取 NPZ，默认输出 `jit_fid_is.json` 和 `jit_fid_is.log`。已用同一批图像与原始 `torch_fidelity.calculate_metrics` PNG 调用对照：IS 完全一致，FID 仅有浮点误差。

### nanogen / fd_evaluator

```bash
NPZ=/path/to/output.npz METRICS=fid,inception_score \
  bash scripts/evaluate_npz_fid_is_fdr6.sh
```

只计算 FID/IS 时设置上述 `METRICS`；默认输出 `offline_metrics.json` 和 `offline_metrics.log`。

### ADM 官方 evaluator

```bash
REPO=/mnt/shared-storage-user/datafactory-share/gushishen/agent-z/Pixel_DIffusion
cd "$REPO/guided-diffusion/evaluations"
python evaluator.py \
  "$REPO/fid_stats/VIRTUAL_imagenet256_labeled.npz" \
  /path/to/output.npz
```

ADM 需要其 `requirements.txt` 中的 TensorFlow 环境，并会在工作目录准备 `classify_image_graph_def.pb`。
