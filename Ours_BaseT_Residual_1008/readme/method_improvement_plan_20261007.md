# BaseT / Residual 方法改进分析与实验优先级

分析日期：2026-10-07。训练诊断快照取北京时间 21:39 附近的日志，离线结果核对至 21:43。
这是改进建议，不是已经实施的代码修改；本文没有改变配置、运行中的任务或实验表。

## 1. 先给结论

**短期最值得做的是：先补齐现有模型的可比评测，再分别验证输入统计与两路融合幅度。**
不要同时叠加新的 projector、dropout、门控和对齐损失，当前证据还不足以确定它们各自的作用。

| 优先级 | 实验 | 推荐原因 |
| --- | --- | --- |
| P0 | 当前 NoDinoPre-PgT-BlendT 的首次 20-epoch online / CFG 评测 | 目前只有训练诊断，不能判断新结构的生成质量 |
| P0 | DinoT4Y8 的 80-epoch checkpoint 补 CFG 2.1、2.3 | no-CFG 改善明显，但 CFG 2.5 未体现 FID 优势；先排除 scale 不适配 |
| P0 | 完成已有 BaseT / DinoTNorm / Residual 的 CFG 2.5 指标队列 | 统一 FID/IS/FDR6 协议，避免用旧记录评价微小提升 |
| P1-1 | 当前 PgT-BlendT 加一个外部幅度校准对照 | 加权 DINO 路范数仍约为 native 路的 6.21 倍，直接检验融合幅度假设 |
| P1-2 | 当前 PgT-BlendT 的 raw 输入改为 DinoTNorm 解析归一化 | 这是已有同协议结果中更有依据的改进方向，且不增加可训练参数 |
| P2 | 标准 ImageNet 预处理对照、native-only + 同款 DDTHead 对照 | 分别厘清 NoDinoPre 的作用和 DINO frontend 的真实收益 |
| P2 | 保留完整 native 主干，用 PgT 输出作逐 token AdaLN 条件 | 更接近 PixelDiT2 的关键机制，但改动和验证成本更大 |
| P3 | 再研究 grounding dropout、门控曲线、LoRA / MLS 简化 | 暂无证据支持优先扩大这些方向 |

**若接下来只新增两个训练实验，我推荐 P1-1 和 P1-2，各自只改一项，先检查 20 epoch，再决定是否推进至 40/80。**
当前运行的实验应作为两者的对照，不因范数较大就立即停掉或改变它。

## 2. 已有结果说明什么

### 2.1 80-epoch 的 online no-CFG 结果

下面来自 [实验表](../../related_tex/Pixel-RAE_tex/exp0926.tex)，使用 heun50、CFG=1、50K 和 EMA1。
这里的 no-CFG 是**仍提供真实类别标签，只是不做 CFG 放大**，不是 null-label 无条件生成。

| 方法 | Diffloss | FID | IS |
| --- | --- | --- | --- |
| BaseT | 0.0642 | 37.76 | 44.39 |
| DinoT4Y8 | 0.0643 | 25.68 | 58.89 |
| DinoTNorm | 0.0642 | 29.47 | 51.23 |
| Residual | 0.0640 | 39.20 | 43.82 |

DinoT4Y8 和 DinoTNorm 改善了不加 guidance 的类别条件生成；Residual 的去噪损失更低，
但 online FID 并没有更好。这已经说明：**Diffloss、alignment cosine 都不能替代生成指标。**
online 与离线 CFG 的 sampler、指标实现/参考统计并不完全相同，不能把两列数值的差直接解释为 CFG 的纯粹贡献。

### 2.2 同一离线协议下的 CFG 结果

下面核对各实验的 `offline_metrics.json`，统一 flow100、interval `[0,1]`、50K 和 nanogen-evals。
FID/IS/FDR6 分别看低/高/低；数值展示保留两位，比较幅度使用未四舍五入的原始结果。

| 方法 | Scale | FID | IS | FDR6 |
| --- | --- | --- | --- | --- |
| BaseT | 2.1 | 5.62 | 231.43 | 19.01 |
| BaseT | 2.3 | 5.17 | 263.61 | 17.67 |
| BaseT | 2.7 | 5.40 | 311.19 | 16.12 |
| BaseT | 2.9 | 5.80 | 332.04 | 15.63 |
| DinoTNorm | 2.1 | 4.90 | 238.65 | 16.16 |
| DinoTNorm | 2.3 | 4.89 | 268.63 | 15.11 |
| DinoTNorm | 2.7 | 5.70 | 310.72 | 14.00 |
| DinoTNorm | 2.9 | 6.30 | 329.26 | 13.67 |
| Residual | 2.1 | 5.48 | 239.42 | 17.56 |
| Residual | 2.3 | 5.14 | 273.70 | 16.09 |
| Residual | 2.7 | 5.68 | 321.07 | 14.52 |
| DinoT4Y8 | 2.5 | 5.96 | 298.28 | 13.62 |

BaseT / DinoTNorm 的 scale 2.9 文件已经存在，虽然本次查看的 `.tex` 仍留空；
Residual 2.9 的最终 `offline_metrics.json` 尚未存在，不补猜测值。
三个方法的 scale 2.5 全指标重测还在 `ours_cfg2p5_metrics_4567` 队列中，本文不把早期 FID/IS 当成已完成的新协议结果。

可以作出的判断：

- **DinoTNorm 当前最有改进依据。**同 scale 2.3 相对 BaseT，FID 下降约 5.49%，FDR6 下降约 14.53%，IS 也提高。仍需额外 seed 确认稳定性。
- **Residual 有收益，但更偏 IS / FDR6。**同 scale 2.3 的 FID 仅下降约 0.56%，不足以据单次实验宣称明确领先。
- **不能只固定 scale 2.5 判断结构。**DinoTNorm 在 2.1/2.3 的 FID 比 2.7/2.9 更好；DinoT4Y8 值得先补低 scale，而不是直接弃用。
- **FID 与 FDR6 有真实权衡。**提高 scale 后，IS 上升、FDR6 下降，但 FID 变差。应同时报告固定 scale 和各自最佳 scale，而不是只挑一个指标。
- FDR6 平均值也会掩盖差异：scale 2.3 的 Residual 在 DINOv2 特征上的距离为 10.06，BaseT 为 9.69，尽管其平均 FDR6 更低；不能称其在所有表示空间都改善。

### 2.3 旧 Residual 消融给出的限制

20 epoch、CFG 2.5、flow100 的历史结果为：

| 方法 | FID | IS |
| --- | --- | --- |
| Residual，`g(t)=t` | 7.89 | 184.94 |
| G1，`g(t)=1` | 8.51 | 174.21 |
| DropRes，label-drop 同时丢残差 | 9.53 | 118.66 |

现有证据支持暂时保留时间门控，而不优先再跑 G1 / DropRes 的完整 80 epoch。
这只是旧结构、20 epoch、固定 scale 下的结论，不是所有 grounding dropout 或所有采样设置都无效。

## 3. 当前 PgT-BlendT：已改善什么，尚未解决什么

### 3.1 当前实际结构

当前配置为 [NoDinoPre-PgT-BlendT YAML](../configs/BaseT-Residual(t,1-t)-NoDinoPre-PgT-B-80.yaml)。

```text
z_t = t*x + (1-t)*noise                 # t=0 为噪声，t=1 为干净图像
n   = JiT_BottleneckPatchEmbed(z_t)
s   = DINOv3-LoRA-MLS(z_t)              # 仅 noisy student 关闭预处理
p   = PixelDiT2_TimestepProjection(s,t)
h   = (1-t)*n + t*p                     # 随后只加一次固定位置编码
h -> JiT -> RAEv2-style DDTHead -> x_pred
loss = velocity_MSE + 0.5*alignment_loss
```

clean teacher 仍做 ImageNet 标准化，输出最后一层；alignment 仍是 JiT block-4 后独立的 3x3 ConvProj。
null-label 分支仍 forward DINO，不开启 grounding drop。DDTHead 还有自己的 noisy RGB 输入，
所以 native 主干 token 权重变小不等于整个网络失去了像素输入。

与旧 Residual 相比，这次同时改了输入预处理、输入 projector 和融合公式；
可训练参数从 308.843M 增到 320.253M，增加约 3.69%。**现有对比不能把效果单独归因于 PgT。**
PgT 本体已对齐官方源码，但其使用位置不是官方的条件调制，详见 [实现审计](implementation_comparison_irepa_raev2_pixeldit2.md)。

### 3.2 匹配训练进度后的诊断

快照时新方案已完成 19 epoch，约在 19.26 epoch；首次 20-epoch online eval 尚未完成。
去噪损失采用 `train.log` 每个 epoch 最后一条的括号内整 epoch running average：

| 已完成 epoch | 旧 Residual | 新方案 |
| --- | --- | --- |
| 1 | 0.1833 | 0.1768 |
| 2 | 0.0858 | 0.0830 |
| 6 | 0.0696 | 0.0694 |
| 12 | 0.0666 | 0.0668 |
| 19 | 0.0657 | 0.0659 |

第 19 epoch 的 x-MSE 为旧方案 0.0317、新方案 0.0316。早期收敛更快，但近期去噪损失基本接近，
不能仅凭这点宣布改进或退步。总损失含负 cosine 项，也不能把总损失更低当成去噪更好。

下面使用两组**第 19 个完整 epoch**的 JSONL 诊断采样，各 52 个点。
旧日志先按 `global_iter` 去重并保留最后一条；它们不是所有训练 step 的均值。

| 每 token L2 范数，APE 之前 | 旧 Residual | 新方案 |
| --- | --- | --- |
| DINO-LoRA-MLS 输出 | 42.79 | 45.03 |
| 输入 adapter / PgT 输出，尚未乘 t | 68.22 | 78.46 |
| native JiT 输出，尚未加权 | 3.93 | 5.87 |
| native 路实际参与融合 | 3.93 | 3.57 |
| DINO 路实际参与融合 | 19.82 | 22.16 |
| 融合 token | 20.80 | 22.81 |

新方案的加权 DINO/native 范数比为 **6.21**，是逐诊断点求比后再平均；
不能用 `mean(t)*mean(norm)` 代替实际加权范数。PgT 输出约为 native 未加权输出的 13 倍。

**系数和为 1 不等于两路幅度相当。**在固定样本、固定 t 下，实际幅度比是
`rho(t)=t*||p|| / ((1-t)*||n||)`。它还取决于 token 自身的范数，而不只取决于 t。
PgT 内部 RMSNorm 不约束最终 Linear 的输出幅度，当前训练确实没有自动把两路拉到同一尺度。

但“幅度较大”只是可验证的假设，不是故障证明：JiT 自身有归一化，范数也不等于信息贡献，
两路方向和相互抵消同样重要。目前没有据此认定梯度失控或生成失败的依据。
**我们的设计目标尚未由范数数据验证，应通过单变量实验判断它是否值得显式约束。**

### 3.3 不要把 alignment projector 与输入 PgT 混淆

第 19 epoch 采样均值中，alignment ConvProj 后 objective cosine 是旧方案 0.658、新方案 0.669；
ConvProj 前的直接 cosine 都约为 0.0025。但两种表征可处于不同坐标系，投影前 cosine 很低
不等于没有语义；投影后 cosine 高也不证明主干表征改善了生成。

新方案 alignment ConvProj 输出范数约 272，**不是**输入 PgT 的 78.46。
cosine loss 不约束幅度，因此大的 alignment 范数或 raw MSE 不应被单独判为失败。
如需判断“投影头是否承担了过多对齐”，应看表征探针和分目标梯度，而非直接改成 MSE 或重新调对齐权重。

## 4. 推荐训练实验及具体改动边界

### E1：只校准 PgT 输出幅度，优先验证融合假设

对照是当前 raw + PgT + BlendT；只在 `_compose_jit_input_tokens` 中、PgT 输出之后加入校准。
**不要改官方 PgT 内部实现，也不要同时改 t 权重、预处理或 dropout。**

推荐先测试一个无可训练参数、逐样本的 RMS 匹配版本：

```text
r(u)  = sqrt(mean_{patch,channel}(u.float()^2) + eps)
p_cal = p * stop_gradient(r(n)) / r(p)
h     = (1-t)*n + t*p_cal
```

这是每张图一个标量缩放，不对各 patch 单独归一化，不减均值，保留 patch 间的相对幅度。
统计使用 FP32、合理 eps，随后转换回模型 dtype；native 参考幅度 detach，避免校准项额外反传到 native embed。
分母保留常规 RMS 归一化的梯度。它依赖当前样本，不维护 batch/running statistics，DDP 不增加同步。

可考虑一个新枚举 `dino_fusion_norm: none | native_rms`；这是**建议的新参数，目前未实现**。
首次只比较这两个值，记录校准前后范数、融合范数和两路 cosine。单位 RMS 或可学习 affine 不作为第一轮额外选项。

预期是避免 DINO 输出幅度掩盖 BlendT 的权重作用，而不是规定两路必须贡献相同信息。
风险是 token 幅度本身可能有用，且 native 参考幅度也会随训练变化；
即使满足 RMS 匹配，仍不保证角度合适、绝对范数恒定或 FID 更好。因此必须重新训练验证，不能只看范数接近。
实现时先测试关闭选项与旧前向等价、t=0/1 的行为、零/近零输入及 FP32/BF16 前后向有限性；
训练与采样共用同一个融合函数，teacher 和独立 alignment projector 不受这个选项影响。

### E2：只把 raw noisy 输入改成 DinoTNorm，优先迁移已有有效方向

同样以当前 raw + PgT + BlendT 为对照，保留未校准的 PgT 输出，只改变 `_encode_noisy_student` 的输入。
直接复用 [DinoTNorm 的现有 helper](../../Ours_BaseT_DinoTNorm_0914/dino_jit_denoiser.py) 的公式：

```text
mu_x    = 2*ImageNet_mean - 1
sigma_x = 2*ImageNet_std
mu_t    = t*mu_x
sigma_t = sqrt(t^2*sigma_x^2 + (1-t)^2*noise_scale^2)
DINO_input = (z_t - mu_t) / clamp(sigma_t, eps)
```

这里使用 ImageNet 先验矩，不是每张图的实际均值方差，也不是训练集精确重估的 noisy 统计。
其假设是 clean/noise 独立，并以这些通道统计近似当前增强后的 clean 数据分布。
t=0 时给 student 标准高斯输入；t=1 时与 clean teacher 的 ImageNet 预处理一致。
它可能减轻随 t 变化的输入尺度和 student/teacher clean 端输入域差异，但必须在新结构上重新验证。

建议输入模式最终用一个枚举 `raw | imagenet | noise_moment` 管理，替代布尔开关的语义，
而不是再叠加多个可能互相覆盖的开关；这是实现建议，不是当前已经存在的接口。
teacher 输入、teacher 最后一层、alignment=0.5、MLS、LoRA 和所有训练超参数保持不变。

**E1 与 E2 必须各自对照当前方案，第一轮不要只跑 E1+E2。**
若两个单独实验都有效，再跑组合，形成 current / E1 / E2 / E1+E2，检查收益是否可叠加。

### E3：标准预处理对照，解释 NoDinoPre 而非盲目跟随

只将当前 `dino_preprocess_input` 从 false 改为 true，其余保持当前方案，得到标准 ImageNet 预处理对照。
这是已有接口即可控制的单变量实验。官方 PixelDiT2 本来就保留该预处理；
它不能作为“官方也直接输入 raw noisy image”的论据。

优先级排在 E2 后：E2 有当前项目的正向结果支撑，而官方 frozen encoder 的结论不能直接移植到 LoRA+MLS student。
如需严格说明关闭预处理本身的作用，应补 E3；不将 raw / ImageNet / moment 的输入差异误写为 PgT 收益。

### E4：保留 native 主干，将 DINO 表示作为 AdaLN 条件

若幅度/输入校准后仍难超过 BaseT，应优先考虑**改变注入位置**，而不是不断增加 PgT 深度。
官方 PixelDiT2 的关键是保留 native patch 输入，用 PgT 表示调制主干；不是把它与 native 初始 token 相加。

```text
h0 = native(z_t) + pos_embed
g  = PgT(DINO(input(z_t,t)),t)
每个 JiT block：逐 token condition = global_time_class_condition + g
DDTHead / teacher alignment 保持原设置
```

这是结构示意，具体 AdaLN 激活、初始化和 context-token 处理应核对官方代码，不能机械套入当前 JiT。
优先复用官方 modulation/block 算子，通过兼容层连接；检查空间条件与 JiT class-context tokens 的形状和 RoPE。
不要同时换输出 head、冻结 LoRA、启用 grounding drop 或改损失，否则无法归因于注入位置。

### E5：必须补的归因基线

增加 `native JiT + 相同 DDTHead + 相同 clean teacher/iREPA alignment`，去掉 noisy DINO student 输入分支。
用它判断增益究竟来自 noisy DINO 表示，还是宽 DDTHead / teacher，而不只与原始 JiT 比较。
应统计实际配置的参数和推理成本；表中未运行的 `JiT+irepa+DDTHead` 的 186M 是占位，不能直接作为该对照的参数量。

我们的可训练参数约 307-320M，PixelDiT-B 为 186M、L 为 572M；
冻结 DINO 也有推理 FLOPs，LoRA student 的冻结主体不体现在可训练参数中。
因此目前不能宣称是在严格相同参数/计算预算下超过 PixelDiT-B，或仅靠结构就能追上 L 的 FID 2.60。

## 5. 先补诊断，暂不优先改变的部分

- **按 t 分桶，而非只看全局均值。**固定真实图像和噪声，检查 t=0.05/0.15/0.3/0.5/0.7/0.9/0.95 下的两路范数、融合范数、分支 cosine 和去噪误差；同时看真实标签/null-label。
- **检查 guidance 增量。**记录 `||v_cond-v_null|| / (||v_cond||+eps)` 随 t 的变化，以解释不同模型为什么需要不同 scale；这只是诊断，不预先改变 interval。
- **检查表征而非裸坐标 cosine。**查看 token 空间方差/有效秩、简单冻结线性探针或 patch 对应检索。必要时单独计算 denoise 与 alignment 对共享参数的梯度夹角；当前按模块的总梯度范数不能识别两目标冲突。
- **不急着启用 grounding drop。**DINO 输入是当前 noisy state，不是外部真实标签，null 分支共享这条路径并不自动构成条件泄漏。若再测试 dropout，应训练和采样一致；BlendT 丢 DINO 时恢复完整 native，而不是只剩 `(1-t)*native`。
- **不急着重新调 alignment 权重。**保留当前固定 0.5 和 iREPA 目标，不恢复已经清理的时间加权开关，也不因 raw MSE 大就改为 MSE 监督。
- **不急着改为 x-prediction。**当前已经预测 clean x 再换算 velocity；这不是新的 PixelDiT2 改进点。暂不同时改 LR、batch、LoRA rank、MLS 层数或 DDTHead。

## 6. 评测协议与继续训练的决策

所有训练对照从各自初始化开始，使用相同 seed，并核对公共模块初始权重一致。
不要加载旧 Residual checkpoint 后改变结构，再把续训结果当作等预算消融。
全局 batch 256、80 epoch 上限、warmup 5、blr=1e-4、constant LR、label-drop 0.1、EMA1=0.9999 等保持当前 YAML。

1. **20 epoch：**与当前 PgT-BlendT 同进度比较，online 保持 heun50/CFG=1，使用共享 `/data0/zhangruobei/Work/Pixel_Diffusion/fid_stats` 的 JiT 统计，只测 FID/IS。
2. **对应 CFG：**先固定 flow100、scale 2.5、interval `[0,1]`、50K class-balanced、EMA1；使用共享 `/data0/zhangruobei/Work/Pixel_Diffusion/nanogen-evals` 测 FID/IS/FDR6，保留六项特征距离与原始 JSON。
3. **候选筛选：**若固定 scale 不佳但 no-CFG 改善，补 scale 2.1/2.3，不能直接否定模型；不同模型采用相同初始噪声和类别序列。10K 只可用于初筛，不能替代最终 50K 报告。
4. **40 epoch：**把同时看生成指标和稳定性的候选推进至 40；若 20/40 都没有可见收益，暂停该方向的优先级。20-epoch 不佳本身不足以证明 80-epoch 一定失败。
5. **80 epoch：**最终比较同时给出固定 scale 和各自最佳 scale；小幅差距用新增 seed、配对采样或不确定性评估确认。更低范数/更高 cosine 但 FID/IS/FDR6 不改善，应判为诊断目标达成、方法收益未验证。

推荐执行顺序是：**完成现有队列及首次评测 -> DinoT4Y8 低 scale 补采样 -> E1/E2 两个单变量训练 -> 有效项组合 -> native-only 归因基线 / AdaLN 结构实验。**
这里的顺序是建议，不会自动追加 tmux 任务，也不抢占现有训练或采样使用的卡。

## 7. 依据与维护

- [主实验表](../../related_tex/Pixel-RAE_tex/exp0926.tex)：后续只更新 `.tex`，不需要编译。
- [当前方法与 iREPA / RAEv2 / PixelDiT2 的实现审计](implementation_comparison_irepa_raev2_pixeldit2.md)。
- [PixelDiT2 与 v1 的源码对比](../../PixelDiT2/PixelDiT2_vs_PixelDiT1_zh.md)。
- [当前融合和采样实现](../dino_jit_denoiser.py)、[online evaluation 实现](../jit/engine_jit.py)。
- [新方案训练日志](../checkpoints/BaseT-Residual(t,1-t)-NoDinoPre-PgT-B-80/train.log) / [结构化诊断](../checkpoints/BaseT-Residual(t,1-t)-NoDinoPre-PgT-B-80/no_dino_pre_pgt_blendt_diagnostics.jsonl)。
- [旧 Residual 训练日志](../checkpoints/Ours-J18-new-drop0.1-l4-BaseT-Residual-B-80/train.log) / [结构化诊断](../checkpoints/Ours-J18-new-drop0.1-l4-BaseT-Residual-B-80/residual_diagnostics.jsonl)。
- [已有两实验对比图和复现命令](../../plot/residual_pgt_comparison/README.md)：该画图快照是当天 14:53、12.779 epoch，不应与本文 19-epoch 统计混用。

新增实验应登记配置、代码版本、参数量、采样 seed、EMA、sampler、interval、统计值来源及结果 JSON。
本文是上述时间点的判断；首次 20-epoch FID 和 CFG 2.5 队列完成后，应据新结果重新排序，而不是把当前假设当作结论。
