# 80 epoch CFG 采样结果汇总（2026-10-08）

本轮排队的五项评测全部完成，最后一项于北京时间 10 月 8 日 06:00:20 结束。
队列已自动回填 [exp0926.tex](../../related_tex/Pixel-RAE_tex/exp0926.tex)。本次逐项核对了四个方法共 17 份完整报告、采样 manifest 与 checkpoint 身份，表中数值与报告一致；没有编译 LaTeX。

## 评测口径

- checkpoint-80，EMA1（0.9999），50K class-balanced，ImageNet 256。
- flow100，interval `[0,1]`，seed 0，每卡采样 batch 128，四卡。
- 离线 `nanogen-evals` 的 FID、IS、FDR6；FID reference 为 `imagenet_256_fid_stats`。
- 表一的 Ours 80-epoch CFG 列取表二各方法已完成实验中 FID 最低的一行，IS 和 FDR6 来自同一行；按原始精度选择，显示两位小数。
- 历史 20/40/60 epoch 和 online no-CFG 记录保持原样，不把它们当成本轮离线协议的新结果。

## 本轮新增结果

| 方法 | CFG scale | FID ↓ | IS ↑ | FDR6 ↓ |
| --- | ---: | ---: | ---: | ---: |
| BaseT | 2.5 | 5.17 | 288.96 | 16.77 |
| DinoTNorm | 2.5 | 5.22 | 292.09 | 14.44 |
| Residual | 2.5 | 5.28 | 299.00 | 15.17 |
| DinoT4Y8 | 2.1 | 5.23 | 249.60 | 15.11 |
| DinoT4Y8 | 2.3 | 5.48 | 277.28 | 14.19 |

前三项替换早期不同评测记录，不再沿用旧 CFG 2.5 的 FID/IS。DinoTNorm 复用了已有样本并重测完整指标；BaseT、Residual 重新采样评测。

## 各方法当前最低 FID

| 方法 | 已完成的 scale | 当前最佳 scale | FID ↓ | IS ↑ | FDR6 ↓ |
| --- | --- | ---: | ---: | ---: | ---: |
| BaseT | 2.1、2.3、2.5、2.7、2.9 | 2.3 | 5.17 | 263.61 | 17.67 |
| DinoT4Y8 | 2.1、2.3、2.5 | 2.1 | 5.23 | 249.60 | 15.11 |
| DinoTNorm | 2.1、2.3、2.5、2.7、2.9 | 2.3 | 4.89 | 268.63 | 15.11 |
| Residual | 2.1、2.3、2.5、2.7 | 2.3 | 5.14 | 273.70 | 16.09 |

BaseT 的 2.3/2.5 显示均为 5.17，但原始 FID 分别为 5.16934534 / 5.17055978，因此标灰 2.3。这个差异不足以证明 2.3 稳定优于 2.5。

## 结论与下一步

1. **DinoTNorm 是当前 FID 改善最明确的方向。**同 scale 2.3 相对 BaseT，FID 降低 5.49%，FDR6 降低 14.53%，IS 由 263.61 升至 268.63。2.1/2.3 的 FID 分别为 4.89570777 / 4.88572648，说明较好的区间在 2.1--2.3 附近，但二者差距很小。
2. **Residual 的 FID 优势较小，IS/FDR6 改善更明显。**同 scale 2.3 的 FID 仅降低 0.56%，FDR6 降低 8.98%，IS 提高到 273.70。固定 scale 2.5 时，Residual 的 FID 5.28 反而高于 BaseT 的 5.17，不能称其在所有 CFG 设置上都改善 FID。
3. **DinoT4Y8 需要较低 guidance。**scale 2.5 → 2.1 时，FID 5.96 → 5.23，降低 12.22%；IS 298.28 → 249.60，FDR6 13.62 → 15.11，体现指标间的权衡。当前最低 FID 仍略高于 BaseT，但最佳点在已测范围的下边界，尚未找到完整的最优区间。
4. **优先补 DinoT4Y8 的 1.9、2.0。**这比继续增大到 2.7 更有助于定位 FID 最低点。之后可对 DinoTNorm 补 2.2、BaseT 补 2.4、Residual 补 2.2/2.4，细化现有 0.2 间隔的搜索；这些是建议，本次没有新增采样任务。
5. **Residual+DinoTNorm 的收益需要组合实验验证。**现有单项结果支持尝试，但并不证明两者一定可叠加。最终对照仍需保持相同 epoch、采样协议，并分别报告固定 scale 和各自最佳已测 scale；微小差距需要额外 seed 验证。

DinoT4Y8 的 2.7 尚未采样，表中保留 `--`；Residual 的 2.9 无完整评测报告，沿用此前决定，不恢复该行。不能把“本轮队列完成”理解成所有 scale 都已覆盖或连续 scale 的全局最优已经确定。

## 结果来源

每个实验目录下使用 `checkpoint-80_cfg{scale}_interval0p0_1p0_flowdpm100_image50000/offline_metrics.json`，例如 2.3 对应 `cfg2p3`。对应 `samples.manifest.json` 记录 checkpoint、EMA、seed 和采样设置。

- [BaseT 评测目录](../../Ours_BaseT_Baseline_0914/checkpoints/eval_checkpoint/Ours-J18-new-drop0.1-l4-BaseT-B-80/)
- [DinoT4Y8 评测目录](../../Ours_BaseT_DinoT4Y8_1008/checkpoints/eval_checkpoint/Ours-J18-new-drop0.1-l4-BaseT-DinoT4Y8-B-80/)
- [DinoTNorm 评测目录](../../Ours_BaseT_DinoTNorm_0914/checkpoints/eval_checkpoint/Ours-J18-new-drop0.1-l4-BaseT-DinoTNorm-B-80/)
- [Residual 评测目录](../checkpoints/eval_checkpoint/Ours-J18-new-drop0.1-l4-BaseT-Residual-B-80/)
- [CFG 2.5 队列日志](../../Ours_BaseT_Baseline_0914/checkpoints/cfg2p5_metrics_4567.log)
- [DinoT4Y8 2.1/2.3 队列日志](../../Ours_BaseT_DinoT4Y8_1008/checkpoints/queue_cfg2p1_2p3_4567.log)
