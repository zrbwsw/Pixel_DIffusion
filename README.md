# Pixel_Diffusion 项目目录

H 集群协作请先阅读 [当前H200集群项目结构](当前H200集群项目结构.md)，其中列出了实际仓库、数据集、评测资源路径及 Codex 分工。

## GitHub 同步范围

仓库 `zrbwsw/Pixel_DIffusion` 发布以下三个版本的代码、配置、脚本、测试及必要
第三方源码，不上传实验输出、数据集、预训练权重或评测资源。三个版本使用相同依赖，
根目录 `require.txt` 引用原版本的依赖；也可使用各版本自己的 `require.txt`。

| 版本 | 融合前的 RMSNorm |
| --- | --- |
| [`Ours_BaseT_Residual_1008/`](Ours_BaseT_Residual_1008/) | 原始实现；启用时两路共享可学习通道权重。 |
| [`Ours_BaseT_rmsnorm2_Residual_1016/`](Ours_BaseT_rmsnorm2_Residual_1016/) | DINO、JiT 各自学习独立的通道权重。 |
| [`Ours_BaseT_nogamma_Residual_1016/`](Ours_BaseT_nogamma_Residual_1016/) | affine-free per-token RMSNorm，无可学习 gamma。 |

两个新版本的远端安装、预检、训练和采样参见 [部署说明](REMOTE_SETUP.md)。
原版本的历史部署说明位于 [remote_setup.md](Ours_BaseT_Residual_1008/readme/remote_setup.md)。
远端使用 `scripts/train_portable.sh` 选择 GPU 和资源路径，不要直接使用本机专用脚本。

以下为原始服务器的完整工作区说明；除上面列出的三个版本外，其他方法和资源目录不会随 Git clone 下载。

`Pixel_Diffusion` 是当前项目根目录。相关工作的源码、论文资料、实验记录、评测资源和我们的方法实现分开放置；以下只列非隐藏目录。

| 目录 | 职责 |
| --- | --- |
| `related_paper/` | 保存项目相关论文的 PDF。 |
| `related_tex/` | 保存相关论文的 LaTeX 源码及我们自己的实验文档。当前实验记录是 [`related_tex/Pixel-RAE_tex/exp0926.tex`](related_tex/Pixel-RAE_tex/exp0926.tex)；后续更新该文件即可，不需要编译。 |
| `fid_stats/` | 在线评测 FID 使用的 JiT 参考统计值。在线评测只记录 FID 和 IS。 |
| `nanogen-evals/` | 离线评测 FID、IS、FDR6 所需的评测网络权重、缓存和真实数据统计值。 |
| `iREPA/` | iREPA 相关工作源码。 |
| `LatentForcing/` | LatentForcing 相关工作源码。 |
| `PixelDiT/` | PixelDiT 相关工作源码及复现实验。 |
| `RAEv2/` | RAEv2 相关工作源码。 |
| `plot/` | 根据指定路径的实验日志绘图的脚本及代码。 |
| `Ours_BaseT_Baseline_0914/` | 我们的 BaseT 基准方法代码。 |
| `Ours_BaseT_DinoTNorm_0914/` | 我们的 BaseT + DinoTNorm 方法代码。 |
| `Ours_BaseT_DinoT4Y8_1008/` | BaseT + DinoT4Y8 方法代码及历史训练、采样结果。 |
| `Ours_BaseT_Residual_1008/` | BaseT + Residual 方法，以及 PgT、时间融合和 NoDDTHead 消融实验。 |
| `Ours_BaseT_rmsnorm2_Residual_1016/` | 两路独立可学习 RMSNorm 的 NoDDTHead 版本。 |
| `Ours_BaseT_nogamma_Residual_1016/` | 两路融合前使用无 gamma RMSNorm 的 NoDDTHead 版本。 |

## Ours 目录命名

方法目录采用 `Ours_<方法>_<最新修改日期>`，例如 `Ours_BaseT_Baseline_0914`。日期采用月日格式，标识代码版本，不代表训练完成日期。2026-10-08 在训练暂停后，将 Residual 和 DinoT4Y8 顶层目录规范为上表名称，并同步修改训练、采样、绘图入口中的路径。

Residual 项目中，本轮仅将以下两个实验的配置、checkpoint 目录和训练脚本与 `exp0926.tex` 的 Method 简称统一：

| 实验名 | 训练脚本 | 当前状态 |
| --- | --- | --- |
| `BaseT-Residual(t,1-t)-NoDinoPre-PgT-B-80` | `scripts/train_BaseT-Residual(t,1-t)-NoDinoPre-PgT-B-80_0123.sh` | 10月8日已在 0123 从 31 epoch 断点恢复，目标 80 epoch。 |
| `BaseT-Residual(t,1-t)-NoDinoPre-PgT-NoDDTHead-B-80` | `scripts/train_BaseT-Residual(t,1-t)-NoDinoPre-PgT-NoDDTHead-B-80_4567.sh` | 10月8日已在 4567 从头训练，目标 80 epoch。 |

其他旧实验及归档目录保持原名。已有 `.pth`、训练日志和历史评测记录保留原内容；恢复训练使用新 YAML 的 `resume` 路径，不改写 checkpoint 内的历史参数元数据。执行含括号的脚本或传递实验路径时，需要用引号包住路径。

各方法的训练配置、脚本、checkpoint 和更详细的实现说明以各自目录为准。在线和离线评测使用不同的参考统计值与指标实现，记录结果时应注明评测协议；实验表格统一更新到上述 `exp0926.tex`，无需编译 LaTeX。

旧采样目录中的 `sampling_config.yaml`、`samples.manifest.json` 和评测结果可能保留改名前的绝对路径，作为历史记录不改写。需要重新采样时，按新目录路径启动，并使用新的评测输出目录，避免与旧配置冲突。
