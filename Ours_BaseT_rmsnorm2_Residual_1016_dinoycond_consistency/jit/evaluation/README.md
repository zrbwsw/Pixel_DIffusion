# Ours Residual 离线评测

本目录提供与 JiT online evaluation 分开的离线评测入口。`main_dino_jit.py`
中的 online eval 从共享目录 `/mnt/shared-storage-user/datafactory-share/gushishen/agent-z/Pixel_DIffusion/fid_stats/jit_in256_stats.npz` 读取统计值，
只计算 FID 和 Inception Score。离线流程在采样结束后读取生成的 PNG 或 NPZ，
使用统一缓存 `/mnt/shared-storage-user/datafactory-share/gushishen/agent-z/Pixel_DIffusion/nanogen-evals`
计算 `fid,inception_score,fdr6`。

## 评测缓存

`nanogen-evals` 中应包含 `stats/.../snapshots/<revision>/` 下的参考统计文件，
以及 `torch/hub/checkpoints/` 的 Inception 权重和 `huggingface/hub/` 下五个
timm extractor 的权重。Online 的 Inception 权重则与 JiT 统计值一起从共享 `fid_stats` 读取，可用 `FID_STATS_ROOT` 覆盖；两条评测路径都不依赖用户 Torch 缓存。
离线程序在当前进程内设置 `TORCH_HOME`、`HF_HOME`、`HF_HUB_CACHE` 和
`HF_HUB_OFFLINE=1`，将网络权重缓存限定到该目录。可用 `NANOGEN_EVALS_ROOT`
覆盖根目录；离线 CLI 的 `--cache-root` 优先级更高。
先执行：

```bash
cd /mnt/shared-storage-user/datafactory-share/gushishen/agent-z/Pixel_DIffusion/Ours_BaseT_rmsnorm2_Residual_1016
PYTHONPATH=jit python -m evaluation.evaluate_fdr6 --check-assets
```

六个评测网络权重和七份参考 NPZ 已统一存放在该目录。Hugging Face snapshot
仅链接到同一缓存中的 `blobs/`，不依赖其他账号的缓存。迁移后若报告
`ready: false`，根据报告补齐本地文件。
`features/*.features.pt` 保存的是已提取的真实图像特征张量，并非网络权重，
不能用它们替代上述 Inception / timm 模型缓存。

## 直接评测

```bash
PYTHONPATH=jit python -m evaluation.evaluate_fdr6 \
  --npz /path/to/samples.npz \
  --metrics fid,inception_score,fdr6 \
  --output-json /path/to/offline_metrics.json
```

NPZ 必须包含 `arr_0`，格式为 `uint8`、RGB、NHWC。也可以直接读取一个只含
编号 PNG 的目录：

```bash
PYTHONPATH=jit python -m evaluation.evaluate_fdr6 \
  --images /path/to/numbered_pngs \
  --metrics fid,inception_score,fdr6
```

## 采样、导出和断点续跑

每个 Ours 采样进程仍然把图片写入原有的
`generated_images/<method>-steps.../00000.png` 目录，因此不会破坏已有的 PNG
路径和 `--resume_eval_images` 行为。需要 NPZ 时使用：

```bash
experiment='BaseT-Residual(1-t,t)-NoDinoPre-PgT-blend_rmsnorm2-NoDDTHead-B-80'
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/sample_portable.sh \
  "configs/$experiment.yaml" \
  "checkpoints/$experiment/checkpoint-80.pth" \
  "checkpoints/eval_checkpoint/$experiment/checkpoint-80_cfg2p5_heun50_example" \
  --sampler=heun --steps=50 --cfg=2.5 --interval-min=0 --interval-max=1
```

脚本先检查全部离线资源、生成并校验采样配置，再用 `torchrun` 采样；采样正常退出后由单进程运行离线 FID/IS/FDR6。追加 `--preflight` 只检查资源和配置，不启动 GPU 任务。新训练若有 `config_resolved.yaml`，优先用它作为第一个参数，以包含命令行覆盖。PNG 是可恢复的中间结果，NPZ 使用 `arr_0.npy` 的 ZIP64 无压缩
格式，并通过 `.incomplete` 临时文件原子生成。默认不覆盖已完成的 NPZ。

注意：离线 Inception Score 使用 vendored evaluator 的实现，online JiT 的
`torch-fidelity` IS 与它不是同一个数值协议；两者的结果不要直接混作同一列。
