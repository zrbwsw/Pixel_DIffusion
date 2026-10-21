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

本次 GitHub 发布不含开发机的 `scripts/sample_portable.sh`。可以直接执行以下 Python 入口（从项目根目录运行；先配置前述本地资源路径）：

```bash
experiment='BaseT-Residual(1-t,t)-NoDinoPre-PgT-blend_rmsnorm2-NoDDTHead-B-80'
EVAL_DIR="checkpoints/eval_checkpoint/$experiment/checkpoint-80_cfg2p5_heun50_example"

PYTHONPATH=.:jit python -m jit.evaluation.prepare_sampling \
  --config "configs/$experiment.yaml" \
  --checkpoint "checkpoints/$experiment/checkpoint-80.pth" \
  --output-dir "$EVAL_DIR" \
  --sampler heun --steps 50 --cfg 2.5 --interval-min 0 --interval-max 1

torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 \
  main_dino_jit.py --config "$EVAL_DIR/sampling_config.yaml"

# 确认采样成功后再评测。
PYTHONPATH=jit python -m evaluation.evaluate_fdr6 \
  --npz "$EVAL_DIR/samples.npz" \
  --metrics fid,inception_score,fdr6 \
  --output-json "$EVAL_DIR/offline_metrics.json"
```

使用与 checkpoint 匹配的配置；若训练时通过命令行覆盖了模型开关，采样配置也必须包含这些覆盖。PNG 是可恢复的中间结果，NPZ 使用 `arr_0.npy` 的 ZIP64 无压缩格式，通过 `.incomplete` 临时文件原子生成，默认不覆盖已完成 NPZ。

注意：离线 Inception Score 使用 vendored evaluator 的实现，online JiT 的
`torch-fidelity` IS 与它不是同一个数值协议；两者的结果不要直接混作同一列。
