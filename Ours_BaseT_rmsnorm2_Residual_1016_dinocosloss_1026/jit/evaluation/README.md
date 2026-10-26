# DINO cosine 对齐方法的采样与离线评测

从本方法目录执行下列命令，先按方法 README 设置 `IMAGENET_ROOT`、`PRETRAIN_MODELS_ROOT`、`FID_STATS_ROOT`、`NANOGEN_EVALS_ROOT`、`DINOV3_REPO` 和 `PYTHONPATH=.:jit`。

在线评测从 `$FID_STATS_ROOT` 读取 JiT ImageNet-256 统计和 Inception 权重，计算 FID/IS。离线评测从 `$NANOGEN_EVALS_ROOT` 读取网络权重、缓存和参考统计，计算 FID、IS、FDR6；这些资源不随源码上传。

先检查离线评测资源：

```bash
python -m jit.evaluation.evaluate_fdr6 --check-assets
```

## 从匹配的训练配置采样

训练若使用了命令行超参数覆盖，应先将其合并到配置中；若本机启动器保存了完整的 `config_resolved.yaml`，优先使用它。采样保留 DINO、MLS、PgT、融合权重和 JiT 的训练结构，**不计算新 cosine loss 或原 iREPA loss，也不运行 teacher**。

```bash
CHECKPOINT=/path/to/this_experiment/checkpoint-80.pth
SAMPLE_DIR=/path/to/samples/dinocosloss-80-heun50-cfg4p1
python -m jit.evaluation.prepare_sampling \
    --config configs/reverse_sigmoid_blend_k20_a0p3_dino_ycond_dinocosloss_B80.yaml \
    --checkpoint "$CHECKPOINT" --output-dir "$SAMPLE_DIR" \
    --sampler heun --steps 50 --cfg 4.1 \
    --interval-min 0.05 --interval-max 1 \
    --num-images 50000 --gen-bsz 64

python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 \
    main_dino_jit.py --config "$SAMPLE_DIR/sampling_config.yaml"

python -m jit.evaluation.evaluate_fdr6 \
    --npz "$SAMPLE_DIR/samples.npz" \
    --metrics fid,inception_score,fdr6 \
    --output-json "$SAMPLE_DIR/offline_metrics.json"
```

`prepare_sampling` 保留 `dino_cos_loss_weight` 及模型设置，只覆盖采样设置。生成图像、NPZ 和评测结果保留在运行机器上。NPZ 包含 `arr_0`，格式为 uint8、RGB、NHWC；也可用 `--images /path/to/numbered_pngs` 直接评测图片目录。

在线 JiT 与离线 evaluator 的 Inception Score 数值协议不同，记录结果时应注明。
