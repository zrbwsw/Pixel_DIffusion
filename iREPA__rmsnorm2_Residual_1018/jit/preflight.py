"""Read-only checks used before allocating a training GPU."""

import os
from pathlib import Path

from dataset import dataset_location
from resource_paths import dino_assets, evaluation_assets


def check_runtime(args):
    world_size = int(os.environ.get('WORLD_SIZE', args.world_size))
    global_batch = world_size * args.batch_size
    if world_size < 1 or args.batch_size < 1:
        raise ValueError('world_size and batch_size must be positive')
    if args.expected_global_batch_size is not None and global_batch != args.expected_global_batch_size:
        raise ValueError(f'Global batch {global_batch} != expected {args.expected_global_batch_size}')
    kind, path = dataset_location(args.data_path)
    report = {'model': args.model, 'dataset_type': kind, 'dataset_path': str(path),
              'world_size': world_size, 'batch_per_gpu': args.batch_size,
              'global_batch': global_batch, 'lr': args.lr if args.lr is not None else args.blr * global_batch / 256,
              'epochs': args.epochs, 'encoder': args.enc_type, 'encoder_depth': args.encoder_depth,
              'output_dir': str(Path(args.output_dir).resolve())}
    if kind == 'imagefolder':
        classes = sorted(p for p in path.iterdir() if p.is_dir())
        if len(classes) != args.class_num:
            raise ValueError(f'Expected {args.class_num} classes; found {len(classes)} at {path}')
        report['classes'] = len(classes)
    if args.enable_repa:
        if args.enc_type not in ('dino-vit-b', 'dino-vit-s'):
            raise ValueError('This offline preflight supports DINOv1; configure other encoders separately')
        repo, weights = dino_assets(args.enc_type.split('-')[-1], args.dino_repo, args.dino_weights)
        report.update(dino_repo=str(repo), dino_weights=str(weights), dino_weight_bytes=weights.stat().st_size)
    if args.online_eval or args.evaluate_gen:
        stats, inception = evaluation_assets(args.img_size)
        report.update(fid_stats=str(stats), inception_weights=str(inception))
    report['validation'] = 'paths and configuration only; run verify_jit_irepa.py for a real-image model check'
    return report
