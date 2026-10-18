"""Run original iREPA, optionally attaching the rmsnorm2 generation input."""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'jit'))

import main_jit as irepa
from generation_input import build_denoiser
from resources import prepare_assets


def parse_args(argv=None):
    parser = argparse.ArgumentParser('iREPA + rmsnorm2 input', parents=[irepa.get_args_parser()])
    parser.set_defaults(report_to='tensorboard', enc_type='dinov3-vit-b16')
    parser.add_argument('--generation_input_mode', choices=['baseline', 'rmsnorm2'], default='rmsnorm2')
    parser.add_argument('--generation_dino_weights', default=None)
    parser.add_argument('--dino_aggregation', choices=['mls', 'none'], default='mls')
    parser.add_argument('--dino_layers', default='default')
    parser.add_argument('--dino_time_blend_reverse', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--dino_lora_r', type=int, default=64)
    parser.add_argument('--dino_lora_alpha', type=float, default=64.0)
    parser.add_argument('--dino_lora_dropout', type=float, default=0.0)
    parser.add_argument('--dino_train_patch_embed', action=argparse.BooleanOptionalAction, default=True)
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument('--config')
    config, _ = probe.parse_known_args(argv)
    if config.config:
        from omegaconf import OmegaConf
        values = OmegaConf.to_container(OmegaConf.load(config.config), resolve=True)
        known = {action.dest for action in parser._actions}
        if set(values) - known:
            parser.error(f'Unknown config keys: {sorted(set(values) - known)}')
        parser.set_defaults(**values)
    args = parser.parse_args(argv)
    if not args.enable_repa:
        parser.error('This baseline requires enable_repa=true')
    if args.generation_input_mode == 'rmsnorm2':
        if (args.dino_aggregation, args.dino_layers) not in [('mls', 'default'), ('none', None)]:
            parser.error('Use dino_aggregation=mls / dino_layers=default or none / null')
    for key in ('output_dir', 'resume'):
        value = getattr(args, key)
        if value:
            path = Path(value).expanduser()
            setattr(args, key, str(path.resolve() if path.is_absolute() else (ROOT / path).resolve()))
    args.implementation = 'iREPA__rmsnorm2_Residual_1018'
    return args


def verify_upstream():
    manifest = json.loads((ROOT / 'upstream_manifest.json').read_text())
    for name, expected in manifest['unchanged_irepa'].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Pinned iREPA source changed: {name}')
    return len(manifest['unchanged_irepa'])


def check_runtime(args):
    from dataset import dataset_location
    world = int(os.environ.get('WORLD_SIZE', args.world_size))
    batch = world * args.batch_size
    if world < 1 or args.batch_size < 1 or args.expected_global_batch_size != batch:
        raise ValueError(f'Global batch {batch} != expected {args.expected_global_batch_size}')
    kind, data = dataset_location(args.data_path)
    if args.resume:
        if args.resume != args.output_dir:
            raise ValueError('Resume only from this experiment output directory')
        if not (Path(args.resume) / 'checkpoint-last.pth').is_file():
            raise FileNotFoundError(Path(args.resume) / 'checkpoint-last.pth')
    assets = prepare_assets(args)
    return dict(implementation=args.implementation, pinned_irepa_files=verify_upstream(),
                model=args.model, generation_input_mode=args.generation_input_mode,
                dino_aggregation=args.dino_aggregation, dino_layers=args.dino_layers,
                dino_time_blend_reverse=args.dino_time_blend_reverse,
                P_mean=args.P_mean, P_std=args.P_std, world_size=world,
                batch_per_gpu=args.batch_size, global_batch=batch,
                effective_lr=args.lr if args.lr is not None else args.blr * batch / 256,
                epochs=args.epochs, data_kind=kind, data_path=str(data),
                output_dir=args.output_dir, resume=args.resume, **assets)


@contextmanager
def output_lock(args):
    if int(os.environ.get('RANK', '0')) != 0:
        yield
        return
    locks = ROOT / '.runtime_locks'
    locks.mkdir(exist_ok=True)
    name = hashlib.sha256(args.output_dir.encode()).hexdigest() + '.lock'
    with (locks / name).open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        output = Path(args.output_dir)
        if args.resume:
            import torch
            if Path(args.resume) != output:
                raise ValueError('Resume only from this experiment output directory')
            state = torch.load(output / 'checkpoint-last.pth', map_location='cpu', mmap=True, weights_only=False)
            saved = state['args']
            saved = saved if isinstance(saved, dict) else vars(saved)
            keys = ('implementation', 'generation_input_mode', 'dino_aggregation', 'dino_layers',
                    'dino_time_blend_reverse', 'dino_lora_r', 'dino_lora_alpha', 'dino_lora_dropout',
                    'dino_train_patch_embed', 'model', 'img_size', 'enc_type', 'encoder_depth',
                    'projection_layer_type', 'proj_kwargs_kernel_size', 'projection_loss_type',
                    'proj_coeff', 'spnorm_method', 'zscore_alpha', 'zscore_proj_skip_std',
                    'P_mean', 'P_std', 'noise_scale', 't_eps', 'label_drop_prob', 'attn_dropout',
                    'proj_dropout', 'ema_decay1', 'ema_decay2', 'batch_size', 'expected_global_batch_size',
                    'blr', 'lr_schedule', 'warmup_epochs', 'weight_decay', 'seed')
            for key in keys:
                if saved.get(key) != getattr(args, key):
                    raise ValueError(f'Resume configuration mismatch: {key}')
            if not 0 < state['epoch'] < args.epochs:
                raise ValueError('Checkpoint is complete or has an invalid epoch')
            del state
        elif output.exists() and any(output.iterdir()):
            raise FileExistsError(f'Output is nonempty; use --resume or a new output: {output}')
        output.mkdir(parents=True, exist_ok=True)
        (output / 'resolved_config.json').write_text(json.dumps(vars(args), indent=2) + '\n')
        yield


def main(argv=None):
    args = parse_args(argv)
    report = check_runtime(args)
    if args.preflight:
        print(json.dumps(report, indent=2))
        return
    # Original main still owns seeding, teacher creation, DDP, optimizer, EMA,
    # the training loop, checkpointing and evaluation. Only its factory changes.
    original_factory = irepa.Denoiser
    try:
        irepa.Denoiser = build_denoiser
        with output_lock(args):
            irepa.main(args)
    finally:
        irepa.Denoiser = original_factory


if __name__ == '__main__':
    main()
