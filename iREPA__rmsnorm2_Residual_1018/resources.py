"""Local asset paths; no changes to the original iREPA encoder implementation."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def resource_root(name):
    defaults = {'PRETRAIN_MODELS_ROOT': ROOT.parent / 'pretrain_models',
                'FID_STATS_ROOT': ROOT.parent / 'fid_stats',
                'IMAGENET_ROOT': Path('/mnt/shared-storage-user/datafactory-share/gushishen/imagenet')}
    return Path(os.environ.get(name) or defaults[name]).expanduser().resolve()


def prepare_assets(args):
    # These are the environment variables read by iREPA's unchanged loader.
    repo = Path(os.environ.get('DINOV3_REPO') or ROOT / 'SSL_code/dinov3').expanduser().resolve()
    weights_root = resource_root('PRETRAIN_MODELS_ROOT')
    os.environ['DINOV3_REPO_DIR'] = str(repo)
    os.environ['DINOV3_CKPT_DIR'] = str(weights_root)
    # The generation wrapper uses this name for the same source tree.
    os.environ['DINOV3_REPO'] = str(repo)
    weights = weights_root / 'dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth'
    paths = [repo / 'hubconf.py']
    if args.enc_type == 'dinov3-vit-b16':
        paths.append(weights)
    elif args.enc_type == 'dino-vit-b':
        from resource_paths import dino_assets
        dino_assets('b', args.dino_repo, args.dino_weights)
    else:
        raise ValueError('Prepared offline teachers: dinov3-vit-b16 or dino-vit-b')
    if args.generation_input_mode == 'rmsnorm2':
        args.generation_dino_weights = str(Path(args.generation_dino_weights).expanduser().resolve()) if args.generation_dino_weights else str(weights)
        paths.append(Path(args.generation_dino_weights))
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.online_eval or args.evaluate_gen:
        from resource_paths import evaluation_assets
        evaluation_assets(args.img_size)
    return {'dinov3_repo': str(repo), 'teacher': args.enc_type,
            'generation_weights': args.generation_dino_weights}
