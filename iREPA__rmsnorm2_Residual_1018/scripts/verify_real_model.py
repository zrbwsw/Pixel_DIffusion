"""Real ImageNet/DINOv3 CPU or GPU smoke check; never starts a training run."""
import argparse
from contextlib import nullcontext
import gc
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

os.environ.setdefault('TORCH_COMPILE_DISABLE','1')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'jit'))

import torch
from main import parse_args, check_runtime
from dataset import build_train_dataset
from vision_encoder import load_encoders
from spnorm import SpatialNormalization
from generation_input import build_denoiser

load_state_dict_from_url = torch.hub.load_state_dict_from_url


def local_only(url, *args, **kwargs):
    # DINOv3's official loader also uses this function for file:// URIs.
    if not str(url).startswith('file://'):
        raise RuntimeError(f'Unexpected network weight URL: {url}')
    return load_state_dict_from_url(url, *args, **kwargs)


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    parser.add_argument('--output',type=Path,required=True)
    options=parser.parse_args()
    args=parse_args(['--config',options.config,'--world_size','8'])
    report=check_runtime(args)
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    device=torch.device(options.device)
    print('Reading ImageNet and loading the unchanged iREPA DINOv3 teacher...',flush=True)
    dataset=build_train_dataset(args.data_path,args.img_size)
    raw,label=dataset[0]
    report.update(dataset_images=len(dataset),raw_dtype=str(raw.dtype),device=str(device))
    del dataset
    gc.collect()
    raw=raw.unsqueeze(0).to(device)
    with patch('torch.hub.load_state_dict_from_url',side_effect=local_only):
        teacher=load_encoders(args.enc_type,device,args.img_size)[0]
        with torch.no_grad():
            with torch.autocast('cuda',dtype=torch.bfloat16) if device.type=='cuda' else nullcontext():
                features=teacher.forward_features(teacher.preprocess(raw))
                target=SpatialNormalization(args.spnorm_method)(feat=features['x_norm_patchtokens'],cls=features['x_norm_clstoken'],cls_weight=args.cls_token_weight,zscore_alpha=args.zscore_alpha,zscore_proj_skip_std=args.zscore_proj_skip_std)
    args.z_dims=[teacher.embed_dim]
    report['teacher_shape']=list(target.shape)
    del teacher,features
    gc.collect()
    # Original JiT hardcodes CUDA transfers in RoPE/attention; redirect those
    # transfers only inside this CPU harness. Production files remain intact.
    transfers=patch.object(torch.Tensor,'cuda',lambda tensor,*a,**k:tensor) if device.type=='cpu' else nullcontext()
    print('Building the real generation branch and original JiT-B...',flush=True)
    with transfers,patch('torch.hub.load_state_dict_from_url',side_effect=local_only):
        model=build_denoiser(args).to(device).train()
    report['trainable_parameters']=sum(p.numel() for p in model.parameters() if p.requires_grad)
    x=raw.float()/255*2-1
    labels=torch.tensor([label],device=device)
    with transfers:
        with torch.autocast('cuda',dtype=torch.bfloat16) if device.type=='cuda' else nullcontext():
            loss,losses=model(x,labels,[target])
        loss.backward()
    assert torch.isfinite(loss)
    gradients={}
    for name,p in model.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(),name
            if any(key in name for key in ['projectors.0.projection_layer.weight','generation_input.patch_norm.weight','generation_input.dino_norm.weight','generation_input.projector.input_proj.weight','x_embedder.proj1.weight']):
                gradients[name]=p.grad.norm().item()
    assert all(value>0 for value in gradients.values()) and len(gradients)==5
    report.update(losses=losses,gradient_norms=gradients,backward_checked=True,cpu_transfer_redirect=device.type=='cpu',limitations='Batch=1 eager check; production 8-GPU BF16/DDP/memory and full 50K evaluation remain unverified.')
    options.output.parent.mkdir(parents=True,exist_ok=True)
    options.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
