"""训练入口：解析配置，接入 DINOJiTDenoiser 和 ImageNet 数据集，调用 iREPA 训练框架。
模型计算在 dino_jit_denoiser.py；训练循环与优化器在 jit/main_jit.py、jit/engine_jit.py。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torchvision.datasets import ImageFolder


# 优先导入本项目及其 jit/ 下的模块。
THIS_DIR = Path(__file__).resolve().parent
JIT_DIR = THIS_DIR / "jit"

for path in (THIS_DIR, JIT_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import main_jit  # noqa: E402
from dino_jit_denoiser import DINOJiTDenoiser  # noqa: E402


# 保存 iREPA 原始读取器，替换入口后 Arrow 分支仍调用它。
_ORIGINAL_HF_IMAGE_DATASET = main_jit.HFImageDataset


class FlexibleImageNet256Dataset(torch.utils.data.Dataset):
    """兼容 raw ImageNet 和 iREPA HF Arrow，输出 uint8 [3,H,W] 图像与类别。
    raw 图像先做 ADM 中心裁剪；Arrow 图像沿用已有预处理。增强由训练配置传入。
    """

    def __init__(self, data_dir: str, split: str = "train", transform=None, image_size: int = 256):
        self.data_dir = Path(data_dir)
        # 沿用 iREPA 的 train/val 约定。
        self.split = "val" if split == "val" else "train"
        self.transform = transform
        self.image_size = int(image_size)
        direct_root = self.data_dir / self.split

        # raw 布局为 train/<class>/*.JPEG；否则使用原始 HF Arrow 读取器。
        if direct_root.is_dir():
            self.impl = ImageFolder(str(direct_root), transform=None)
            self.direct = True
        else:
            self.impl = _ORIGINAL_HF_IMAGE_DATASET(data_dir, split=split, transform=transform)
            self.direct = False

    def __len__(self) -> int:
        return len(self.impl)

    def __getitem__(self, idx):
        # Arrow 分支已在原读取器中完成增强和 Tensor 转换。
        if not self.direct:
            return self.impl[idx]

        image, label = self.impl[idx]
        # raw：PIL 图像 → ADM 中心裁剪 → 配置中的增强。
        image = main_jit.center_crop_arr(image.convert("RGB"), self.image_size)
        if self.transform is not None:
            image = self.transform(image)
        # PIL 转为 uint8 CHW；已由 transform 转成 Tensor 的图像直接保留。
        if not torch.is_tensor(image):
            image = np.array(image.convert("RGB")).transpose(2, 0, 1)
            image = torch.from_numpy(image)
        return image, torch.tensor(label)


def add_dino_jit_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """在 iREPA 参数上追加模型配置；每个布尔开关同时支持 --no-xxx。"""

    # DINO backbone 与生成侧选层；聚合和 LN 开关不影响 teacher。
    parser.add_argument('--dino_model_name', default='dinov3_vitb16', type=str,
                        help='DINOv3 backbone 名称，teacher/student 各自加载。')
    parser.add_argument('--dino_weights', default='', type=str,
                        help='预训练权重路径；留空时从 PRETRAIN_MODELS_ROOT 查找。')
    parser.add_argument('--dino_aggregation', default='none', choices=['none', 'mean', 'mls'], type=str,
                        help='生成侧取层：none 取最后层，mean 取层均值，mls 再加空间均值。')
    parser.add_argument('--dino_layers', default=None, type=str,
                        help='从0开始的层索引；支持 last、last-4、2,5,8,11 或 default。')
    parser.add_argument('--dino_generation_final_norm_affine', action=argparse.BooleanOptionalAction, default=True,
                        help='保留生成侧 final LN 的 gamma/beta；关闭即 NoAffine。')

    # student（DINO+LoRA）输入预处理三选一：none=不处理（NoDinoPre）；
    # t_moment=t 相关解析 norm；imagenet=与 teacher 相同的固定 ImageNet 标准化（PixelDiT2 grounding 方式）。
    # teacher 一侧的 ImageNet 固定标准化写死，不属于这里的选项。
    parser.add_argument('--dino_preprocess', choices=('none', 't_moment', 'imagenet'), default=None, type=str,
                        help='student 输入预处理三选一；未设置时跟随 dino_preprocess_input，都没有则默认 t_moment。')
    parser.add_argument('--dino_preprocess_input', action=argparse.BooleanOptionalAction, default=None,
                        help='[兼容旧配置] true=t_moment、false=none；新配置请改用 --dino_preprocess。')
    parser.add_argument('--dino_student_time_cond', action=argparse.BooleanOptionalAction, default=False,
                        help='patch_embed 后给每个 patch token 加 JiT 同款 t_embedder(t)，让 LoRA 感知时间。')
    parser.add_argument('--dino_student_label_cond', action=argparse.BooleanOptionalAction, default=False,
                        help='patch_embed 后给每个 patch token 加 JiT 同款 y_embedder(y)，跟随 label drop/null 类。')

    # iREPA 对齐总开关：--no-irepa_alignment 不构建 teacher、不算对齐 loss。
    parser.add_argument('--irepa_alignment', action=argparse.BooleanOptionalAction, default=True,
                        help='默认开启；关闭则跳过 frozen teacher 前向与 iREPA 对齐，只训练生成路径。')

    # iREPA 对齐目标的空间标准化。
    parser.add_argument('--dino_zscore_alpha', default=0.8, type=float, help='对齐目标的空间均值扣除系数，默认0.8。')

    # JiT 输入与两路 token 融合。
    parser.add_argument('--jit_add_pos_embed_to_dino_tokens', action=argparse.BooleanOptionalAction, default=True,
                        help='在融合后的空间 token 上添加 JiT 位置编码。')
    parser.add_argument('--dino_jit_input_mode',
                        default='time_blend', choices=['native', 'time_blend'], type=str,
                        help='native 仅用 JiT patch 输入；time_blend 按时间融合 JiT 与 DINO token。')
    parser.add_argument('--dino_input_projector', default='none', choices=['none', 'pixeldit2'], type=str,
                        help='生成侧 PgT，位于 DINO 与融合之间；与对齐 ConvProj 独立。')
    parser.add_argument('--dino_time_blend_reverse', action=argparse.BooleanOptionalAction, default=False,
                        help='time_blend 使用 (1-t)*DINO+t*JiT。')
    parser.add_argument('--dino_time_blend_transform', choices=('linear', 'smooth', 'smoother', 'sigmoid'), default='linear',
                        help='融合权重：linear=t；smooth=3t²-2t³；smoother=6t⁵-15t⁴+10t³；sigmoid 为归一化版本。')
    parser.add_argument('--dino_time_blend_key', type=float, default=100.0,
                        help='归一化 sigmoid 的固定斜率，默认100；仅 sigmoid 使用。')
    parser.add_argument('--dino_time_blend_a', type=float, default=0.1,
                        help='归一化 sigmoid 的中心点，默认0.1；仅 sigmoid 使用，需位于(0,1)。')
    parser.add_argument('--dino_grounding_drop_on_label_drop',
                        action=argparse.BooleanOptionalAction, default=False,
                        help='time_blend 的 null 类别只保留 JiT token；训练与 CFG 采样共用。')
    parser.add_argument('--dino_jit_pre_blend_rmsnorm', action=argparse.BooleanOptionalAction, default=False,
                        help='融合前两路分别做 RMSNorm，各自学习 gamma。')

    # JiT 中间层 → ConvProj → teacher 对齐。
    parser.add_argument('--dino_projector_kernel_size', default=3, choices=[1, 3, 5, 7], type=int,
                        help='对齐 ConvProj 的卷积核大小。')
    parser.add_argument('--jit_alignment_depth', default=4, type=int,
                        help='用于对齐的 JiT 层数；从1计数，默认4对应 blocks[3]。')

    # 生成侧 LoRA；原 Linear 冻结，只训练低秩 A/B。
    parser.add_argument('--dino_lora_r', default=16, type=int,
                        help='LoRA 秩。')
    parser.add_argument('--dino_lora_alpha', default=16.0, type=float,
                        help='LoRA 增量缩放为 alpha/r。')
    parser.add_argument('--dino_lora_dropout', default=0.0, type=float,
                        help='仅作用于 LoRA 分支的 dropout。')
    parser.add_argument('--dino_lora_target_modules', default='qkv,proj,fc1,fc2', type=str,
                        help='注入 LoRA 的线性层，逗号分隔。')

    # 额外可训练参数；CLS token 始终冻结。
    parser.add_argument('--dino_train_patch_embed', action=argparse.BooleanOptionalAction, default=True,
                        help='训练生成侧 DINO patch embedding。')
    parser.add_argument('--dino_train_final_norm', action=argparse.BooleanOptionalAction, default=False,
                        help='训练生成侧 final LN 的 gamma/beta；teacher 和 block norm 保持冻结。')

    # 可选像素输出头。
    parser.add_argument('--use_ddt_head', action=argparse.BooleanOptionalAction, default=True,
                        help='启用 DDTHead；关闭后使用 JiT 原生输出层。')
    parser.add_argument('--ddt_head_dim', default=0, type=int,
                        help='DDT 宽度；0沿用 JiT 宽度。')
    parser.add_argument('--ddt_head_num_heads', default=0, type=int,
                        help='DDT attention 头数；0自动选择。')
    parser.add_argument('--ddt_head_depth', default=2, type=int,
                        help='DDT block 数量。')
    parser.add_argument('--ddt_head_bottleneck_dim', default=128, type=int,
                        help='DDT 图像入口的瓶颈通道数。')
    return parser


def main() -> None:
    """解析配置并替换模型、数据集入口，复用 iREPA 训练循环。"""

    # 只替换模型工厂，训练循环、优化器和 checkpoint 仍由 iREPA 框架负责。
    main_jit.Denoiser = DINOJiTDenoiser

    # 优先级：命令行覆盖 YAML，YAML 覆盖默认值。
    parser = add_dino_jit_args(main_jit.get_args_parser())
    args = main_jit.parse_args_with_config(parser)

    # 兼容两种数据布局，并传入本次配置的图像分辨率。
    main_jit.HFImageDataset = (
        lambda data_dir, split="train", transform=None: FlexibleImageNet256Dataset(
            data_dir,
            split=split,
            transform=transform,
            image_size=args.img_size,
        )
    )
    # 训练产物统一写入 output_dir。
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main_jit.main(args)


if __name__ == "__main__":
    main()
