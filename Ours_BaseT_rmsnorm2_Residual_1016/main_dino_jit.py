"""DINO + JiT 的训练入口，尽量原样复用 JiT 官方训练循环。

这个文件本身不实现训练 step、优化器、分布式训练或在线评估，而是作为一层
“适配器”完成三件事：

1. 在导入路径中加入项目根目录和 ``jit/``，从而复用原始 ``main_jit.py``；
2. 将 JiT 默认的 ``Denoiser`` 替换为包含 DINO student/teacher、LoRA、
   对齐损失和可选 DDTHead 的 ``DINOJiTDenoiser``；
3. 将 JiT 默认的 HF Arrow 数据集构造器替换为兼容 raw ImageNet
   ``train/val`` 目录和 iREPA HF Arrow 目录的实现。

因此，阅读这个文件时应把它理解为“配置和接线层”。真正的前向传播与损失在
``dino_jit_denoiser.py``，训练循环在 ``jit/main_jit.py``。

本文档化注释统一以 J18-new 的 B-scale 配置为例：ImageNet-256、DINOv3-B/16、
JiT-B/16、有效全局 batch 256。需要注意，代码中的 ``B`` 总是单个 DDP rank 的
本地 batch，而不是全局 batch；例如两卡 H100、每卡 batch 128 时，模型前向看到
的图像是 ``[128, 3, 256, 256]``，两卡梯度平均后才对应全局 batch 256。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torchvision.datasets import ImageFolder


# 当前文件位于项目根目录；JiT 代码位于其子目录 jit/ 中。
THIS_DIR = Path(__file__).resolve().parent
JIT_DIR = THIS_DIR / "jit"

# 使用 insert(0) 而不是 append，确保优先导入当前项目中的模块，避免环境中存在
# 同名的 main_jit、dataset 等模块时误用外部版本。
for path in (THIS_DIR, JIT_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

# 这些导入必须发生在上面的 sys.path 注入之后，因此忽略 E402（导入不在文件顶部）。
import main_jit  # noqa: E402
from dino_jit_denoiser import DINOJiTDenoiser  # noqa: E402


# main() 中会把 main_jit.HFImageDataset 替换成下面的兼容数据集。这里必须提前保存
# 原始类，否则兼容数据集处理 HF Arrow 时会再次调用自己，形成递归。保存 jit/dataset.py 中原始的 HFImageDataset
_ORIGINAL_HF_IMAGE_DATASET = main_jit.HFImageDataset


class FlexibleImageNet256Dataset(torch.utils.data.Dataset):
    """统一 raw ImageNet 与 iREPA/HF Arrow 两种磁盘布局。

    新实验的默认输入是 raw ImageNet；HF Arrow 仅作为历史复现和 I/O 对照的兼容路径。
    raw 分支对 ``image_size`` 执行确定性的 ADM/Dhariwal center crop，因此
    ``image_size=256`` 对齐 RAEv2/PixelDiT-256，``image_size=512`` 对齐 EDM2
    ``center-crop-dhariwal``。两种分辨率默认都不做水平翻转。

    支持的布局：

    - raw ImageNet：``data_dir/train/<class>/*.JPEG`` 和
      ``data_dir/val/<class>/*.JPEG``；
    - iREPA HF Arrow：由原始 ``main_jit.HFImageDataset`` 读取的
      ``data_dir/imagenet-latents-images[/val]``。

    raw 图像保留原分辨率，因此先执行 ADM 风格的 resize + center crop 到
    ``image_size``；HF Arrow 中的图像已经预处理到目标尺寸，不再重复 resize。
    两条路径都会应用 JiT 显式传入的 augmentation。当前新实验默认
    ``hflip_prob=0``；历史 JiT/iREPA 实验使用 ``hflip_prob=0.5``。

    J18-new 的单样本输出是 ``image: uint8 [3, 256, 256]`` 与标量类别标签；
    DataLoader 拼接后得到 ``[B_local, 3, 256, 256]`` 和 ``[B_local]``。像素在
    ``jit/engine_jit.py`` 中才会从 ``[0,255]`` 转成 float32 的 ``[-1,1]``。
    """

    def __init__(self, data_dir: str, split: str = "train", transform=None, image_size: int = 256):
        self.data_dir = Path(data_dir)
        # JiT 只区分 train 和 val；其他传入值统一按 train 处理。
        self.split = "val" if split == "val" else "train"
        self.transform = transform
        self.image_size = int(image_size)
        direct_root = self.data_dir / self.split

        # 优先检测标准 ImageFolder 目录。ImageFolder 负责扫描类别子目录并按排序后的
        # 类别名生成 0..999 标签，但这里暂不传 transform，以便先做确定性的中心裁剪。

        if direct_root.is_dir():
            self.impl = ImageFolder(str(direct_root), transform=None)
            self.direct = True
        else:
            # 未发现 data_dir/train 或 data_dir/val 时，按 iREPA 的 HF Arrow 布局读取。
            # 原始 HFImageDataset 自己负责 PIL -> CHW Tensor，并在内部调用 transform。在/jit/dataset.py中
            self.impl = _ORIGINAL_HF_IMAGE_DATASET(data_dir, split=split, transform=transform)
            self.direct = False

    def __len__(self) -> int:
        return len(self.impl)

    def __getitem__(self, idx):
        # HF 分支已经在原始实现中完成取图、转 Tensor、标签转换和数据增强。
        if not self.direct:
            return self.impl[idx]

        # raw ImageNet 分支：ImageFolder 返回 PIL.Image 和 Python int 标签。
        image, label = self.impl[idx]
        # center_crop_arr 先用 BOX 多次降采样，再以双三次插值缩放短边，最后中心裁剪；
        # 这与 ADM/JiT 对 raw ImageNet 的标准预处理一致。
        image = main_jit.center_crop_arr(image.convert("RGB"), self.image_size)
        if self.transform is not None:
            image = self.transform(image)
        # transform 可能为空，也可能只含 RandomHorizontalFlip；输出仍可能是 PIL。
        # 统一兜底转换为 [C, H, W] 的 uint8 Tensor；训练循环稍后会除以 255。
        if not torch.is_tensor(image):
            image = np.array(image.convert("RGB")).transpose(2, 0, 1)
            image = torch.from_numpy(image)
        # 显式构造 Tensor 标签，以保持与 HFImageDataset 返回类型一致。
        return image, torch.tensor(label)


def add_dino_jit_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """在 JiT 原始参数解析器上追加 DINO、LoRA 和 DDTHead 配置。

    这里仅声明参数；参数的校验、模块构建和实际计算主要发生在
    ``DINOJiTDenoiser.__init__`` 与 ``forward`` 中。
    """

    # ------------------------------------------------------------------
    # 1. DINO 编码器及多层特征聚合
    # ------------------------------------------------------------------
    # 当前 class-to-image 主线只保留 DINOv3；B/16 是 J18-new 的默认编码器。
    parser.add_argument("--dino_model_name", default="dinov3_vitb16", type=str,
                        help="DINOv3 encoder name, e.g. dinov3_vitb16 or dinov3_vitl16.")
    # 预训练权重文件。可传空字符串，让构建器从 PRETRAIN_MODELS_ROOT
    # （默认 Git 仓库根目录下的 pretrain_models/）自动解析；
    # 明确给路径更利于离线训练和复现实验。
    parser.add_argument(
        "--dino_weights",
        default="",
        type=str,
        help="Path to encoder weights. Pass empty string to auto-resolve from "
             "PRETRAIN_MODELS_ROOT (default: pretrain_models/ under the Git repository root).",
    )
    # 多层 token 的合并方式：
    # - none：只取单层（通常为最后一层）；
    # - mean：对选中层逐元素平均；
    # - mls：RAEv2 风格的选中层平均 + 最后一层空间均值广播。
    parser.add_argument("--dino_aggregation", default="none",
                        choices=["none", "mean", "mls"], type=str,
                        help="DINOv3 multi-layer aggregation. none=raw last layer and "
                             "requires dino_layers to be omitted; mean=stack(layers).mean(0); "
                             "mls=RAEv2 mean + last_mean broadcast.")
    # 指定参与聚合的 block 索引。字符串语法由编码器解析，既支持 last/last-4，
    # 也支持显式的逗号分隔索引。
    parser.add_argument("--dino_layers", default=None, type=str,
                        help="Zero-based DINOv3 layer indices used by mean/mls aggregation: "
                             "'last', 'last-4', '4,7,10,11', or 'default' "
                             "(RAEv2 defaults for the chosen encoder).")
    parser.add_argument(
        "--dino_generation_final_norm_affine",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep pretrained gamma/beta in the generation DINO's final LayerNorm. "
             "Disable for the no-affine ablation; frozen alignment teacher is unchanged.",
    )
    # Student uses the selected aggregation; frozen teacher always uses raw last.
    # ------------------------------------------------------------------
    # 2. DINO patch 对齐损失与特征预处理
    # ------------------------------------------------------------------
    parser.add_argument(
        "--dino_preprocess_input", action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply (x+1)/2 and ImageNet mean/std to the noisy DINO-LoRA student input. "
             "The frozen clean-image teacher is always preprocessed.",
    )
    parser.add_argument(
        "--dino_input_normalization",
        default="standard",
        choices=["standard", "t_moment"],
        type=str,
        help="Noisy student input normalization: fixed ImageNet statistics or "
             "analytic t-dependent diffusion moments.",
    )
    # Teacher target uses spatial z-score before the fixed negative-cosine loss.
    parser.add_argument(
        "--dino_zscore_alpha",
        default=0.8,
        type=float,
        help="Teacher spatial-mean subtraction strength.",
    )
    # DINO token 替代 JiT 原始 patch embedding 后，是否继续加 JiT 的固定绝对位置编码。
    # 在 gated_residual 模式中该开关必须为 True，APE 会在 native+residual 融合后
    # 统一添加一次。
    parser.add_argument("--jit_add_pos_embed_to_dino_tokens", action=argparse.BooleanOptionalAction, default=True)

    # DINO 与 JiT 的输入接口。native 是纯 JiT+iREPA 消融：DINO 只作为冻结
    # teacher 产生对齐目标，不构建 noisy-image student，也不向 JiT 注入 token。
    # time_blend 在原生 JiT 与投影后的 DINO token 之间按 (1-t) 和 t 插值；
    # 旧模式保持原样。
    parser.add_argument(
        "--dino_jit_input_mode",
        default="replace",
        choices=["native", "replace", "gated_residual", "time_blend"],
        type=str,
    )
    parser.add_argument(
        "--dino_input_projector",
        default="none",
        choices=["none", "pixeldit2"],
        type=str,
        help="Project noisy DINO-LoRA tokens with official PixelDiT2 TimestepProjection "
             "before time_blend; separate from the alignment ConvProj.",
    )
    parser.add_argument(
        "--dino_time_blend_reverse",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use DINO weight 1-t and JiT weight t instead of DINO t and JiT 1-t. "
             "Only valid with time_blend; grounding dropout still sets DINO=0, JiT=1.",
    )
    parser.add_argument(
        "--dino_grounding_drop_on_label_drop",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="In time_blend, use full native JiT tokens for null labels in both training and CFG sampling."
             " Disabled by default, so the DINO path remains active for null labels.",
    )
    parser.add_argument(
        "--dino_jit_pre_blend_rmsnorm",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply separate per-token RMSNorm modules with independently learned channel gains "
             "to JiT and projected DINO tokens before time blending.",
    )
    # residual gate 使用当前 clean-weight 时间约定：clean_linear 表示 g(t)=t，
    # constant_one 用于消融实验，表示始终以 1 注入 residual。replace 路径必须
    # 保留 none，避免无效超参数被静默忽略。
    parser.add_argument(
        "--dino_residual_gate",
        default="none",
        choices=["none", "clean_linear", "constant_one"],
        type=str,
    )
    # 独立 residual adapter 的中间宽度；B-scale 第一版为 768。
    parser.add_argument("--dino_residual_hidden_dim", default=0, type=int)
    parser.add_argument(
        "--dino_residual_drop_on_label_drop",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="When label dropout produces the null class, also zero the DINO "
        "representation residual. The same rule is applied to CFG's null-label "
        "branch at sampling time.",
    )

    # ------------------------------------------------------------------
    # 3. student token 投影与对齐来源
    # ------------------------------------------------------------------
    # Fixed ConvProj on the captured JiT block output.
    parser.add_argument(
        "--dino_projector_kernel_size",
        default=3,
        choices=[1, 3, 5, 7],
        type=int,
        help="Alignment ConvProj kernel size.",
    )
    parser.add_argument(
        "--jit_alignment_depth",
        default=4,
        type=int,
        help="One-based JiT block depth captured for teacher alignment.",
    )

    # ------------------------------------------------------------------
    # 4. DINO student 的 LoRA 容量
    # ------------------------------------------------------------------
    # r 决定低秩瓶颈维度；alpha/r 是 LoRA residual 的有效缩放；dropout 只作用于
    # LoRA 分支。target_modules 是逗号分隔的被注入线性层名称。
    parser.add_argument("--dino_lora_r", default=16, type=int)
    parser.add_argument("--dino_lora_alpha", default=16.0, type=float)
    parser.add_argument("--dino_lora_dropout", default=0.0, type=float)
    parser.add_argument("--dino_lora_target_modules", default="qkv,proj,fc1,fc2", type=str)

    # ------------------------------------------------------------------
    # 5. 除 LoRA 外允许直接训练的 DINO 参数
    # ------------------------------------------------------------------
    # BooleanOptionalAction 同时提供 --foo 与 --no-foo。DINOv3 的预训练 patch
    # embedding 可以选择 fine-tune；空间位置由无可训练参数的 RoPE 提供，因此没有
    # train_pos_embed 开关。CLS token 和末端 norm 仍可独立控制，teacher 始终冻结。
    parser.add_argument("--dino_train_patch_embed", action=argparse.BooleanOptionalAction, default=True)

    # ------------------------------------------------------------------
    # 6. DDT 风格像素解码头
    # ------------------------------------------------------------------
    # 默认用 DDTHead 替换 JiT final_layer；关闭后复用并训练原生像素输出层。
    parser.add_argument(
        "--use_ddt_head", action=argparse.BooleanOptionalAction, default=True,
        help="Use DDTHead; --no-use_ddt_head restores the native JiT final layer.",
    )
    # DDTHead 重新嵌入 noisy image，并用 JiT 输出通过逐 token AdaLN 调制。
    # 0 表示沿用 JiT hidden size；非零值允许使用更宽的头（例如 J8 的 2048）。
    parser.add_argument("--ddt_head_dim", default=0, type=int)
    # 0 表示自动选择：优先 JiT head 数，否则回退到可整除 head_dim 的设置。
    parser.add_argument("--ddt_head_num_heads", default=0, type=int)
    # DDT Transformer block 数量。
    parser.add_argument("--ddt_head_depth", default=2, type=int)
    # bottleneck embed 模式中 noisy image 的中间卷积通道数。
    parser.add_argument("--ddt_head_bottleneck_dim", default=128, type=int)
    return parser


def main() -> None:
    """完成模块替换、参数解析和数据集接线，然后进入 JiT 原始主循环。"""

    # main_jit.main 在构建模型时调用模块级符号 Denoiser(args)。在调用训练入口前
    # 替换该符号，即可复用完整 JiT 训练循环而无需复制或分叉其实现。
    main_jit.Denoiser = DINOJiTDenoiser

    # 先取得 JiT 的基础参数，再追加本文件定义的参数。分组 YAML 的叶子键会先成为
    # argparse 默认值，随后显式命令行参数可以覆盖路径、每卡 batch 等机器相关设置。
    parser = add_dino_jit_args(
        main_jit.get_args_parser()
    )
    args = main_jit.parse_args_with_config(parser)

    # main_jit.main 会按 HFImageDataset(args.data_path, ...) 构建数据集。替换该模块级
    # 类后，它无需知道底层是 raw ImageFolder 还是 HF Arrow。lambda 捕获最终配置中
    # 的 img_size，保证 YAML 覆盖后的分辨率也能正确传入。
    main_jit.HFImageDataset = (
        lambda data_dir, split="train", transform=None: FlexibleImageNet256Dataset(
            data_dir,
            split=split,
            transform=transform,
            image_size=args.img_size,
        )
    )
    # 提前创建输出目录，训练循环将在这里写日志、checkpoint 和在线评估结果。
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main_jit.main(args)


if __name__ == "__main__":
    main()
