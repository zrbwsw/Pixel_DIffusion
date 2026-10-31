"""PixelDiT v1 的训练图像与采样数据集。

训练返回两种图像：normalized_image [-1,1] 给扩散模型，raw_image [0,1]
给 frozen DINOv2 teacher。HDF5 与 ImageFolder 都使用 ADM 中心裁剪；
RandomNDataset 则提供在线验证/离线采样所需的噪声、类别和随机种子。
"""
# Modified from https://github.com/MCG-NJU/PixNerd and https://github.com/End2End-Diffusion/REPA-E/

import copy
import io
import json
import os
import random
import re
import time
import unicodedata
from typing import Any, List, Optional, Union

import h5py
import numpy as np
import PIL.Image
import torch
from lightning.pytorch import LightningDataModule
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader, Dataset, IterableDataset


class CustomINH5Dataset(Dataset):
    """PixelDiT v1 的 HDF5 训练集，读取 REPA-E 离线裁剪好的图像。

    默认 random_flip=False，与上游 pix256_xl 一致。__getitem__ 返回
    normalized_image [3,H,W]、整数类别和 teacher 使用的 raw_image。
    """

    def __init__(self, data_dir: str, random_flip: bool = False):
        PIL.Image.init()
        self.random_flip = random_flip  # 是否额外做50%水平翻转；默认不改变 HDF5 图像
        supported_ext = PIL.Image.EXTENSION.keys() | {'.npy'}

        self.data_dir = data_dir
        self.h5_path = os.path.join(self.data_dir, "images.h5")  # 图像和 dataset.json 的 HDF5 容器
        self.h5_json_path = os.path.join(self.data_dir, "images_h5.json")  # HDF5 内对象路径清单
        self.h5f = h5py.File(self.h5_path, 'r')

        with open(self.h5_json_path, 'r') as f:
            self.h5_json = json.load(f)
        self.filelist = {fname for fname in self.h5_json}
        self.filelist = sorted(fname for fname in self.filelist if self._file_ext(fname) in supported_ext)  # 只保留 PNG/NPY 图像，排序固定样本顺序

        labels = self._load_h5_file("dataset.json")["labels"]  # [[文件路径,类别],...]
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')] for fname in self.filelist]  # 与排序后的图像逐一对应
        labels = np.array(labels)
        self.labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])

    def _load_h5_file(self, path):
        """读取一个 HDF5 对象；PNG 解码为 CHW uint8，JSON/NPY 按原格式返回。"""
        if path.endswith('.png'):
            rtn = np.array(PIL.Image.open(io.BytesIO(np.array(self.h5f[path]))))  # [H,W,C] uint8
            rtn = rtn.reshape(*rtn.shape[:2], -1).transpose(2, 0, 1)  # [H,W,C] → [C,H,W]
        elif path.endswith('.json'):
            rtn = json.loads(np.array(self.h5f[path]).tobytes().decode('utf-8'))
        elif path.endswith('.npy'):
            rtn = np.array(self.h5f[path])
        else:
            raise ValueError(f'Unknown file type: {path}')
        return rtn

    def __len__(self):
        return len(self.filelist)

    def _file_ext(self, fname):
        return os.path.splitext(fname)[1].lower()

    def __del__(self):
        self.h5f.close()

    def __getitem__(self, index):
        """返回生成模型输入、类别和 teacher 原图，三者严格对应同一样本。"""
        image_fname = self.filelist[index]
        image = self._load_h5_file(image_fname)  # [3,H,W]，已完成中心裁剪

        if self.random_flip and torch.rand(1).item() < 0.5:
            # 可选的50%水平翻转：反转 W 维；默认配置不会进入这里。
            image = np.ascontiguousarray(image[..., ::-1])

        image_tensor = torch.from_numpy(image).float() / 255.0  # raw_image：[3,H,W]，[0,1]，供 DINOv2 teacher
        normalized_image = (image_tensor - 0.5) / 0.5  # 生成模型输入：同一图像映射到 [-1,1]

        target = int(self.labels[index])
        metadata = {
            "raw_image": image_tensor,  # teacher 使用，不能再减均值/方差到 [-1,1]
            "class": target,
        }
        return normalized_image, target, metadata


def center_crop_arr(pil_image, image_size):
    """ADM 中心裁剪：先缩放短边，再取中央 image_size 正方形。

    大图先用 BOX 连续减半以降低计算量，随后用 BICUBIC 把短边调到
    image_size，最后沿 H/W 中心裁切。REPA-E HDF5 离线预处理也用它。
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=PIL.Image.BOX
        )  # 尺寸至少为2倍目标时先减半，减少后续 BICUBIC 成本
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=PIL.Image.BICUBIC
    )  # 等比例缩放，使短边等于 image_size
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return PIL.Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])  # 中央正方形，保持 RGB 和像素值域


class ImageFolderDataset(Dataset):
    """在线读取原始 ImageNet JPEG，并执行与 HDF5 相同的中心裁剪。

    random_flip 控制是否额外做50%水平翻转；当前 DinoYCond 配方设为
    False，以匹配 PixelDiT v1 的 HDF5 预处理。返回值布局与 CustomINH5Dataset 一致。
    """

    def __init__(self, data_dir: str, image_size: int = 512, random_flip: bool = True):
        from torchvision.datasets import ImageFolder

        self.dataset = ImageFolder(os.path.join(data_dir, "train"))  # data_dir/train/<class>/<image>，自动给出类别 id
        self.image_size = image_size
        self.random_flip = random_flip

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        """JPEG 解码 → ADM 中心裁剪 → 可选翻转 → 两种数值域的图像。"""
        image, target = self.dataset[index]  # PIL RGB 原图和类别 id
        image = center_crop_arr(image.convert("RGB"), self.image_size)  # [image_size,image_size] 的 PIL 图像
        if self.random_flip and random.random() < 0.5:
            image = image.transpose(PIL.Image.FLIP_LEFT_RIGHT)  # 可选水平翻转，关闭时与 HDF5 像素一致
        image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0  # [3,H,W]，[0,1]，供 teacher
        normalized_image = (image_tensor - 0.5) / 0.5  # 同一图像归一化到 [-1,1]，供生成模型
        return normalized_image, target, {"raw_image": image_tensor, "class": target}


def _clean_filename(s: str) -> str:
    s = s.strip().strip('.')
    s = unicodedata.normalize('NFKD', s).encode('ASCII', 'ignore').decode('ASCII')
    illegal_chars = r'[/]'
    s = re.sub(illegal_chars, '_', s)
    s = re.sub(r'_{2,}', '_', s)
    s = s.lower()
    max_length = 200
    s = s[:max_length]
    if not s:
        return 'untitled'
    return s


def _save_fn(image, metadata, root_path):
    image_path = os.path.join(root_path, f"{metadata['filename']}.png")
    PIL.Image.fromarray(image).save(image_path)


class RandomNDataset(Dataset):
    """为采样生成噪声和类别，而不是读取真实训练图像。

    每个样本由 (condition, seed) 决定：噪声形状为 latent_shape [C,H,W]，
    metadata 记录文件名、类别和 PNG 保存函数。在线 validation 与离线
    predict 都使用它，保证不同 checkpoint 可以复用同一组输入。
    """

    def __init__(self, latent_shape=(4, 64, 64), conditions: Union[int, List, str] = None,
                 seeds=None, max_num_instances=50000, num_samples_per_instance=-1):
        if isinstance(conditions, int):
            conditions = list(range(conditions))  # 整数 N → 类别 0..N-1
        elif isinstance(conditions, str):
            if os.path.exists(conditions):
                conditions = open(conditions, "r").read().splitlines()  # 文本文件每行一个条件
            else:
                raise FileNotFoundError(conditions)
        elif isinstance(conditions, list):
            conditions = conditions
        self.conditions = conditions
        self.num_conditons = len(conditions)
        self.seeds = seeds

        if num_samples_per_instance > 0:
            max_num_instances = num_samples_per_instance * self.num_conditons  # 每个类别固定生成这么多样本
        else:
            max_num_instances = max_num_instances

        if seeds is not None:
            self.max_num_instances = len(seeds) * self.num_conditons
            self.num_seeds = len(seeds)  # 每个类别使用这一组固定种子
        else:
            self.num_seeds = (max_num_instances + self.num_conditons - 1) // self.num_conditons  # 根据总数计算每类样本数
            self.max_num_instances = self.num_seeds * self.num_conditons
        self.latent_shape = latent_shape

    def __getitem__(self, idx):
        """按索引选择类别/种子，并生成可复现的 [C,H,W] 初始噪声。"""
        condition = self.conditions[idx // self.num_seeds]  # 样本按“类别 × 种子”的顺序枚举

        seed = random.randint(0, 1 << 31)
        if self.seeds is not None:
            seed = self.seeds[idx % self.num_seeds]  # 固定种子列表时，同一索引总得到同一噪声

        filename = f"{_clean_filename(str(condition))}_{seed}"  # 输出 PNG 文件名包含类别和种子，便于追踪
        generator = torch.Generator().manual_seed(seed)
        latent = torch.randn(self.latent_shape, generator=generator, dtype=torch.float32)  # [C,H,W]，采样的初始 x_0

        metadata = dict(
            filename=filename,
            seed=seed,
            condition=condition,
            save_fn=_save_fn,  # SaveImagesHook 用该函数把生成图像写为 PNG
        )
        return latent, condition, metadata

    def __len__(self):
        return self.max_num_instances


class ClassLabelRandomNDataset(RandomNDataset):
    """ImageNet 类别版 RandomNDataset：默认枚举全部1000类。

    默认 max_num_instances=50000，因此每类约50张，形成标准50K 采样集。
    """

    def __init__(self, latent_shape=(4, 64, 64), num_classes=1000, conditions: Union[int, List, str] = None,
                 seeds=None, max_num_instances=50000, num_samples_per_instance=-1):
        if conditions is None:
            conditions = list(range(num_classes))
        super().__init__(latent_shape, conditions, seeds, max_num_instances, num_samples_per_instance)


def mirco_batch_collate_fn(batch):
    batch = copy.deepcopy(batch)
    new_batch = []
    for micro_batch in batch:
        new_batch.extend(micro_batch)
    x, y, metadata = list(zip(*new_batch))
    stacked_metadata = {}
    for key in metadata[0].keys():
        try:
            if isinstance(metadata[0][key], torch.Tensor):
                stacked_metadata[key] = torch.stack([m[key] for m in metadata], dim=0)
            else:
                stacked_metadata[key] = [m[key] for m in metadata]
        except Exception:
            pass
    x = torch.stack(x, dim=0)
    return x, y, stacked_metadata


def collate_fn(batch):
    batch = copy.deepcopy(batch)
    x, y, metadata = list(zip(*batch))
    stacked_metadata = {}
    for key in metadata[0].keys():
        try:
            if isinstance(metadata[0][key], torch.Tensor):
                stacked_metadata[key] = torch.stack([m[key] for m in metadata], dim=0)
            else:
                stacked_metadata[key] = [m[key] for m in metadata]
        except Exception:
            pass
    x = torch.stack(x, dim=0)
    return x, y, stacked_metadata


def eval_collate_fn(batch):
    """把噪声堆成 [B,C,H,W]；类别/metadata 保持列表供采样和保存使用。"""
    batch = copy.deepcopy(batch)
    x, y, metadata = list(zip(*batch))
    x = torch.stack(x, dim=0)  # 每个元素 [C,H,W] → batch [B,C,H,W]
    return x, y, metadata


class DataModule(LightningDataModule):
    """组织训练、在线验证和离线采样三种数据流。

    train_dataset 提供真实图像及 teacher 原图；eval_dataset 提供周期性
    在线采样的噪声/类别；pred_dataset 提供正式生成评测的噪声/类别。
    batch_size 是每个 rank 的大小，分布式采样由 DistributedSampler 完成。
    """

    def __init__(self,
                 train_dataset: Dataset = None,
                 eval_dataset: Dataset = None,
                 pred_dataset: Dataset = None,
                 train_batch_size: int = 64,
                 train_num_workers: int = 16,
                 train_prefetch_factor: int = 8,
                 eval_batch_size: int = 32,
                 eval_num_workers: int = 4,
                 pred_batch_size: int = 32,
                 pred_num_workers: int = 4,
                 seed: int = None):
        super().__init__()
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.pred_dataset = pred_dataset
        self.train_batch_size = train_batch_size
        self.train_num_workers = train_num_workers
        self.train_prefetch_factor = train_prefetch_factor
        self.eval_batch_size = eval_batch_size
        self.pred_batch_size = pred_batch_size
        self.pred_num_workers = pred_num_workers
        self.eval_num_workers = eval_num_workers
        self.seed = seed if seed is not None else int(time.time())
        self._train_dataloader: Optional[DataLoader] = None

    def on_before_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        return batch

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        """构建真实图像训练 loader，按 rank 分片并 drop_last 保证 batch 一致。"""
        micro_batch_size = getattr(self.train_dataset, "micro_batch_size", None)
        if micro_batch_size is not None:
            assert self.train_batch_size % micro_batch_size == 0
            dataloader_batch_size = self.train_batch_size // micro_batch_size
            train_collate_fn = mirco_batch_collate_fn
        else:
            dataloader_batch_size = self.train_batch_size
            train_collate_fn = collate_fn

        if not isinstance(self.train_dataset, IterableDataset):
            sampler = torch.utils.data.distributed.DistributedSampler(
                self.train_dataset,
                seed=int(self.seed) if self.seed is not None else 0
            )  # 每个 rank 看到不同的训练子集，seed 控制 shuffle 顺序
        else:
            sampler = None

        self._train_dataloader = DataLoader(
            self.train_dataset,
            dataloader_batch_size,
            timeout=6000,
            num_workers=self.train_num_workers,
            prefetch_factor=self.train_prefetch_factor,
            collate_fn=train_collate_fn,
            sampler=sampler,
            pin_memory=True,
            persistent_workers=True,
            drop_last=True,
        )
        return self._train_dataloader

    def val_dataloader(self) -> EVAL_DATALOADERS:
        """构建在线验证 loader，不打乱类别/种子顺序，覆盖各自 rank 的采样子集。"""
        global_rank = self.trainer.global_rank
        world_size = self.trainer.world_size
        from torch.utils.data import DistributedSampler
        sampler = DistributedSampler(self.eval_dataset, num_replicas=world_size, rank=global_rank, shuffle=False)
        return DataLoader(
            self.eval_dataset,
            self.eval_batch_size,
            num_workers=self.eval_num_workers,
            prefetch_factor=2,
            sampler=sampler,
            collate_fn=eval_collate_fn,
        )

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        """构建正式采样 loader，使用与验证相同的分布式、非打乱采样方式。"""
        global_rank = self.trainer.global_rank
        world_size = self.trainer.world_size
        from torch.utils.data import DistributedSampler
        sampler = DistributedSampler(self.pred_dataset, num_replicas=world_size, rank=global_rank, shuffle=False)
        return DataLoader(
            self.pred_dataset,
            batch_size=self.pred_batch_size,
            num_workers=self.pred_num_workers,
            prefetch_factor=4,
            sampler=sampler,
            collate_fn=eval_collate_fn,
        )

