import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class HFImageDataset(Dataset):
    def __init__(self, data_dir, split="train", transform=None):
        from datasets import load_from_disk
        self.data_dir = data_dir
        split_str = "val" if split == "val" else ""
        self.img_dataset = load_from_disk(os.path.join(data_dir, f"imagenet-latents-images", split_str))
        self.transform = transform

    def __getitem__(self, idx):
        img_elem = self.img_dataset[idx]
        image, label = img_elem["image"], img_elem["label"]

        image = np.array(image.convert("RGB")).transpose(2, 0, 1)  # [H, W, C] -> [C, H, W]

        image = torch.from_numpy(image)
        label = torch.tensor(label)
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.img_dataset)


def dataset_location(data_dir):
    """Accept ImageNet root, original iREPA data root, or its Arrow layout."""
    root = Path(data_dir).expanduser().resolve()
    if (root / "imagenet-latents-images").is_dir():
        return "arrow", root
    for train in (root / "train", root / "imagenet/train"):
        if train.is_dir():
            return "imagefolder", train
    raise FileNotFoundError(
        f"No ImageNet dataset under {root}: expected train/, imagenet/train/, "
        "or imagenet-latents-images/."
    )


def build_train_dataset(data_dir, img_size):
    from torchvision import transforms
    from torchvision.datasets import ImageFolder
    from util.crop import center_crop_arr

    kind, path = dataset_location(data_dir)
    if kind == "arrow":
        # Upstream Arrow images are already cropped to the training resolution.
        return HFImageDataset(str(path), transform=transforms.RandomHorizontalFlip())
    return ImageFolder(str(path), transform=transforms.Compose([
        transforms.Lambda(lambda img: center_crop_arr(img, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.PILToTensor(),
    ]))
