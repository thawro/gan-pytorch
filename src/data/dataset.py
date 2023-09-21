"""Dataset classes"""

from torchvision.datasets import VisionDataset
import albumentations as A
from pathlib import Path
from geda.datasets.mnist import MNISTClassificationDataset
from typing import Any
from src.data.transforms import DataTransform
from PIL import Image
import glob
from torch import Tensor


class BaseDataset(VisionDataset):
    root: Path

    def __init__(
        self,
        root: str,
        split: str = "test",
        transform: DataTransform | None = None,
        target_transform: A.Compose | None = None,
    ):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.split = split
        self.root = Path(root)


class MNISTDataset(MNISTClassificationDataset, BaseDataset):
    def __init__(
        self,
        root: str,
        split: str = "test",
        transform: DataTransform | None = None,
        target_transform: A.Compose | None = None,
        download: bool = True,
    ):
        MNISTClassificationDataset.__init__(self, root, split, download)
        BaseDataset.__init__(self, root, split, transform, target_transform)

    def __getitem__(self, idx: int) -> Any:
        image, label = self.get_raw_data(idx)

        if self.transform is not None:
            image = self.transform(image)
        return image, label


class CelebADataset(BaseDataset):
    """http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html"""

    def __init__(
        self,
        root: str,
        split: str = "test",
        transform: DataTransform | None = None,
        target_transform: A.Compose | None = None,
    ):
        super().__init__(root, split, transform, target_transform)
        self.images_paths = glob.glob(f"{str(self.root)}/{split}/*")

    def get_raw_data(self, idx: int) -> Image.Image:
        image_fpath = self.images_paths[idx]
        image = Image.open(image_fpath)
        if not image.mode == "L":
            image = image.convert("RGB")
        return image

    def __getitem__(self, idx: int) -> Any:
        image = self.get_raw_data(idx)

        if self.transform is not None:
            image = self.transform(image)
        return image, Tensor()

    def __len__(self):
        return len(self.images_paths)
