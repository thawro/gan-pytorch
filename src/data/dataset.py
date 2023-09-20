"""Dataset classes"""

from torchvision.datasets import VisionDataset
import albumentations as A
from pathlib import Path
from geda.datasets.mnist import MNISTClassificationDataset
from typing import Any, Callable
from src.data.transforms import DataTransform


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
