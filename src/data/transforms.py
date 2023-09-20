"""Data transforms for train and inference."""
import torchvision.transforms as T
import numpy as np
import torch
from functools import partial

_mean_std = float | tuple[float, ...]

# [-1, 1]
# MEAN = (0.5, 0.5, 0.5)
# STD = (0.5, 0.5, 0.5)

# Imagenet
# MEAN = (0.485, 0.456, 0.406)
# STD = (0.229, 0.224, 0.225)

# [0, 1]
MEAN = (0, 0, 0)
STD = (1, 1, 1)


class DataTransform:
    def __init__(self, is_train: bool, image_size: int, mean: _mean_std, std: _mean_std):
        transform_fn = train_transform if is_train else inference_transform
        self.transform = transform_fn(image_size, mean, std)
        self.inverse_preprocessing = partial(inverse_preprocessing, mean=mean, std=std)

    def __call__(self, image, **kwargs):
        return self.transform(image, **kwargs)


def train_transform(image_size: int, mean: _mean_std, std: _mean_std) -> T.Compose:
    """Return train transforms."""
    return T.Compose(
        [
            T.ToTensor(),
            T.Resize(image_size, antialias=True),
            T.RandomCrop(image_size),
            T.Normalize(mean, std),
        ]
    )


def inference_transform(image_size: int, mean: _mean_std, std: _mean_std) -> T.Compose:
    """Return inference transforms."""
    return T.Compose(
        [
            # SquarePad(),
            T.ToTensor(),
            T.Resize(image_size, antialias=True),
            T.CenterCrop(image_size),
            T.Normalize(mean, std),
        ]
    )


def inverse_preprocessing(
    image: np.ndarray | torch.Tensor, mean: _mean_std, std: _mean_std
) -> np.ndarray:
    """Apply inverse of preprocessing to the image (for visualization purposes)."""
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    _image = image.transpose(1, 2, 0)
    _image = (_image * np.array(std)) + np.array(mean)
    return _image
