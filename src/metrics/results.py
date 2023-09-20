from dataclasses import dataclass
import numpy as np
import torch


@dataclass
class Result:
    data: np.ndarray
    y_true: np.ndarray
    y_pred: np.ndarray


@dataclass
class SegmentationResult:
    image: np.ndarray
    y_true: np.ndarray
    y_pred: np.ndarray


@dataclass
class GANResult:
    gen_imgs: torch.Tensor
