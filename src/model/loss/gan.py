"""Implementation of loss Base and weighted loss classes"""

from torch import Tensor
from torch.nn.modules.loss import _Loss
from .base import WeightedLoss


class GANLoss(_Loss):
    def __init__(self, loss_fn: WeightedLoss):
        super().__init__()
        self.loss_fn = loss_fn

    def calculate_discriminator_loss(self, pred: Tensor, target: Tensor) -> Tensor:
        loss = self.loss_fn(pred, target)
        return loss
