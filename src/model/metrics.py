from torch import Tensor
import torch


class DiscriminatorAccuracy:
    def __init__(self, prefix: str):
        self.prefix = prefix

    def __call__(self, preds: Tensor, target: Tensor) -> dict[str, float]:
        _preds = preds.round().to(torch.int8)
        _target = target.to(torch.int8)
        accuracy = torch.eq(_preds, _target).float().mean().item()
        return {f"{self.prefix}_accuracy": accuracy}


class GANMetrics:
    def __init__(self):
        self.G_fake_accuracy = DiscriminatorAccuracy("G_fake")
        self.D_fake_accuracy = DiscriminatorAccuracy("D_fake")
        self.D_real_accuracy = DiscriminatorAccuracy("D_real")

    def calculate_metrics(
        self,
        G_fake_validity: Tensor,
        G_fake_target: Tensor,
        D_fake_validity: Tensor,
        D_fake_target: Tensor,
        D_real_validity: Tensor,
        D_real_target: Tensor,
    ) -> dict[str, float]:
        return {
            **self.G_fake_accuracy(G_fake_validity, G_fake_target),
            **self.D_fake_accuracy(D_fake_validity, D_fake_target),
            **self.D_real_accuracy(D_real_validity, D_real_target),
        }
