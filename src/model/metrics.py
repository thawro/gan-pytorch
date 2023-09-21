from torch import Tensor


class DiscriminatorProbs:
    def __init__(self, prefix: str):
        self.prefix = prefix

    def __call__(self, preds: Tensor) -> dict[str, float]:
        return {f"{self.prefix}_probs": preds.mean().item()}


class GANMetrics:
    def __init__(self):
        self.D_fake_probs = DiscriminatorProbs("D_fake")
        self.D_real_probs = DiscriminatorProbs("D_real")

    def calculate_metrics(
        self, D_fake_validity: Tensor, D_real_validity: Tensor
    ) -> dict[str, float]:
        return {**self.D_fake_probs(D_fake_validity), **self.D_real_probs(D_real_validity)}
