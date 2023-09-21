"""Implementation of specialized Module"""
from .base import BaseModule, SPLITS
from torch import Tensor, optim
from src.model.gan import GANModel
from src.model.loss.gan import GANLoss
import torch
from src.model.metrics import GANMetrics
from src.metrics.results import GANResult


class GANModule(BaseModule):
    model: GANModel
    loss_fn: GANLoss
    metrics: GANMetrics
    results: dict[str, GANResult]

    def __init__(self, model: GANModel, loss_fn: GANLoss):
        optimizers = {
            "generator": optim.Adam(model.generator.parameters(), lr=2e-4, betas=(0.5, 0.999)),
            "discriminator": optim.Adam(
                model.discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999)
            ),
        }
        metrics = GANMetrics()
        super().__init__(model, loss_fn, metrics, optimizers, schedulers={})
        self.results = {split: None for split in SPLITS}

    def _common_step(self, batch: Tensor, batch_idx: int, stage: str):
        if stage == "train":
            self.optimizers["discriminator"].zero_grad()
        real_imgs, *_ = batch
        device = real_imgs.device
        N, C, H, W = real_imgs.shape
        valid_label = torch.full((N,), 1.0, dtype=torch.float, device=device)
        fake_label = torch.full((N,), 0.0, dtype=torch.float, device=device)

        #  DISCRIMINATOR
        D_real_validity = self.model.discriminate(real_imgs).view(-1)
        D_real_loss = self.loss_fn.calculate_discriminator_loss(D_real_validity, valid_label)

        if stage == "train":
            D_real_loss.backward()

        latent_dim = self.model.generator.input_size[1]
        z = torch.randn(N, latent_dim, device=device)

        gen_imgs = self.model.generate(z)

        D_fake_validity = self.model.discriminate(gen_imgs.detach()).view(-1)
        D_fake_loss = self.loss_fn.calculate_discriminator_loss(D_fake_validity, fake_label)

        if stage == "train":
            D_fake_loss.backward()

        D_loss = (D_real_loss + D_fake_loss) / 2

        if stage == "train":
            self.optimizers["discriminator"].step()

        #  GENERATOR
        if stage == "train":
            self.optimizers["generator"].zero_grad()
        G_fake_validity = self.model.discriminate(gen_imgs).view(-1)

        G_loss = self.loss_fn.calculate_discriminator_loss(G_fake_validity, valid_label)

        if stage == "train":
            G_loss.backward()
            self.optimizers["generator"].step()

        losses = {"G_loss": G_loss.item(), "D_loss": D_loss.item()}
        self.steps_metrics_storage.append(losses, stage)

        metrics = self.metrics.calculate_metrics(
            D_fake_validity=D_fake_validity, D_real_validity=D_real_validity
        )
        self.steps_metrics_storage.append(metrics, stage)

        if batch_idx == self.total_batches[stage] - 1:
            self.results[stage] = GANResult(gen_imgs=gen_imgs.detach().cpu())
