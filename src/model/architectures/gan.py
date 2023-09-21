"""https://arxiv.org/pdf/1406.2661.pdf"""

from torch import nn, Tensor, optim
import numpy as np
from .base import Generator, Discriminator
from torch.nn.utils import spectral_norm

_size = tuple[int, ...]


class GANGenerator(Generator):
    def __init__(self, latent_dim: int, output_size: _size):
        flat_img_size = int(np.prod(output_size))

        net = nn.Sequential(
            *self.make_block(latent_dim, 128),
            *self.make_block(128, 256),
            *self.make_block(256, 512),
            *self.make_block(512, 1024),
            nn.Linear(1024, flat_img_size),
            nn.Tanh(),
        )
        input_size = (1, latent_dim)  # 1 for batch
        super().__init__(net, input_size)
        self.output_size = output_size

    def make_block(self, in_feats: int, out_feats: int):
        return [
            nn.Linear(in_feats, out_feats),
            nn.BatchNorm1d(out_feats, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
        ]

    def forward(self, z: Tensor) -> Tensor:
        N, *_ = z.shape
        img = self.net(z)
        img = img.view(N, *self.output_size)
        return img


class GANDiscriminator(Discriminator):
    def __init__(self, input_size: _size):
        flat_img_size = int(np.prod(input_size))
        net = nn.Sequential(
            spectral_norm(nn.Linear(flat_img_size, 512)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Linear(512, 256)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Linear(256, 1)),
            nn.Sigmoid(),
        )
        super().__init__(net, (1, *input_size))

    def forward(self, images: Tensor) -> Tensor:
        N, *_ = images.shape
        images_flat = images.view(N, -1)
        validity = self.net(images_flat)
        return validity
