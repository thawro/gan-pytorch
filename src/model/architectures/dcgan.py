"""https://arxiv.org/pdf/1511.06434.pdf"""

from torch import nn, Tensor
from .base import Generator, Discriminator
from torch.nn.utils import spectral_norm

_size = tuple[int, ...]


class DCGANGenerator(Generator):
    def __init__(self, latent_dim: int, mid_channels: int, out_channels: int):
        net = nn.Sequential(
            *self.make_block(latent_dim, mid_channels * 8, 4, 1, 0),
            *self.make_block(mid_channels * 8, mid_channels * 4, 4, 2, 1),
            *self.make_block(mid_channels * 4, mid_channels * 2, 4, 2, 1),
            *self.make_block(mid_channels * 2, mid_channels, 4, 2, 1),
            nn.ConvTranspose2d(mid_channels, out_channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )
        input_size = (1, latent_dim)  # 1 for batch
        super().__init__(net, input_size)

    def make_block(
        self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int
    ):
        return [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        ]

    def forward(self, z: Tensor) -> Tensor:
        z = z.unsqueeze(-1).unsqueeze(-1)  # N x L -> N x L x 1 x 1
        return self.net(z)


class DCGANDiscriminator(Discriminator):
    def __init__(self, in_channels: int, mid_channels: int, input_size: _size):
        net = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, mid_channels, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            *self.make_block(mid_channels, mid_channels * 2, 4, 2, 1),
            *self.make_block(mid_channels * 2, mid_channels * 4, 4, 2, 1),
            *self.make_block(mid_channels * 4, mid_channels * 8, 4, 2, 1),
            spectral_norm(nn.Conv2d(mid_channels * 8, 1, 4, 1, 0, bias=False)),
            nn.Sigmoid(),
        )
        super().__init__(net, (1, *input_size))

    def make_block(
        self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int
    ):
        return [
            spectral_norm(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        ]

    def forward(self, images: Tensor) -> Tensor:
        validity = self.net(images)
        return validity.squeeze(-1).squeeze(-1)  # N x L x 1 x 1 -> N x L
