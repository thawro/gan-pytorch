from torch import nn
import torch
from torch.nn.utils import spectral_norm

_size = tuple[int, ...]


class ConditionalDCGANGenerator(nn.Module):
    def __init__(self, n_classes: int, latent_dim: int, mid_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            *self.make_block(latent_dim + n_classes, mid_channels * 8, 4, 1, 0),
            *self.make_block(mid_channels * 8, mid_channels * 4, 4, 2, 1),
            *self.make_block(mid_channels * 4, mid_channels * 2, 4, 2, 1),
            *self.make_block(mid_channels * 2, mid_channels, 4, 2, 1),
            nn.ConvTranspose2d(mid_channels, out_channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )
        self.latent_dim = latent_dim
        self.n_classes = n_classes
        self.input_size = (1, latent_dim + n_classes)  # 1 for batch
        self.label_emb = nn.Embedding(n_classes, n_classes)

    def make_block(
        self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int
    ):
        return [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        ]

    def forward(self, z, labels):
        labels = self.label_emb(labels)
        gen_input = torch.cat((labels, z), -1)
        gen_input = gen_input.unsqueeze(-1).unsqueeze(-1)  # N x L -> N x L x 1 x 1
        return self.net(gen_input)


class ConditionalDCGANDiscriminator(nn.Module):
    def __init__(self, n_classes: int, in_channels: int, mid_channels: int, input_size: _size):
        super().__init__()
        h, w = input_size[-2:]
        self.label_emb = nn.Embedding(n_classes, n_classes)
        self.label_mapping = nn.Linear(n_classes, h * w)
        self.net = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels + 1, mid_channels, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            *self.make_block(mid_channels, mid_channels * 2, 4, 2, 1),
            *self.make_block(mid_channels * 2, mid_channels * 4, 4, 2, 1),
            *self.make_block(mid_channels * 4, mid_channels * 8, 4, 2, 1),
            spectral_norm(nn.Conv2d(mid_channels * 8, 1, 4, 1, 0, bias=False)),
            nn.Sigmoid(),
        )
        self.input_size = (1, *input_size)  # add batch dim

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

    def forward(self, images, labels):
        h, w = self.input_size[-2:]
        labels_emb = self.label_emb(labels)
        labels = self.label_mapping(labels_emb)
        labels = labels.view(-1, 1, h, w)
        disc_input = torch.cat((images, labels), 1)
        validity = self.net(disc_input)
        return validity
