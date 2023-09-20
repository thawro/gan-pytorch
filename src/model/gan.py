from .architectures.base import Generator, Discriminator
from torch import nn, Tensor


class GANModel(nn.Module):
    def __init__(self, generator: Generator, discriminator: Discriminator):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator

    def generate(self, z: Tensor) -> Tensor:
        return self.generator(z)

    def discriminate(self, images: Tensor) -> Tensor:
        return self.discriminator(images)

    def forward(self, z: Tensor) -> Tensor:
        gen_imgs = self.generate(z)
        validity = self.discriminate(gen_imgs)
        return validity
