# PyTorch implementation of Generative Adversarial Networks (GAN) 
Implemented models:
* [GAN](https://arxiv.org/abs/1406.2661)
* [DCGAN](https://arxiv.org/abs/1511.06434)
* [Conditional DCGAN](https://arxiv.org/abs/1411.1784)

Analyzed Datasets:
* [MNIST](http://yann.lecun.com/exdb/mnist/)
* [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

# MNIST

## GAN

#### Learning curves
![gan_metrics](./plots/MNIST/gan/metrics.jpg)

#### Generated examples
https://github.com/thawro/gan-pytorch/assets/50373360/c420da6a-f517-4e50-b123-f59d317e164b


## DCGAN

#### Learning curves
![dcgan_metrics](./plots/MNIST/dcgan/metrics.jpg)

### Generated examples
https://github.com/thawro/gan-pytorch/assets/50373360/337e2b91-ea26-4ac6-b79d-e5d126780c59


## Conditional DCGAN

#### Learning curves
![conditional_dcgan_metrics](./plots/MNIST/conditional_dcgan/metrics.jpg)

#### Examples (each row is conditioned with specific digit)
https://github.com/thawro/gan-pytorch/assets/50373360/79e9d0b0-c291-4d47-835b-19b28bcff0b2


# CelebA

## GAN

#### Learning curves
![gan_metrics](./plots/CelebA/gan/metrics.jpg)

#### Generated examples
https://github.com/thawro/gan-pytorch/assets/50373360/f91a2ecc-2434-4f1c-8312-17283d5ed4b0



## DCGAN

#### Learning curves
![gan_metrics](./plots/CelebA/dcgan/metrics.jpg)

#### Generated examples
https://github.com/thawro/gan-pytorch/assets/50373360/f9931527-45ec-4cf9-9c3f-a8e48eebe62a

#### Interpolation of latent variable
For each latent dimension `latent_dim` a new **`z`** noise was sampled from gaussian distribution and `z[latent_dim]` was then interpolated between `-3.5` and `3.5`

![CelebA_dcgan](https://github.com/thawro/gan-pytorch/assets/50373360/0b78f5b6-0c12-4397-8194-f7048dadad64)

