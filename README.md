# PyTorch implementation of Generative Adversarial Networks (GAN) 
Implemented models:
* [GAN](https://arxiv.org/abs/1406.2661)
* [DCGAN](https://arxiv.org/abs/1511.06434)

Analysed datasets:
* [MNIST](http://yann.lecun.com/exdb/mnist/)
* [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)


# MNIST dataset experiments

## GAN

Simple GAN based on [Generative Adversarial Nets](https://arxiv.org/abs/1406.2661) paper
### Learning curves
![gan_metrics](./plots/gan/metrics.jpg)

### Examples of generated images in each epoch
https://github.com/thawro/gan-pytorch/assets/50373360/c420da6a-f517-4e50-b123-f59d317e164b


## DCGAN

GAN based on [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434) paper
### Learning curves
![dcgan_metrics](./plots/dcgan/metrics.jpg)

### Examples of generated images
https://github.com/thawro/gan-pytorch/assets/50373360/337e2b91-ea26-4ac6-b79d-e5d126780c59



# CelebA dataset experiments
