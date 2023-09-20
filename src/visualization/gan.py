"""Segmentation related plotting functions."""

import matplotlib.pyplot as plt
from src.metrics.results import GANResult
import torchvision.utils as vutils
import numpy as np


def plot_generator_results(results: GANResult, filepath: str | None) -> None:
    """Plot image, y_true and y_pred (masks) for each result."""
    plt.figure(figsize=(15, 15))
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.title("Generates images")
    plt.imshow(
        np.transpose(
            vutils.make_grid(results.gen_imgs[:64], padding=2, normalize=True).cpu(), (1, 2, 0)
        )
    )

    if filepath is not None:
        plt.savefig(filepath, bbox_inches="tight")
    plt.close()
