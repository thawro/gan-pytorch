"""Segmentation related plotting functions."""

import matplotlib.pyplot as plt
from src.metrics.results import GANResult
import torchvision.utils as vutils
import numpy as np
from moviepy.editor import ImageSequenceClip
from PIL import Image
import glob


def plot_generator_results(results: GANResult, filepath: str | None, title: str) -> None:
    """Plot image, y_true and y_pred (masks) for each result."""
    plt.figure(figsize=(15, 15))
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.title(title)
    plt.imshow(
        np.transpose(
            vutils.make_grid(results.gen_imgs[:64], padding=2, normalize=True).cpu(), (1, 2, 0)
        )
    )

    if filepath is not None:
        plt.savefig(filepath, bbox_inches="tight")
    plt.close()


def save_images_as_mp4(dirpath: str, fps: int, filepath: str, sort_by_numbers: bool = True):
    filepaths = sorted(glob.glob(f"{dirpath}/*.jpg"))
    if sort_by_numbers:
        filepaths = sorted(filepaths, key=lambda path: int(path.split("/")[-1].split(".")[0]))
    images = [Image.open(path) for path in filepaths]
    frames = np.stack([np.array(img) for img in images])
    clip = ImageSequenceClip(list(frames), fps=fps)
    clip.write_videofile(filepath, fps=fps)
