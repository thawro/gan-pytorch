from __future__ import annotations

from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from src.model.module.trainer import Trainer

from .base import BaseCallback
from src.visualization.gan import plot_generator_results, save_images_as_mp4


class GeneratorExamplesPlotterCallback(BaseCallback):
    """Plot prediction examples"""

    def __init__(self, stage: str, dirpath: str):
        self.dirpath = dirpath
        self.stage = stage

    def on_epoch_end(self, trainer: Trainer) -> None:
        results = trainer.module.results[self.stage]
        if results is None:
            return
        filepath = f"{self.dirpath}/{trainer.current_epoch}.jpg"
        plot_generator_results(
            results=results, filepath=filepath, title=f"Epoch {trainer.current_epoch}"
        )
        save_images_as_mp4(self.dirpath, fps=5, filepath=f"{self.dirpath}/video.mp4")
