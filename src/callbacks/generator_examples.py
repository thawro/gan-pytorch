from __future__ import annotations

from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from src.model.module.trainer import Trainer

from .base import BaseCallback
from src.visualization.gan import plot_generator_results


class GeneratorExamplesPlotterCallback(BaseCallback):
    """Plot prediction examples"""

    def __init__(self, stage: str, filepath: str, inverse_prcessing: Callable):
        if filepath is None:
            filepath = "examples.jpg"
        self.filepath = filepath
        self.stage = stage
        self.inverse_processing = inverse_prcessing

    def on_epoch_end(self, trainer: Trainer) -> None:
        results = trainer.module.results[self.stage]
        if results is None:
            return
        plot_generator_results(results=results, filepath=self.filepath)
