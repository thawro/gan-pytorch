from torch import Tensor, nn, optim
import torch
from torchinfo import summary
from src.utils.files import save_txt_to_file
from abc import abstractmethod


class BaseModel(nn.Module):
    def __init__(
        self,
        net: nn.Module,
        input_size: tuple[int, ...],
        input_names: list[str] = ["input"],
        output_names: list[str] = ["output"],
    ):
        super().__init__()
        self.net = net
        self.input_size = input_size
        self.input_names = input_names
        self.output_names = output_names

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

    def freeze(self) -> None:
        for param in self.parameters():
            param.requires_grad = False

    def _dummy_input(self, device: str) -> Tensor:
        """Return dummy input (according to input_size)."""
        return torch.randn(*self.input_size).to(device)

    def export_to_onnx(self, device: str, filepath: str):
        torch.onnx.export(
            self.net,
            self._dummy_input(device),
            filepath,
            verbose=False,
            input_names=self.input_names,
            output_names=self.output_names,
            export_params=True,
        )

    def export_summary_to_txt(self, filepath: str, depth: int = 4):
        model_summary = str(
            summary(
                self.net,
                input_size=self.input_size,
                depth=depth,
                col_names=["input_size", "output_size", "num_params", "params_percent"],
            )
        )
        if filepath is not None:
            save_txt_to_file(model_summary, filepath)

    def export_layers_description_to_txt(self, filepath: str) -> str:
        layers_description = str(self.net)
        if filepath is not None:
            save_txt_to_file(layers_description, filepath)
        return layers_description


class Generator(BaseModel):
    def __init__(self, net: nn.Module, input_size: tuple[int, ...]):
        input_names = ["noise"]
        output_names = ["image"]
        super().__init__(net, input_size, input_names, output_names)

    def forward(self, z: Tensor) -> Tensor:
        return self.net(z)


class Discriminator(BaseModel):
    def __init__(self, net: nn.Module, input_size: tuple[int, ...]):
        input_names = ["image"]
        output_names = ["validity"]
        super().__init__(net, input_size, input_names, output_names)

    def forward(self, images: Tensor) -> Tensor:
        validity = self.net(images)
        return validity
