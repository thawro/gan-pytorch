"""Train the model"""

import torch
from functools import partial

from torch import nn
from src.data import DataModule, MNISTDataset
from src.data.transforms import DataTransform
from src.logging import TerminalLogger, get_pylogger
from src.callbacks import (
    LoadModelCheckpoint,
    MetricsPlotterCallback,
    MetricsSaverCallback,
    ModelSummary,
    SaveModelCheckpoint,
    GeneratorExamplesPlotterCallback,
)
from src.model.architectures import (
    GANGenerator,
    GANDiscriminator,
    DCGANGenerator,
    DCGANDiscriminator,
    ConditionalDCGANGenerator,
    ConditionalDCGANDiscriminator,
)
from src.model.gan import GANModel, ConditionalGANModel

from src.model.loss import GANLoss, WeightedLoss
from src.model.module import Trainer, GANModule, ConditionalGANModule
from src.model.utils import seed_everything

from src.utils import DS_ROOT, NOW, ROOT

log = get_pylogger(__name__)

EXPERIMENT_NAME = "test"

CFG = {
    "seed": 42,
    "dataset": "MNIST",
    "latent_dim": 100,
    "image_size": (image_size := 64),
    "image_channels": 1,
    "transform": {"mean": 0.5, "std": 0.5, "image_size": image_size},
    "max_epochs": 500,
    "batch_size": 100,
    "device": "cuda",
    "limit_batches": -1,
    "gan_type": "conditional_dcgan",
    "n_classes": 10,
}

CFG["image_shape"] = (CFG["image_channels"], CFG["image_size"], CFG["image_size"])

if CFG["gan_type"] in ["gan", "dcgan"]:
    CFG["input_size"] = (1, CFG["latent_dim"])
elif CFG["gan_type"] in ["conditional_dcgan"]:
    CFG["input_size"] = [(1, CFG["latent_dim"]), (1,)]

if CFG["limit_batches"] != -1:
    EXPERIMENT_NAME = "debug"

RUN_NAME = f"{CFG['gan_type']}/{NOW}"
CFG["logs_path"] = str(ROOT / "results" / EXPERIMENT_NAME / RUN_NAME)


def weights_init(model):
    classname = model.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)


def create_datamodule(
    ds_path: str,
    train_transform: DataTransform,
    inference_transform: DataTransform,
    batch_size: int,
) -> DataModule:
    train_ds = MNISTDataset(ds_path, "train", train_transform)
    val_ds = MNISTDataset(ds_path, "test", inference_transform)
    return DataModule(train_ds=train_ds, val_ds=val_ds, test_ds=None, batch_size=batch_size)


def create_callbacks(
    logger: TerminalLogger, input_size: tuple[int, ...], ckpt_path: str | None
) -> list:
    ckpt_saver_params = dict(ckpt_dir=logger.ckpt_dir, stage="val", mode="min")
    summary_filepath = str(logger.model_dir / "model_summary.txt")
    metrics_plot_path = str(logger.log_path / "metrics.jpg")
    metrics_yaml_path = str(logger.log_path / "metrics.yaml")
    examples_dirpath = logger.log_path / "examples"
    examples_dirpath.mkdir(exist_ok=True, parents=True)
    callbacks = [
        MetricsPlotterCallback(metrics_plot_path),
        MetricsSaverCallback(metrics_yaml_path),
        ModelSummary(input_size=input_size, depth=4, filepath=summary_filepath),
        SaveModelCheckpoint(name="best_G", metric="G_loss", **ckpt_saver_params),
        SaveModelCheckpoint(name="best_D", metric="D_loss", **ckpt_saver_params),
        SaveModelCheckpoint(name="last", last=True, top_k=0, **ckpt_saver_params),
        GeneratorExamplesPlotterCallback("val", str(examples_dirpath)),
    ]
    if ckpt_path is not None:
        callbacks.append(LoadModelCheckpoint(ckpt_path))
    return callbacks


def create_gan_model(latent_dim: int, image_shape: tuple[int, ...]) -> GANModel:
    generator = GANGenerator(latent_dim, image_shape)
    discriminator = GANDiscriminator(image_shape)
    return GANModel(generator, discriminator)


def create_dcgan_model(latent_dim: int, image_shape: tuple[int, ...]) -> GANModel:
    mid_channels = 64
    img_channels = image_shape[0]
    generator = DCGANGenerator(latent_dim, mid_channels, img_channels)
    discriminator = DCGANDiscriminator(img_channels, mid_channels, input_size=image_shape)
    return GANModel(generator, discriminator)


def create_conditional_dcgan_model(
    latent_dim: int, image_shape: tuple[int, ...], n_classes: int
) -> ConditionalGANModel:
    mid_channels = 64
    img_channels = image_shape[0]
    generator = ConditionalDCGANGenerator(n_classes, latent_dim, mid_channels, img_channels)
    discriminator = ConditionalDCGANDiscriminator(
        n_classes, img_channels, mid_channels, input_size=image_shape
    )
    return ConditionalGANModel(generator, discriminator)


def create_module(
    gan_type: str, latent_dim: int, image_shape: tuple[int, ...]
) -> GANModule | ConditionalGANModule:
    if gan_type == "gan":
        create_model = create_gan_model
        Module = GANModule
    elif gan_type == "dcgan":
        create_model = create_dcgan_model
        Module = GANModule
    elif gan_type == "conditional_dcgan":
        create_model = partial(create_conditional_dcgan_model, n_classes=CFG["n_classes"])
        Module = ConditionalGANModule
    model = create_model(latent_dim, image_shape)
    model.apply(weights_init)
    return Module(model, loss_fn=GANLoss(WeightedLoss(nn.BCELoss())))


def main() -> None:
    seed_everything(CFG["seed"])
    torch.set_float32_matmul_precision("medium")
    ds_path = str(DS_ROOT / CFG["dataset"])
    train_transform = DataTransform(is_train=True, **CFG["transform"])
    inference_transform = DataTransform(is_train=False, **CFG["transform"])

    datamodule = create_datamodule(ds_path, train_transform, inference_transform, CFG["batch_size"])

    module = create_module(CFG["gan_type"], CFG["latent_dim"], CFG["image_shape"])

    logger = TerminalLogger(CFG["logs_path"], config=CFG)

    callbacks = create_callbacks(logger, CFG["input_size"], CFG.get("ckpt_path", None))

    trainer = Trainer(
        logger=logger,
        device=CFG["device"],
        callbacks=callbacks,
        max_epochs=CFG["max_epochs"],
        limit_batches=CFG["limit_batches"],
    )
    trainer.fit(module, datamodule)


if __name__ == "__main__":
    main()
