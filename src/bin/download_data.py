"""Download all the data needed in the project"""
from geda.data_providers.mnist import MNISTDataProvider
from src.utils import DS_ROOT


def download_mnist():
    ds_path = DS_ROOT / "MNIST"
    dp = MNISTDataProvider(str(ds_path))
    dp.get_data()


if __name__ == "__main__":
    download_mnist()
