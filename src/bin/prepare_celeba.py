"""Place img_align_celeba.zip file in datasets directory (DS_ROOT variable)

download it from http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
"""

import zipfile
from src.utils.config import DS_ROOT
import glob
import random
import shutil

ZIP_FILEPATH = str(DS_ROOT / "img_align_celeba.zip")
TRAIN_RATIO = 0.8
SEED = 42


def unzip_zip(file_path, dst_path):
    with zipfile.ZipFile(file_path, "r") as zip_ref:
        zip_ref.extractall(dst_path)


def main():
    random.seed(SEED)
    # 1. Unzip the file
    ds_path = DS_ROOT / "CelebA"
    ds_path.mkdir(parents=True, exist_ok=True)
    (ds_path / "train").mkdir(parents=True, exist_ok=True)
    (ds_path / "test").mkdir(parents=True, exist_ok=True)
    ds_path = str(ds_path)
    # unzip_zip(ZIP_FILEPATH, ds_path)

    # Count the files and prepare splits based on filenames
    all_filepaths = glob.glob(f"{ds_path}/img_align_celeba/*")
    n_images = len(all_filepaths)

    n_train_images = int(TRAIN_RATIO * n_images)
    train_filepaths = random.sample(all_filepaths, k=n_train_images)
    test_filepaths = list(set(all_filepaths).difference(set(train_filepaths)))

    for filepath in train_filepaths:
        shutil.move(filepath, f"{ds_path}/train")

    for filepath in test_filepaths:
        shutil.move(filepath, f"{ds_path}/test")


if __name__ == "__main__":
    main()
