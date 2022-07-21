import glob
import os
import os.path
from typing import Tuple

import numpy as np
from PIL import Image


def compute_mean_and_std(dir_name: str) -> Tuple[float, float]:
    """
    Computes the mean and the standard deviation of all images present within
    the directory.

    Note: converts the image in grayscale and then in [0,1] before computing the
    mean and standard deviation

    Mean = (1/n) * \sum_{i=1}^n x_i
    Variance = (1 / (n-1)) * \sum_{i=1}^n (x_i - mean)^2
    Standard Deviation = 1 / Variance

    Args:
        dir_name: the path of the root dir

    Returns:
        mean: mean value of the gray color channel for the dataset
        std: standard deviation of the gray color channel for the dataset
    """
    mean = None
    std = None

    # dir_name = dir_name.replace('\\', '/')

    all_img_paths = []
    for dirpath, dirnames, filenames in os.walk(dir_name):
        for filename in [f for f in filenames if f.endswith(".png")]:
            all_img_paths.append(os.path.join(dirpath, filename))

    all_imgs = []
    means = []
    stds = []
    for idx, image_path in enumerate(all_img_paths):
        print(image_path)
        img = np.array(Image.open(image_path).convert("L"))
        all_imgs = np.append(all_imgs, img.ravel() / 255.)
        if idx % 5 == 0:
            (mean, std) = (np.mean(all_imgs), np.std(all_imgs))
            means.append(mean)
            stds.append(std)
            all_imgs = []

    mean = np.mean(means)
    std = np.mean(stds)


    # all_imgs = np.array([])
    # # uses train and test images - for next time only use train images
    # for image in glob.iglob(dir_name + '**/*.jpg', recursive=True):  # recursive=True
    #     print(image)
    #     # import pdb; pdb.set_trace()
    #     img = np.array(Image.open(image).convert("L"))
    #     all_imgs = np.append(all_imgs, img.ravel()/255.)

    # calculate mean and std with pytorch internal functions
    # (mean, std) = (np.mean(all_imgs), np.std(all_imgs))

    return mean, std
