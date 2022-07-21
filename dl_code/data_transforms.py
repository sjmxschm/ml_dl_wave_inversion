"""
Contains functions with different data transforms
"""

from typing import Tuple, Sequence

import numpy as np
import torchvision.transforms as transforms


def get_fundamental_transforms(
    inp_size: Tuple[int, int]) -> transforms.Compose:
    """
    Returns the core transforms necessary to feed the images to our model.

    Args:
        inp_size: tuple denoting the dimensions for input to the model

    Returns:
        fundamental_transforms: transforms.Compose with the fundamental
            transforms
    """

    fundamental_transforms = transforms.Compose([transforms.Resize(inp_size),
                                                 transforms.ToTensor(),
                                                 ])
    return fundamental_transforms


def get_fundamental_augmentation_transforms(
    inp_size: Tuple[int, int]) -> transforms.Compose:
    """
    Returns the core transforms in addition to augmentation.

    A few suggestions: Jittering, Flipping, Cropping, Rotating.
    for this case only Jittering makes sense, but try the other ones
    later too!
    Args:
        inp_size: tuple denoting the dimensions for input to the model

    Returns:
        fund_aug_transforms: transforms.Compose with fundamental and
            augmentation transforms
    """

    fund_aug_transforms = transforms.Compose([
        transforms.Resize(inp_size),
        # transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(),
        # transforms.RandomRotation(degrees=2),
        transforms.ToTensor(),
        # transforms.RandomCrop(),
    ])

    return fund_aug_transforms


def get_fundamental_normalization_transforms(
    inp_size: Tuple[int, int],
    pixel_mean: Sequence[float],
    pixel_std: Sequence[float]
) -> transforms.Compose:
    """
    Returns the core transforms in addition to normalization.

    These transforms will be applied to the validation set because we don't
    want to augment them, but we still want these other basic transformations.

    Args:
        inp_size: tuple denoting the dimensions for input to the model
        pixel_mean: image channel means, over all images of the raw dataset
        pixel_std: image channel standard deviations, for all images of the raw
            dataset

    Returns:
        fund_norm_transforms: transforms.Compose with the fundamental
            and normalization transforms
    """
    fund_norm_transforms = transforms.Compose(
        [transforms.Resize(inp_size),
         transforms.ToTensor(),
         transforms.Normalize(mean=pixel_mean, std=pixel_std),
         ])

    return fund_norm_transforms


def get_all_transforms(
    inp_size: Tuple[int, int],
    pixel_mean: Sequence[float],
    pixel_std: Sequence[float]
) -> transforms.Compose:
    """
    Returns the core, augmentation, and normalization transforms.

    These transforms will be applied to the training set.

    Args:
        inp_size: tuple denoting the dimensions for input to the model
        pixel_mean: image channel means, over all images of the raw dataset
        pixel_std: image channel standard deviations, for all images of the
            raw dataset

    Returns:
        all_transforms: transforms.Compose with all the transforms
    """

    all_transforms = transforms.Compose([  # in: PIL Image or torch.tensor, out: PIL Image or torch.tensor
                                          transforms.Resize(inp_size),

                                          # # in: PIL Image or torch.tensor, out: PIL Image or torch.tensor
                                          # transforms.RandomHorizontalFlip(),

                                          # in: PIL Image or torch.tensor, out: PIL Image or torch.tensor
                                          transforms.ColorJitter(),

                                          # in: PIL Image or torch.tensor, out: PIL Image or torch.tensor
                                          # transforms.RandomRotation(degrees=2),

                                          # in: PIL Image or np.array, out: torch.tensor
                                          transforms.ToTensor(),

                                          # # in: Tensor, out: tensor
                                          transforms.Normalize(mean=pixel_mean, std=pixel_std),
                                          ])


    return all_transforms
