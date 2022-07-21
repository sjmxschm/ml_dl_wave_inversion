from dl_code.image_loader import ImageLoader
from dl_code.data_transforms import get_fundamental_transforms
from pathlib import Path

import numpy as np
import torch

# PROJ_ROOT = Path(__file__).resolve().parent.parent
PROJ_ROOT = Path(__file__).resolve().parent
# change forward slash to backslash since later on forward slashes as strings are used
# (this is dumb since it destroyes the usefulness of the pathlib lib but it is how it is)
PROJ_ROOT = str(PROJ_ROOT).replace("\\","/")


def test_dataset_length():
    train_image_loader = ImageLoader(
        root_dir=f"{PROJ_ROOT}/data/",
        split="train",
        transform=get_fundamental_transforms(inp_size=(64, 64)),
    )

    test_image_loader = ImageLoader(
        root_dir=f"{PROJ_ROOT}/data/",
        split="test",
        transform=get_fundamental_transforms(inp_size=(64, 64)),
    )

    assert train_image_loader.__len__() == 4
    assert test_image_loader.__len__() == 4


def test_unique_vals():
    train_image_loader = ImageLoader(
        root_dir=f"{PROJ_ROOT}/data/",
        split="train",
        transform=get_fundamental_transforms(inp_size=(64, 64)),
    )

    item1 = train_image_loader.__getitem__(0)
    item2 = train_image_loader.__getitem__(1)

    assert not torch.allclose(item1[0], item2[0])


def test_class_values():
    """ """
    test_image_loader = ImageLoader(
        root_dir=f"{PROJ_ROOT}/data/",
        split="test",
        transform=get_fundamental_transforms(inp_size=(64, 64)),
    )

    class_labels = test_image_loader.class_dict
    class_labels = {ele.lower(): class_labels[ele] for ele in class_labels}

    # should be 2 unique keys and 2 unique values in the dictionary
    assert len(set(class_labels.values())) == 2
    assert len(set(class_labels.keys())) == 2

    # indices must be ordered from [0,1] only
    assert set(list(range(2))) == set(class_labels.values())

    # must be ordered alphabetically
    assert class_labels['-1'] == 0
    assert class_labels['1'] == 1


def test_load_img_from_path():
    test_image_loader = ImageLoader(
        root_dir=f"{PROJ_ROOT}/data/",
        split="train",
        transform=get_fundamental_transforms(inp_size=(64, 64)),
    )
    im_path = f"{PROJ_ROOT}/data/train/-1/09-23_00-54-33_max_analysis_job_10-05_19-47-47_contf_300_100_cnn.png"

    im_np = np.asarray(test_image_loader.load_img_from_path(im_path))
    # with open('example_disp_image.txt', 'w') as f:
    #     np.savetxt(f, im_np)
    # from matplotlib import pyplot as plt
    # plt.imshow(im_np)
    # plt.show()

    expected_data = np.loadtxt(f"{PROJ_ROOT}/data/example_disp_image.txt")

    assert np.allclose(expected_data, im_np)


if __name__ == "__main__":
    test_load_img_from_path()
