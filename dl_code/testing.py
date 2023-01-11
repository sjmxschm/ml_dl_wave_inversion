import os
from typing import Tuple, Union
from pathlib import Path
import glob
import json
import numpy as np
import copy

import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

try:
    from dl_code.image_loader import ImageLoader
    from dl_code.dl_utils import compute_accuracy, compute_loss, compute_probabilities
    from dl_code.simple_net import SimpleNet
    from dl_code.my_resnet import MyResNet18, MyResNet34
except ModuleNotFoundError:
    from image_loader import ImageLoader
    from dl_utils import compute_accuracy, compute_loss, compute_probabilities
    from simple_net import SimpleNet
    from my_resnet import MyResNet18, MyResNet34

# from image_loader import ImageLoader
# from dl_utils import compute_accuracy, compute_loss
# from simple_net import SimpleNet
# from my_resnet import MyResNet18

"""
This file creates a class Tester which allows to run solely the test or
validation data and allows to push single images into the network

"""

# Vector = list[float]


class Tester:

    def __init__(
            self,
            data_dir: Path,
            data_set: Union["test", "test_hard"],
            model: Union[SimpleNet, MyResNet18, MyResNet34],
            model_name: str,
            save_dir: Path,
            data_transforms: transforms.Compose,
            sims_path: Path,
            batch_size: int = 20,
            load_trained_model: bool = True,
            is_checkpoint: bool = True,
            cuda: bool = False,
    ) -> None:
        """
        args
            - data_dir: Path, path object to folder with images to load
            - data_set: Union["test", "test_hard"],
            - model: Union[SimpleNet, MyResNet18],
            - model_name: str, name of model to load, e.g. 'trained_SimpleNetBig_final.pt'
            - optimizer: Optimizer, do not really need optimizer for validation
            - save_dir: Path, path to folder with stored final models
            - data_transforms: transforms.Compose,
            - batch_size: int = 20,
            - load_trained_model: bool = True, specify if pretrained gains should be loaded: YES!
            - is_checkpoint: bool = True, define if pretrained gains are from a checkpoint
            - cuda: bool = False,
        """

        self.sims_path = sims_path

        self.batch_size = batch_size

        dataloader_args = {"num_workers": 1, "pin_memory": True} if cuda else {}

        self.dataset = ImageLoader(
            str(data_dir), split=data_set, transform=data_transforms
        )
        self.dataset_length = self.dataset.__len__()

        self.data_loader = DataLoader(
            self.dataset, batch_size=batch_size, shuffle=False, **dataloader_args
        )

        # self.model_dir = save_dir

        self.model = model

        self.cuda = cuda
        if cuda:
            self.model.cuda()

        if load_trained_model:
            if is_checkpoint:
                checkpoint = torch.load(os.path.join(save_dir, "checkpoint.pt"))
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model_info = torch.load(save_dir / model_name)
                self.model.load_state_dict(copy.deepcopy(model_info['state_dict']))

        self.model.eval()

    def push_single(self, idx: int = 1) -> Tuple[int, int, float, float, float]:
        """
        returns: Tuple(ground truth class, predicted class, probability for that class,
            thickness, gap depth)

        args:
            - idx, index or name of img to load
        """

        assert idx <= self.dataset.__len__() - 1, "Index out of range for dataset"

        img, gt_class = self.dataset.__getitem__(idx)
        img = torch.unsqueeze(img, dim=0).cuda()

        logits = self.model(img)

        predicted_class = torch.argmax(logits, dim=1).cpu().numpy()
        prob = compute_probabilities(logits).cpu().detach().numpy().ravel()[predicted_class]

        thick, gd = self.get_simulation_information(idx, self.sims_path)

        return gt_class, predicted_class, prob, thick, gd

    def push_all(self):
        """ Returns a matrix with dimension:
                #samples x 5 (gt_class, predicted_class, probs, thick, gd)

        """
        f = np.zeros((self.dataset_length, 5))
        for idx in range(self.dataset_length):
            f[idx, :] = np.array(self.push_single(idx))

        return f

    def push_batch(self) -> Tuple[float, float, Tuple[float, float], int]:
        """
        Either calls push single to get the single values or better use the torch.Dataloader
        to load the dataset and loop over it like in runner.py

        Returns: accuracy, loss, probabilities, ground truth for batch
        """

        # loop over whole val set
        for (x, y) in self.data_loader:
            if self.cuda:
                x = x.cuda()
                y = y.cuda()

            logits = self.model(x)

            acc = compute_accuracy(logits, y).cpu().numpy()
            loss = compute_loss(self.model, logits, y, is_normalize=True).cpu().detach().numpy()
            probs = compute_probabilities(logits).cpu().detach().numpy()

        return acc, loss, probs, y.cpu().numpy()

    def get_simulation_information(self, idx: int, sims_path: Path) -> Tuple[float, float]:
        """
        Extracts the simulation information from the simulation info json file
        and returns thickness and gap depth.

        args:
            - idx: index of sample in dataset
        """
        img_paths = self.dataset.dataset[idx][0]
        id = img_paths[img_paths.rfind('/') + 1: img_paths.rfind('job') + 3]

        sim_info_path = glob.glob(str(sims_path) + f"/**/{id}_info.json", recursive=True)[0]
        with open(sim_info_path) as info_file:
            sim_info = json.load(info_file)

        thickness = sim_info['c_height'] * 1E6
        try:
            gap_depth = sim_info['geometric_properties']['cg_gap_depth'] * 1E6
        except KeyError:
            gap_depth = 0

        return thickness, gap_depth
