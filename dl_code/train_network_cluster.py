"""
This script is meant to do the same as the Jupyter notebook maxWaveNet_local.ipynb
meaning training a specified network. This is done to be able to submit the training
process to the scheduler via an .pbs script
"""

import os
import torch
from torch import nn
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import copy

from typing import Union

try:
    from dl_code.data_transforms import (get_fundamental_normalization_transforms,
                                         get_all_transforms)
    from dl_code.simple_net import SimpleNet
    from dl_code.simple_net_final import SimpleNetFinal
    from dl_code.simple_net_big import SimpleNetBig
    from dl_code.my_resnet import MyResNet18, MyResNet34
    from dl_code.optimizer import get_optimizer
    from dl_code.runner import Trainer
    from dl_code.image_loader import ImageLoader
    from dl_code.dl_utils import save_trained_model_weights
    from dl_code.stats_helper import compute_mean_and_std
    from dl_code.confusion_matrix import (generate_confusion_data, generate_confusion_matrix,
                                             plot_confusion_matrix, get_pred_images_for_target,
                                             generate_and_plot_confusion_matrix)

except ModuleNotFoundError:
    from data_transforms import (get_fundamental_transforms,
                                         get_fundamental_augmentation_transforms,
                                         get_fundamental_normalization_transforms,
                                         get_all_transforms)
    from simple_net import SimpleNet
    from simple_net_final import SimpleNetFinal
    from simple_net_big import SimpleNetBig
    from my_resnet import MyResNet18, MyResNet34
    from optimizer import get_optimizer
    from runner import Trainer
    from image_loader import ImageLoader
    from dl_utils import save_trained_model_weights
    from stats_helper import compute_mean_and_std
    from confusion_matrix import (generate_confusion_data, generate_confusion_matrix,
                                             plot_confusion_matrix, get_pred_images_for_target,
                                             generate_and_plot_confusion_matrix)


def plot_loss_accuracy_history(trainer, save_plot: bool = False) -> None:
    """ plot loss and accuracy history as specified in runner.py """
    trainer.plot_loss_history(save=save_plot)
    trainer.plot_accuracy(save=save_plot)


def train_model(
        model: Union[SimpleNet, SimpleNetFinal, MyResNet18, nn.Module],
        optimizer: torch.optim,
        data_path: str,
        model_path: str,
        save_path: Path,
        inp_size: tuple,
        batch_size: int,
        save_freq: int,
        dataset_mean: float,
        dataset_std: float,
        num_epochs: int,
        load_from_disk: bool = False,
        load_trained_model: bool = False,
        is_cuda: bool = True
):
    if load_trained_model:
        model_info = torch.load(save_path / 'trained_SimpleNetBig_final.pt')
        model_state = copy.deepcopy(model_info['state_dict'])
        model.load_state_dict(model_state)
        model.eval()

    trainer = Trainer(data_dir=data_path,
                      model=model,
                      optimizer=optimizer,
                      model_dir=os.path.join(model_path, 'simple_net_big'),
                      train_data_transforms=get_all_transforms(inp_size, [dataset_mean], [dataset_std]),
                      val_data_transforms=get_fundamental_normalization_transforms(inp_size, [dataset_mean],
                                                                                   [dataset_std]),
                      val_split='normal',
                      batch_size=batch_size,  # 32,
                      load_from_disk=load_from_disk,
                      cuda=is_cuda,
                      save_freq=save_freq
                      )

    trainer.run_training_loop(num_epochs=num_epochs)

    return trainer


def main():
    # inp_size = (512, 512)
    inp_size = (1024, 1024)

    data_path = '../dl_code/data/'
    model_path = '../dl_code/model_checkpoints/'
    save_path = Path.cwd() / 'trained_models'

    torch.cuda.empty_cache()
    is_cuda = True
    is_cuda = is_cuda and torch.cuda.is_available()  # will turn off cuda if the machine doesn't have a GPU
    print(f'is_cuda = {is_cuda}')

    optimizer_config = {
        "optimizer_type": "adam",
        "lr": 5e-4,
        "weight_decay": 1.1e-3,
        "momentum": 1e-7
    }

    print(optimizer_config)

    network_name = "SimpleNetBig"
    my_model = SimpleNetBig()  # MyResNet18()

    # network_name = "ResNet18"
    # my_model = MyResNet18()

    # network_name = "ResNet34"
    # my_model = MyResNet34()

    optimizer = get_optimizer(my_model, optimizer_config)
    print(my_model)

    trainer = train_model(
        model=my_model,
        optimizer=optimizer,
        data_path=data_path,
        model_path=model_path,
        save_path=save_path,
        inp_size=inp_size,
        batch_size=25,  # 32,  #48,  # 32,  # 32
        save_freq=5,
        num_epochs=200,  # 250, 150, 200
        dataset_mean=0.3150067262879628,
        dataset_std=0.1554323642999201,
        load_from_disk=False,
        load_trained_model=False,
        is_cuda=True
    )

    save_trained_model_weights(
        my_model,
        save_path,
        trainer.train_loss_history,
        trainer.validation_loss_history,
        trainer.train_accuracy_history,
        trainer.validation_accuracy_history,
    )

    print('>> network training sufficiently completed!')

    plot_loss_accuracy_history(trainer, save_plot=True)

    generate_and_plot_confusion_matrix(
        my_model, trainer.val_dataset, network_name, use_cuda=True, save_matrix=True)


if __name__ == '__main__':
    main()
