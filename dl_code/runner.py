import os

from typing import Tuple, Union

import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

try:
    from dl_code.image_loader import ImageLoader
    from dl_code.dl_utils import compute_accuracy, compute_loss
    from dl_code.simple_net import SimpleNet
    from dl_code.my_resnet import MyResNet18
except ModuleNotFoundError:
    from image_loader import ImageLoader
    from dl_utils import compute_accuracy, compute_loss
    from simple_net import SimpleNet
    from my_resnet import MyResNet18

import pdb
from datetime import date, datetime


def get_output_name(file_name: str = 'accuracy_history') -> str:
    """
    create the name of the plot output file. Name will be the same as
    the .csv data input file but with time stamp of creation added

    example output:
    -> accuracy_history_04-23_13-35-39.png


    args:
        - data_path: path to simulation data (.csv data)
        - file_name: name of simulation file which is analyzed
        - c_thick: thickness of coating
        - plt_type: which plotting type is used (cont or contf)
        - plt_res: which plotting resolution is used
    """
    now = datetime.now()
    current_time = now.strftime("%H-%M-%S")
    today = date.today()

    output_file = str(today.strftime("%m-%d")) \
                  + '_' \
                  + str(current_time) + '_' \
                  + file_name \
                  + '.png'
    return output_file


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name: str, fmt: str = ":f") -> None:
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self) -> None:
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class Trainer:
    """Class that stores model training metadata."""

    def __init__(
            self,
            data_dir: str,
            model: Union[SimpleNet, MyResNet18],
            optimizer: Optimizer,
            model_dir: str,
            train_data_transforms: transforms.Compose,
            val_data_transforms: transforms.Compose,
            val_split: Union['normal', 'hard'],
            batch_size: int = 100,
            load_from_disk: bool = True,
            cuda: bool = False,
            save_freq: int = 10,
    ) -> None:
        self.model_dir = model_dir

        self.model = model

        self.save_freq = save_freq

        self.cuda = cuda
        if cuda:
            self.model.cuda()

        dataloader_args = {"num_workers": 1, "pin_memory": True} if cuda else {}

        self.train_dataset = ImageLoader(
            data_dir, split="train", transform=train_data_transforms
        )

        self.train_loader = DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True, **dataloader_args
        )

        if val_split == "normal":
            self.val_dataset = ImageLoader(
                data_dir, split="test", transform=val_data_transforms
            )
            self.val_loader = DataLoader(
                self.val_dataset, batch_size=batch_size, shuffle=True, **dataloader_args
            )
        elif val_split == "hard":
            self.val_dataset = ImageLoader(
                data_dir, split="test_hard", transform=val_data_transforms
            )
            self.val_loader = DataLoader(
                self.val_dataset, batch_size=batch_size, shuffle=True, **dataloader_args
            )

        self.optimizer = optimizer

        self.train_loss_history = []
        self.validation_loss_history = []
        self.train_accuracy_history = []
        self.validation_accuracy_history = []

        # load the model from the disk if it exists
        if os.path.exists(model_dir) and load_from_disk:
            checkpoint = torch.load(os.path.join(self.model_dir, "checkpoint.pt"))
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        self.model.train()

    def save_model(self, epoch: int = 0) -> None:
        """
        Saves the model state and optimizer state on the dict for current epoch
        """
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            os.path.join(self.model_dir, f"checkpoint_{epoch}.pt"),
        )

    def run_training_loop(self, num_epochs: int) -> None:
        """Train for num_epochs, and validate after every epoch."""
        for epoch_idx in range(num_epochs):
            train_loss, train_acc = self.train_epoch()

            self.train_loss_history.append(train_loss)
            self.train_accuracy_history.append(train_acc)

            val_loss, val_acc = self.validate()
            self.validation_loss_history.append(val_loss)
            self.validation_accuracy_history.append(val_acc)

            if epoch_idx % self.save_freq == 0:
                self.save_model(epoch=epoch_idx)

            print(
                f"Epoch:{epoch_idx + 1}"
                + f" Train Loss:{train_loss:.4f}"
                + f" Val Loss: {val_loss:.4f}"
                + f" Train Accuracy: {train_acc:.4f}"
                + f" Validation Accuracy: {val_acc:.4f}"
            )

    def train_epoch(self) -> Tuple[float, float]:
        """Implements the main training loop."""
        self.model.train()

        train_loss_meter = AverageMeter("train loss")
        train_acc_meter = AverageMeter("train accuracy")

        # pdb.set_trace()

        # loop over each mini batch
        for (x, y) in self.train_loader:
            if self.cuda:
                x = x.cuda()  # batch of images from database
                y = y.cuda()  # corresponding label

            n = x.shape[0]
            logits = self.model(x)
            batch_acc = compute_accuracy(logits, y)
            train_acc_meter.update(val=batch_acc, n=n)

            batch_loss = compute_loss(self.model, logits, y, is_normalize=True)
            train_loss_meter.update(val=float(batch_loss.cpu().item()), n=n)

            self.optimizer.zero_grad()
            batch_loss.backward()
            self.optimizer.step()

        return train_loss_meter.avg, train_acc_meter.avg

    def validate(self) -> Tuple[float, float]:
        """Evaluate on held-out split (either val or test)"""
        self.model.eval()

        val_loss_meter = AverageMeter("val loss")
        val_acc_meter = AverageMeter("val accuracy")

        # loop over whole val set
        for (x, y) in self.val_loader:
            if self.cuda:
                x = x.cuda()
                y = y.cuda()

            n = x.shape[0]
            logits = self.model(x)

            batch_acc = compute_accuracy(logits, y)
            val_acc_meter.update(val=batch_acc, n=n)

            batch_loss = compute_loss(self.model, logits, y, is_normalize=True)
            val_loss_meter.update(val=float(batch_loss.cpu().item()), n=n)

        return val_loss_meter.avg, val_acc_meter.avg

    def plot_loss_history(self, save=True) -> None:
        """Plots the loss history"""
        plt.figure()
        epoch_idxs = range(len(self.train_loss_history))

        plt.plot(epoch_idxs, self.train_loss_history, "-b", label="training")
        plt.plot(epoch_idxs, self.validation_loss_history, "-r", label="validation")
        plt.title("Loss history")
        plt.legend()
        plt.ylabel("Loss")
        plt.xlabel("Epochs")
        if save:
            out_name = get_output_name(file_name='loss_history')
            plt.savefig('../dl_code/figures/' + out_name)
            # plt.savefig('../dl_code/figures/loss_history.png')
        plt.show()

    def plot_accuracy(self, save=True) -> None:
        """Plots the accuracy history"""
        plt.figure()
        epoch_idxs = range(len(self.train_accuracy_history))
        # plt.plot(epoch_idxs, self.train_accuracy_history, "-b", label="training")
        # plt.plot(epoch_idxs, self.validation_accuracy_history, "-r", label="validation")

        train_accuracy_history = [elem.cpu() for elem in self.train_accuracy_history]
        validation_accuracy_history = [elem.cpu() for elem in self.validation_accuracy_history]

        plt.plot(epoch_idxs, train_accuracy_history, "-b", label="training")
        plt.plot(epoch_idxs, validation_accuracy_history, "-r", label="validation")

        plt.title("Accuracy history")
        plt.legend()
        plt.ylabel("Accuracy")
        plt.xlabel("Epochs")
        if save:
            out_name = get_output_name(file_name='accuracy_history')
            plt.savefig('../dl_code/figures/' + out_name)
        plt.show()


