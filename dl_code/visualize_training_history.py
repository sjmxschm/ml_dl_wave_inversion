"""
If the training process was aborted because of some reason,
this file loads the epoch loss and accuracy history and plots it.

Make sure you deleted every other information from the file with the
training history, i.e. the file looks like:
_____________________________________________________________________________________________
Epoch:1 Train Loss:0.0272 Val Loss: 0.0268 Train Accuracy: 0.4816 Validation Accuracy: 0.4769
Epoch:2 Train Loss:0.0259 Val Loss: 0.0268 Train Accuracy: 0.5221 Validation Accuracy: 0.4769
Epoch:3 Train Loss:0.0250 Val Loss: 0.0267 Train Accuracy: 0.5257 Validation Accuracy: 0.4846
Epoch:4 Train Loss:0.0252 Val Loss: 0.0259 Train Accuracy: 0.5515 Validation Accuracy: 0.5308
...
_____________________________________________________________________________________________

created by: Max Schmitz on 11/10/2021
"""

from pathlib import Path
import numpy as np
import matplotlib
# # comment this in if you want to export to latex
matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

import matplotlib.pyplot as plt
from typing import Tuple

from scipy.signal import medfilt


def load_training_history(file_path: Path) -> Tuple[list, list, list, list]:
    """
    Loads training history from .out file

    args:
        - file_path: Path object to the .out file to be loaded
    """
    with open(file_path, 'r') as f:
        train_hist = f.readlines()
    train_hist = [elem.rstrip() for elem in train_hist]

    train_loss = [float(elem[elem.find('Train Loss:') +
                             len('Train Loss:'):elem.find('Train Loss:') +
                             len('Train Loss:') + 6])
                  for elem in train_hist]
    val_loss = [float(elem[elem.find('Val Loss:') +
                           len('Val Loss:') +
                           1:elem.find('Val Loss:') +
                           len('Val Loss:') + 7])
                for elem in train_hist]
    train_acc = [
        float(elem[elem.find('Train Accuracy:') +
                   len('Train Accuracy:') +
                   1:elem.find('Train Accuracy:') +
                   len('Train Accuracy:') + 7])
        for elem in train_hist]
    val_acc = [
        float(elem[elem.find('Validation Accuracy:') +
                   len('Validation Accuracy:') +
                   1:elem.find('Validation Accuracy:') +
                   len('Validation Accuracy:') + 7])
        for elem in train_hist]

    return train_loss, val_loss, train_acc, val_acc


def weight_func(elem_l, elem, elem_r, N):
    """
    takes element, and elements before (l) and after (r) and calculates
    weighted average or applies different weighting function

    args:
        - elem_l: element before current element
        - elem: current element
        - elem_r: element after current element
        - N: number of elements in set
    """
    # Laplacian Smoothing:
    # out = (elem_l + elem + elem_r)/3

    # Additive Smooting
    alpha = 2
    d = 2
    out = (elem + alpha)/(N + alpha * d)
    return out


def main():

    save_publication = True
    date = '02042022' #'01132022'

    #f_dir = Path.cwd() / 'epoch train history cluster' / '11092021_training'
    # f_dir = Path.cwd() / 'epoch train history cluster' / '11192021_training'
    # f_dir = Path.cwd() / 'epoch train history cluster' / '11272021_training'
    # f_dir = Path.cwd() / 'epoch train history cluster' / '12042021_training'
    # f_dir = Path.cwd() / 'epoch train history cluster' / '18062021_training'
    # f_dir = Path.cwd() / 'epoch train history cluster' / '01072022_training'
    # f_dir = Path.cwd() / 'epoch train history cluster' / '01132022_training'
    f_dir = Path.cwd() / 'epoch train history cluster' / f'{date}_training'
    f_name = 'train_network.out'

    train_loss, val_loss, train_acc, val_acc = load_training_history(f_dir / f_name)

    # val_acc_new = [weight_func(val_acc[idx-1], acc, val_acc[+1], len(val_acc))
    #                if idx not in {0, len(val_acc)} else acc for idx, acc in enumerate(val_acc)]
    # val_acc = val_acc_new

    train_loss = medfilt(train_loss)
    val_loss = medfilt(val_loss)

    train_acc = medfilt(train_acc)
    val_acc = medfilt(val_acc)

    if not save_publication:
        f = plt.figure(figsize=(10, 3))
        ax_l = f.add_subplot(121)
        ax_r = f.add_subplot(122)


        # plt.subplot(1, 2, 1)
        ax_l.plot(train_loss, "-b", linewidth=0.6, label="training") # linewidth=1
        ax_l.plot(val_loss, "-r", linewidth=0.6, label="validation")
        ax_l.set_title('Loss history')
        ax_l.legend()
        ax_l.set_ylabel("Loss")
        ax_l.set_xlabel("Epochs")

        # # plt.subplot(1, 2, 2)
        ax_r.plot(train_acc, "-b", linewidth=0.6, label="training")
        ax_r.plot(val_acc, "-r", linewidth=0.6, label="validation")
        ax_r.set_title("Accuracy history")
        ax_r.legend()
        ax_r.set_ylabel("Accuracy")
        ax_r.set_xlabel("Epochs")

        if False:
            plt.savefig(f'{f_dir}_sim_history.png', dpi=200)

        plt.tight_layout()
        plt.show()

    else:
        fig1 = plt.figure(4, dpi=300)  #
        fig1.set_size_inches(w=2.9, h=2)
        ax1 = plt.gca()
        # plt.subplot(1, 2, 1)
        ax1.plot(train_loss, "-b", linewidth=0.6, label="training")  # linewidth=1
        ax1.plot(val_loss, "-r", linewidth=0.6, label="validation")
        # ax.set_title('Loss history')
        ax1.legend()
        ax1.set_ylabel("Loss")
        ax1.set_xlabel("Epochs")

        pub_out_name = f_dir / f'{date}_training_loss_history.pgf'
        plt.savefig(pub_out_name, backend='pgf', format='pgf',
                    bbox_inches='tight', pad_inches=0.1, dpi=50)
        pub_out_name = f_dir / f'{date}_training_loss_history.png'
        plt.savefig(pub_out_name, backend='pgf', format='png',
                    bbox_inches='tight', pad_inches=0.1, dpi=200)

        plt.close(4)


        fig2 = plt.figure(5, dpi=300)  #
        fig2.set_size_inches(w=2.9, h=2)
        ax2 = plt.gca()
        ax2.plot(train_acc, "-b", linewidth=0.6, label="training")
        ax2.plot(val_acc, "-r", linewidth=0.6, label="validation")
        # ax.set_title("Accuracy history")
        ax2.legend()
        ax2.set_ylabel("Accuracy")
        ax2.set_xlabel("Epochs")

        pub_out_name = f_dir / f'{date}_training_acc_history.pgf'
        plt.savefig(pub_out_name, backend='pgf', format='pgf',
                    bbox_inches='tight', pad_inches=0.1, dpi=50)
        pub_out_name = f_dir / f'{date}_training_acc_history.png'
        plt.savefig(pub_out_name, backend='pgf', format='png',
                    bbox_inches='tight', pad_inches=0.1, dpi=200)

        plt.close(5)


if __name__ == '__main__':
    main()
