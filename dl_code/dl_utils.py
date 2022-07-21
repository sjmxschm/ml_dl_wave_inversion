"""
Utilities to be used along with the deep inversion model
"""
from typing import Union, Tuple
from pathlib import Path
from datetime import datetime, date

import torch
from torch import nn

try:
    from dl_code.simple_net import SimpleNet
    from dl_code.simple_net_final import SimpleNetFinal
    from dl_code.my_resnet import MyResNet18
except ModuleNotFoundError:
    from simple_net import SimpleNet
    from simple_net_final import SimpleNetFinal
    from my_resnet import MyResNet18


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute the accuracy given the prediction logits and the ground-truth
    labels.

    Args:
        logits: the (batch_size, num_classes) output of the forward pass
            through the model. For K classes, logits[k] (where 0 <= k < K)
            corresponds to the log-odds of class `k` being the correct one.
        labels: the ground truth label of shape (batch_size) for each instance
            in the batch

    Returns:
        accuracy: The accuracy of the predicted logits
            (number of correct predictions / total number of examples)
    """
    batch_accuracy = 0.0

    # # # -- single class problem:
    # # Use sigmoid function for binary outputs/loss function to obtain probabilities
    # m = nn.Sigmoid()
    #
    # probs = m(logits)
    #
    # thick0, thick1 = torch.zeros(logits.shape), torch.ones(logits.shape)
    # # thick enough when prob greater 0.5:
    # preds = torch.where(probs.cpu() > 0.5, thick1, thick0).view(-1)
    # batch_accuracy = torch.sum(torch.abs(preds - labels.cpu()))/logits.cpu().shape[0]

    # -- multi class problem:
    # find most likely label for each element in batch_size
    most_likely_labels = torch.argmax(logits, dim=1)

    # get the total number of correct predictions
    num_correct = torch.sum(torch.eq(most_likely_labels, labels))

    # obtain accuracy by dividing by the total number of examples (=batch_size)
    batch_accuracy = num_correct/logits.shape[0]

    return batch_accuracy


def compute_loss(
    model: Union[SimpleNet, SimpleNetFinal, MyResNet18],
    model_output: torch.Tensor,
    target_labels: torch.Tensor,
    is_normalize: bool = True,
) -> torch.Tensor:
    """
    Computes the loss between the model output and the target labels.

    Args:
        model: a model (which inherits from nn.Module)
        model_output: the raw scores output by the net
        target_labels: the ground truth class labels
        is_normalize: bool flag indicating that loss should be divided by the
            batch size
    Returns:
        the loss value
    """
    loss = None

    # target_labels_oh = nn.functional.one_hot(target_labels,num_classes=-1)

    # most_likely_output = model_output
    most_likely_output = model_output

    # Use softmax for multi class outputs to convert it into probabilities
    m = nn.LogSoftmax(dim=1)

    # Use sigmoid function for binary outputs/loss function
    # m = nn.Sigmoid()
    # most_likely_output = model_output.view(-1)

    if is_normalize:
        # loss = model.loss_criterion(m(most_likely_output), target_labels.to(torch.float32))/model_output.shape[0] # binary classification
        loss = model.loss_criterion(m(most_likely_output), target_labels)/model_output.shape[0]  # multi class
    else:
        loss = model.loss_criterion(most_likely_output, target_labels)

    return loss


def compute_probabilities(logits) -> Tuple[float, float]:
    """ Converts raw logits of the two classe into probabilites """
    m = nn.Softmax(dim=1)
    return m(logits)


def get_current_timestamp() -> str:
    """
    Returns current timestamp in format:
    MM-DD_hh-mm-ss
    """
    now = datetime.now()
    today = date.today().strftime("%m-%d")
    current_time = now.strftime("%H-%M-%S")
    return today + '_' + current_time


def save_trained_model_weights(
    model: Union[SimpleNet, SimpleNetFinal, MyResNet18, nn.Module],
    out_dir: Union[str, Path],
    train_loss_history: list = None,
    validation_loss_history: list = None,
    train_accuracy_history: list = None,
    validation_accuracy_history: list = None,
) -> None:
    """
    Saves the weights of a trained model along with class name and loss
    and accuracy history

    Args:
        - model: The model to be saved
        - out_dir: The path to the folder to store the save file in
        - train_loss_history,
        - validation_loss_history,
        - train_accuracy_history,
        - validation_accuracy_history,
    """
    class_name = model.__class__.__name__
    state_dict = model.state_dict()
    loss_histories = {'train loss history': train_loss_history,
                      'val loss history': validation_loss_history}
    acc_histories = {'train accuracy history': train_accuracy_history,
                     'val accuracy history': validation_accuracy_history}

    assert class_name in {"SimpleNet", "SimpleNetFinal", "SimpleNetBig",
                          "MyResNet18", "MyResNet34"}, "Please save only supported models"

    save_dict = {
        "class_name": class_name,
        "state_dict": state_dict,
        "loss histories": loss_histories,
        "accuracy histories": acc_histories,
    }
    timestmp = get_current_timestamp()
    torch.save(save_dict, f"{out_dir}/{timestmp}_trained_{class_name}_final.pt")
