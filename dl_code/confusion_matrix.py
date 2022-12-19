from typing import Tuple, Sequence, List

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
import numpy as np
from matplotlib import pyplot as plt

from datetime import date, datetime

try:
    from dl_code.image_loader import ImageLoader
    from dl_code.dl_utils import compute_loss
except ModuleNotFoundError:
    from image_loader import ImageLoader
    from dl_utils import compute_loss


def generate_confusion_data(
        model: nn.Module,
        dataset: ImageLoader,
        use_cuda: bool = False,
) -> Tuple[Sequence[int], Sequence[int], Sequence[str]]:
    """
    Get the accuracy on the val/train dataset.

    Args:
        model: Model to generate confusion matrix data for
        dataset: ImageLoader dataset that corresponds to training or val data
        use_cuda: whether to evaluate on CPU or GPU

    Returns:
        targets: a numpy array of shape (N) containing the targets indices
        preds: a numpy array of shape (N) containing the predicted indices
        class_labels: A list containing the class labels at the index of their
            label_number (e.g., if the labels are
            {"Cat": 0, "Monkey": 2, "Dog": 1}, the return value should be
            ["Cat", "Dog", "Monkey"])
    """

    batch_size = 32
    cuda = use_cuda and torch.cuda.is_available()
    dataloader_args = {"num_workers": 1, "pin_memory": True} if cuda else {}
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, **dataloader_args)

    preds = np.zeros(len(dataset)).astype(np.int32)
    targets = np.zeros(len(dataset)).astype(np.int32)
    label_to_idx = dataset.get_classes()  # label_to_idx = dict(zip(class_names, indices))
    class_labels = [""] * len(label_to_idx)

    model.eval()

    for idx, (x, y) in enumerate(loader):
        if use_cuda:
            x = x.cuda()  # path to image in database
            y = y.cuda()  # corresponding label

        logits = model(x)
        most_likely_label = torch.argmax(logits, dim=1)

        preds[idx * y.shape[0]: idx * y.shape[0] + most_likely_label.cpu().shape[0]] = most_likely_label.cpu()
        targets[idx * y.shape[0]: idx * y.shape[0] + y.cpu().shape[0]] = y.cpu()

        # class_labels.append(label_to_idx.index(y))

    # # get class labels from label_to_idx
    label_to_idx_sorted = {k: v for k, v in sorted(label_to_idx.items(), key=lambda item: item[1])}
    class_labels = label_to_idx_sorted.keys()

    model.train()

    return targets, preds, class_labels
    # return targets.cpu().numpy(), preds.cpu().numpy(), class_labels


def generate_confusion_matrix(
        targets: np.ndarray, preds: np.ndarray, num_classes: int, normalize=True
) -> np.ndarray:
    """
    Generate the actual confusion matrix values.

    The confusion matrix is a num_classes x num_classes matrix that shows the
    number of classifications made to a predicted class, given a ground truth
    class.

    If the classifications are:
        ground_truths: [1, 0, 1, 2, 0, 1, 0, 2, 2]
        predicted:     [1, 1, 0, 2, 0, 1, 1, 2, 0]

    Then the confusion matrix is:
        [1 2 0],
        [1 1 0],
        [1 0 2],

    Each ground_truth value corresponds to a row,
    and each predicted value is a column

    A confusion matrix can be normalized by dividing all the entries of
    each ground_truth prior by the number of actual instances of the ground truth
    in the dataset.

    Args:
        targets: a numpy array of shape (N) containing the targets indices
        preds: a numpy array of shape (N) containing the predicted indices
        num_classes: Number of classes in the confusion matrix
        normalize: Whether to normalize the confusion matrix or not

    Returns:
        confusion_matrix: a (num_classes, num_classes) numpy array representing
            the confusion matrix
    """

    confusion_matrix = np.zeros((num_classes, num_classes))

    for target, prediction in zip(targets, preds):
        confusion_matrix[target, prediction] += 1

    if normalize:
        for row in range(confusion_matrix.shape[0]):
            confusion_matrix[row, :] = confusion_matrix[row, :] / np.sum(confusion_matrix[row, :])

    return confusion_matrix


def plot_confusion_matrix(
        confusion_matrix: np.ndarray, class_labels: Sequence[str],
        network_name: str = 'ResNet18', save_matrix: bool = False
) -> None:
    """
    Plots the confusion matrix

    Args:
        confusion_matrix: a (num_classes, num_classes) numpy array representing
            the confusion matrix
        class_labels: A list containing the class labels at the index of their
            label_number (e.g., if the labels are
            {"Cat": 0, "Monkey": 2, "Dog": 1}, the return value should be
            ["Cat", "Dog", "Monkey"]). The length of class_labels should be
            num_classes
        network_name: specify which network is used for figure saving
        save_matrix: specify if image of matrix should be saved
    """
    fig, ax = plt.subplots()
    fig.set_figheight(10)
    fig.set_figwidth(10)

    num_classes = len(class_labels)

    ax.imshow(confusion_matrix, cmap="Blues")

    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xticklabels(class_labels)
    ax.set_yticklabels(class_labels)

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Ground-Truth label")
    ax.set_title("Confusion Matrix")

    plt.setp(
        ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(num_classes):
        for j in range(num_classes):
            _ = ax.text(
                j,
                i,
                f"{confusion_matrix[i, j]:.2f}",
                ha="center",
                va="center",
                color="black",
            )
    # plt.savefig('../figures/confusion_matrix.png')
    if save_matrix:
        now = datetime.now()
        current_time = now.strftime("%H-%M-%S")
        today = date.today().strftime("%m-%d")

        plt.savefig(f'./figures/{today}_{current_time}_{network_name}_confusion_matrix.png')
    plt.show()


def generate_and_plot_confusion_matrix(
        model: nn.Module,
        dataset: ImageLoader,
        network_name: str = 'ResNet18',
        use_cuda: bool = False,
        save_matrix: bool = False
) -> None:
    """
    Runs the entire confusion matrix pipeline for convenience

    Args:
        model: Model to generate confusion matrix data for
        dataset: The ImageLoader dataset that corresponds to training or
            validation data
        network_name: specify which network is used for this
        use_cuda: Whether to evaluate on CPU or GPU
        save_matrix: specify if matrix should be saved or not
    """

    targets, predictions, class_labels = generate_confusion_data(
        model, dataset, use_cuda=use_cuda
    )

    confusion_matrix = generate_confusion_matrix(
        np.array(targets, dtype=np.int32),
        np.array(predictions, np.int32),
        len(class_labels),
    )

    plot_confusion_matrix(confusion_matrix, class_labels, network_name, save_matrix)


def get_pred_images_for_target(
        model: nn.Module,
        dataset: ImageLoader,
        predicted_class: int,
        target_class: int,
        use_cuda: bool = False,
) -> Sequence[str]:
    """
    Returns a list of image paths that correspond to a particular prediction
    for a given target class

    Args:
        model: Model to generate confusion matrix data for
        dataset: The ImageLoader dataset that corresponds to training or
            validation data
        predicted_class: The class predicted by the model
        target_class: The actual class of the image
        use_cuda: Whether to evaluate on CPU or GPU

    Returns:
        valid_image_paths: Image paths that are classified as <predicted_class>
            but actually belong to <target_class>
    """
    model.eval()

    dataset_list = dataset.dataset  # the path to each single image in the test data set
    # print(len(dataset_list))

    # # _________ select only the images from the target class: _________
    indices = []
    image_paths = []
    for i, (image_path, class_label) in enumerate(dataset_list):
        if class_label == target_class:
            indices.append(i)
            image_paths.append(image_path)
    subset = Subset(dataset, indices)

    dataloader_args = {"num_workers": 1, "pin_memory": True} if use_cuda else {}
    loader = DataLoader(subset, batch_size=32, shuffle=False, **dataloader_args)
    preds = []
    for i, (inp, _) in enumerate(loader):
        if use_cuda:
            inp = inp.cuda()  # path to image
        logits = model(inp)
        p = torch.argmax(logits, dim=1)
        preds.append(p)
    predictions = torch.cat(preds, dim=0).cpu().tolist()
    valid_image_paths = [
        image_paths[i] for i, p in enumerate(predictions) if p == predicted_class
    ]
    model.train()
    return valid_image_paths


def sort_dataset_by_loss(
        model: nn.Module,
        dataset: ImageLoader,
        class_labels: Sequence[str],
        use_cuda: bool = False,
        elems_considered=10,
) -> List[Tuple[str, str]]:
    """
    Function is based on get_pred_images_for_target from above

    Returns a list of image paths with the highest loss in the dataset

    Args:
        model: Model to generate confusion matrix data for
        dataset: The ImageLoader dataset that corresponds to training or
        use_cuda: Whether to evaluate on CPU or GPU
        elems_considered: number of elements with highest loss to be returned

    Returns:
        highest_loss_image_paths: Image paths that with the highest loss
    """
    model.eval()

    dataset_list = dataset.dataset  # contains the path to each single image in the test data set

    image_paths = [dataset_list[i][0] for i in range(len(dataset_list))]

    image_labels = [dataset_list[i][1] for i in range(len(dataset_list))]

    dataloader_args = {"num_workers": 1, "pin_memory": True} if use_cuda else {}
    loader = DataLoader(
        dataset,
        batch_size=1,  # set batch size to 1 to feed in single images
        shuffle=False,
        **dataloader_args
    )

    loss = []
    preds = []
    ind_class_changes = []
    class_loss = 0
    class_losses = []
    for i, (x, y) in enumerate(loader):
        if use_cuda:
            x = x.cuda()  # path to image in database
            y = y.cuda()  # corresponding label
        logits = model(x)

        p = torch.argmax(logits, dim=1).cpu()
        preds.append(int(p))
        image_loss = compute_loss(model, logits, y, is_normalize=True).cpu()
        loss.append(float(image_loss))
        class_loss += float(image_loss) / 100
        # if class changes:
        if i == 1499 or image_labels[i] != image_labels[i + 1]:  # i<1480
            ind_class_changes.append(i)
            class_losses.append(class_loss)
            class_loss = 0

    print('ind_class_changes =', ind_class_changes)

    plt.plot(loss)
    vz = 1
    for j in range(len(ind_class_changes)):
        plt.axvline(ind_class_changes[j], c='r', label='loss image')
        plt.plot([ind_class_changes[j] - 100, ind_class_changes[j]],
                 [class_losses[j], class_losses[j]], label='loss class', color='pink')
        plt.text(ind_class_changes[j], 3 + 0.25 * vz * np.random.uniform(0, 1), [*class_labels][j], size='x-small',
                 ha='right')
        vz = -vz

    # plt.plot([1400, 1500], [class_losses[-1], class_losses[-1]], label='loss class',color='pink')
    # plt.text(1500, 3.5 + 0.05 * vz, 'tall building', size='x-small', ha='center')

    plt.title('Loss over images')
    plt.xlabel('image number')
    plt.ylabel('loss')

    # plt.savefig('../figures/images_with_highest_loss/loss_plot.png',dpi=300)
    plt.show()

    loss_sorted = sorted(loss, reverse=True)  # sort in ascending order

    l_indices = []
    for elem in range(elems_considered):
        l_indices.append(loss.index(loss_sorted[elem]))  # append index of elem-highest loss

    # check if max value of loss is included in l_indices
    max_value = max(loss)
    max_index = loss.index(max_value)
    # print(f'max index = {max_index}')

    # [*newdict] unpacks dict_keys into list
    highest_loss_image_paths = [(image_paths[i], [*class_labels][preds[i]]) for i in l_indices]

    model.train()
    return highest_loss_image_paths
