"""
Contains helper functions which will help get the optimizer.
"""

import torch

from typing import Dict, Any


def get_optimizer(
        model: torch.nn.Module, config: Dict[str, Any]
) -> torch.optim.Optimizer:
    """
    Returns the optimizer initializer according to the config.

    Note: config has a minimum of three entries.
    Feel free to add more entries if you want, but do not change the name of
    the three existing entries.

    Args:
        model: the model to optimize for
        config: a dictionary containing parameters for the config

    Returns:
        optimizer: the optimizer
    """
    optimizer = None

    optimizer_type = config.get("optimizer_type", "adam")
    learning_rate = config.get("lr", 1e-3)
    weight_decay = config.get("weight_decay", 1e-5)
    momentum = config.get("momentum", 1e-3)

    if optimizer_type == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
    elif optimizer_type == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_type == 'adamax':
        optimizer = torch.optim.Adamax(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        print('Did not find an optimizer')

    return optimizer
