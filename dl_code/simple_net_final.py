import torch
import torch.nn as nn


class SimpleNetFinal(nn.Module):
    def __init__(self):
        """
        Constructor for SimpleNetFinal class to define the layers and loss
        function.
        """
        super(SimpleNetFinal, self).__init__()

        self.conv_layers = None
        self.fc_layers = None
        self.loss_criterion = None
        self.test = None


        # third try with smaller network (reduce number of feature maps = filters)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=11),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3),
            nn.Conv2d(64, 70, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, padding=1),
            nn.Dropout(),
            nn.Conv2d(70, 500, kernel_size=5),
            nn.BatchNorm2d(num_features=500),
            nn.ReLU(inplace=True),
        )

        self.fc_layers = nn.Sequential(
            nn.Dropout(),
            nn.Linear(500, 1000),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1000, 1000),
            nn.ReLU(inplace=True),
            # nn.Linear(1000, 15),
            nn.Linear(1000, 2),  # I have only 2 output classes!
        )

        # -- define loss function:
        # self.loss_criterion = nn.MSELoss(reduction='mean')
        # Use negative Log likelihood as loss function
        self.loss_criterion = nn.NLLLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward pass with the network.

        Args:
            x: the (N,C,H,W) input images

        Returns:
            y: the (N,15) output (raw scores) of the net
        """
        model_output = None

        x = self.conv_layers(x)

        x = torch.flatten(x, 1)  # for deeper network

        model_output = self.fc_layers(x)

        return model_output
