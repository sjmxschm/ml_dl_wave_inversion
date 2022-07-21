import torch
import torch.nn as nn

from torchvision.models import resnet18


class SimpleNet(nn.Module):
    def __init__(self):
        """
        Constructor for SimpleNet class to define the layers and loss function.

        Note: Use 'mean' reduction in the loss_criterion. Read Pytorch's
        documention to understand what this means.
        """
        super(SimpleNet, self).__init__()

        self.conv_layers = None
        self.fc_layers = None
        self.loss_criterion = None

        # model = resnet18(pretrained=True)

        # -- define convolutional layers:
        # num_feature_maps_conv1 = 10
        kernel_size_conv = 5
        kernel_size_pool = 3

        # no padding necessary when I want to reduce the resolution!
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=kernel_size_conv, stride=1),  #,
                      # padding=(kernel_size_conv // 2)),
            nn.MaxPool2d(kernel_size=kernel_size_pool),
            nn.ReLU(),
            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=kernel_size_conv, stride=1),
            nn.MaxPool2d(kernel_size=kernel_size_pool),
            nn.ReLU(),
            nn.Flatten(),
        )

        # -- define fully connected layers:
        input_fc = 500
        self.fc_layers = nn.Sequential(
            nn.Linear(in_features=input_fc, out_features=100),
            nn.ReLU(),
            # nn.Linear(in_features=100, out_features=15),
            nn.Linear(in_features=100, out_features=2),  # I have only two classes to predict!
            # or maybe I just have one class to predict (thick enough)?!
            # nn.Sigmoid(),
            # nn.Softmax()
        )

        # -- define loss function:
        # self.loss_criterion = nn.MSELoss(reduction='mean')
        # Use negative Log likelihood as loss function - works best with multi class
        # self.loss_criterion = nn.NLLLoss()

        # try 2 class classification loss now
        self.loss_criterion = nn.CrossEntropyLoss()  # seems to be good for binary classification - maybe, works
        #                                                                                             with multi class
        # self.loss_criterion = nn.BCELoss()  # standard loss for binary classification
        # self.loss_criterion = nn.SoftMarginLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward pass with the network.

        Args:
            x: the (N,C,H,W) input images

        Returns:
            y: the (N,15) output (raw scores) of the net
        """
        model_output = None

        identity = x

        conv_features = self.conv_layers(x)
        # flattening is already part of the conv_layers

        model_output = self.fc_layers(conv_features)

        return model_output
