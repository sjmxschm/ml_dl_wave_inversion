import torch
import torch.nn as nn


class SimpleNetBig(nn.Module):
    def __init__(self):
        """
        Constructor for SimpleNetFinal class to define the layers and loss
        function.

        Note: Use 'mean' reduction in the loss_criterion. Read Pytorch's
        documention to understand what this means.
        """
        super(SimpleNetBig, self).__init__()

        self.conv_layers = None
        self.fc_layers = None
        self.loss_criterion = None
        self.test = None

        # # reduce input size from 512 to 224
        # self.down_sampling_layers = nn.Sequential(
        #     nn.Conv2d(1, 3, kernel_size=11, stride=2),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=11, stride=1),
        #     nn.Conv2d(3, 6, kernel_size=7),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=7, stride=1),
        #     nn.Conv2d(6, 10, kernel_size=6),
        #     nn.ReLU(inplace=True)
        # )

        # Use ReLU
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=11),  # out_feats was 64, in_feats was 1
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3),
            nn.Conv2d(16, 32, kernel_size=3, dilation=2),  # was 64->70
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3),
            nn.Dropout(),
            nn.Conv2d(64, 128, kernel_size=3),
            # nn.Conv2d(30, 40, kernel_size=5),  # was 70->500
            nn.BatchNorm2d(num_features=128),   # was 500
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3)
        )

        # Use LeakyReLU
        # self.conv_layers_leakyrelu = nn.Sequential(
        #     nn.Conv2d(1, 16, kernel_size=11),  # out_feats was 64, in_feats was 1
        #     nn.LeakyReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=3),
        #     nn.Conv2d(16, 32, kernel_size=3, dilation=2),  # was 64->70
        #     nn.LeakyReLU(inplace=True),
        #     nn.Conv2d(32, 64, kernel_size=5),
        #     nn.LeakyReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=3),
        #     nn.Dropout(),
        #     nn.Conv2d(64, 128, kernel_size=3),
        #     # nn.Conv2d(30, 40, kernel_size=5),  # was 70->500
        #     nn.BatchNorm2d(num_features=128),  # was 500
        #     nn.LeakyReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=3)
        # )

        self.adaptive_avg_pooling = nn.AdaptiveAvgPool2d(5)

        # num_ftrs = self.adaptive_avg_pooling.output_size
        num_ftrs = 128
        self.fc_layers = nn.Sequential(
            nn.Dropout(),
            nn.Linear(3200, 1000),  # input was 500, 4608 was it for 512 input size
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
        self.loss_criterion = nn.NLLLoss() # was before as written in thesis 11/01/2022
        # self.loss_criterion = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward pass with the network.

        Args:
            x: the (N,C,H,W) input images

        Returns:
            y: the (N,2) output (raw scores) of the net
        """

        # added this layer to make network work on input of 512 pixels
        # x = self.down_sampling_layers(x)

        # import pdb
        # pdb.set_trace() # torch.Size([1, 1, 512, 512]) - torch.Size([1, 1, 1024, 1024]) -

        x = self.conv_layers(x)
        # pdb.set_trace()
        # x = self.conv_layers_leakyrelu(x)

        # pdb.set_trace() # torch.Size([1, 128, 17, 17]) - torch.Size([1, 128, 36, 36]) -

        x = self.adaptive_avg_pooling(x)

        # pdb.set_trace() # torch.Size([1, 128, 1, 1]) - torch.Size([1, 128, 1, 1]) - torch.Size([1, 128, 5, 5])

        x = torch.flatten(x, 1)  # for deeper network

        # pdb.set_trace() # torch.Size([1, 128]) - - torch.Size([1, 3200])

        model_output = self.fc_layers(x)

        return model_output
