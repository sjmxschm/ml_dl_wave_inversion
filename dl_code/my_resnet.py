import torch
import torch.nn as nn

from torchvision.models import resnet18, resnet34


class MyResNet18(nn.Module):
    def __init__(self):
        """
        Initializes network layers.

        Note: Do not forget to freeze the layers of ResNet except the last one.
        Note: Use 'mean' reduction in the loss_criterion. Read Pytorch's
            documention to understand what this means.

        Download pretrained ResNet using pytorch's API. No, use CrossEntropyLoss
            first

        Hint: see the import statements

        find original constructor of ResNet here:
        https://github.com/pytorch/vision/blob/21153802a3086558e9385788956b0f2808b50e51/torchvision/models/resnet.py#L161
        """
        super().__init__()

        self.conv_layers = None
        self.fc_layers = None
        self.loss_criterion = None

        # # reduce input size from 512 to 224 - results do not look good- deprecated
        # self.down_sampling_layers = nn.Sequential(
        #     nn.Conv2d(1, 3, kernel_size=11, stride=2),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=11, stride=1),
        #     nn.Conv2d(3, 6, kernel_size=7),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=7, stride=1),
        #     nn.Conv2d(6, 3, kernel_size=6),
        #     nn.ReLU(inplace=True)
        # )

        self.adaptive_avg_pooling = nn.AdaptiveAvgPool2d(1)

        # model = resnet18(pretrained=True)
        model = resnet18(pretrained=False)
        # self.model = resnet18(pretrained=False)

        # # unfreeze layers:
        # for param in model.parameters():
        #     ''' freeze all model layers '''
        #     param.requires_grad = False

        self.conv_layers = nn.Sequential(
            *list(model.children())[:-1]
        )

        num_ftrs = model.fc.in_features
        self.fc_layers = nn.Sequential(
            nn.Linear(num_ftrs, 2),  # num_ftrs = 512
            # nn.Linear(num_ftrs, 512),  # num_ftrs = 512
            # nn.ReLU(inplace=True),
            # nn.Dropout(),
            # nn.Linear(512, 1024),
            # nn.ReLU(inplace=True),
            # nn.Linear(512, 2)
        )
        # since fc.layers has new layer, required_grad = True here

        self.loss_criterion = nn.CrossEntropyLoss()

        # # ___________________ first approach ___________________
        # self.conv_layers = nn.Sequential(
        #     self.model.conv1(),
        #     self.model.bn1(),
        #     self.model.relu(),
        #     self.model.maxpool(),
        #     self.model.layer1(),
        #     self.model.layer2(),
        #     self.model.layer3(),
        #     self.model.layer4(),
        #     self.model.avgpool(),
        # )

        # num_ftrs = model_ft.fc.in_features
        # self.fc_layers = nn.Sequential(
        #     nn.Linear(num_ftrs, 15)
        # )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward pass with the net, duplicating grayscale channel to
        3 channels.

        Args:
            x: tensor of shape (N,C,H,W) representing input batch of images

        Returns:
            y: tensor of shape (N,num_classes) representing the output
                (raw scores) of the network. Note: we set num_classes=15
        """
        model_output = None

        x = x.repeat(1, 3, 1, 1)  # as ResNet accepts 3-channel color images - use without down sampling layers

        # added this layer to make network work on input of 512 pixels
        # x = self.down_sampling_layers(x)

        x = self.conv_layers(x)

        x = self.adaptive_avg_pooling(x)

        x = x.view(x.size(0), -1)
        model_output = self.fc_layers(x)

        # model_output = self.model(x)

        return model_output


class MyResNet34(nn.Module):
    def __init__(self):
        """
        Initializes network layers.

        Note: Do not forget to freeze the layers of ResNet except the last one.
        Note: Use 'mean' reduction in the loss_criterion. Read Pytorch's
            documention to understand what this means.

        Download pretrained ResNet using pytorch's API. No, use CrossEntropyLoss
            first

        Hint: see the import statements

        find original constructor of ResNet here:
        https://github.com/pytorch/vision/blob/21153802a3086558e9385788956b0f2808b50e51/torchvision/models/resnet.py#L161
        """
        super().__init__()

        self.conv_layers = None
        self.fc_layers = None
        self.loss_criterion = None

        self.adaptive_avg_pooling = nn.AdaptiveAvgPool2d(1)

        # model = resnet18(pretrained=True)
        model = resnet34(pretrained=False)

        for param in model.parameters():
            ''' freeze all model layers '''
            param.requires_grad = False

        self.conv_layers = nn.Sequential(
            *list(model.children())[:-1]
        )

        num_ftrs = model.fc.in_features
        self.fc_layers = nn.Sequential(
            nn.Linear(num_ftrs, 2),
        )

        self.loss_criterion = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward pass with the net, duplicating grayscale channel to
        3 channels.

        Args:
            x: tensor of shape (N,C,H,W) representing input batch of images

        Returns:
            y: tensor of shape (N,num_classes) representing the output
                (raw scores) of the network. Note: we set num_classes=15
        """
        model_output = None

        x = x.repeat(1, 3, 1, 1)  # as ResNet accepts 3-channel color images - use without down sampling layers

        x = self.conv_layers(x)

        x = self.adaptive_avg_pooling(x)

        x = x.view(x.size(0), -1)
        model_output = self.fc_layers(x)

        return model_output
