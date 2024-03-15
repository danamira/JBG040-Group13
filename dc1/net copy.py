import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, n_classes: int) -> None:
        super(Net, self).__init__()

        self.cnn_layers = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv2d(1, 256, kernel_size=4, stride=1),
            # input 128 x 128 (image size), kernel cannot be greater
            # If lower than max, it can still reduce the image too much for it to fit the next layer, e.g. (3 x 3). Kernel size: (4 x 4)
            # Next, we can have that the input and weights cannot be multiplied
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            torch.nn.Dropout(p=0.5, inplace=True),
            # Defining another 2D convolution layer
            nn.Conv2d(256, 128, kernel_size=4, stride=1), # 3 -> 31, 4 -> max 31, 6 -> 30, 8 -> 30, 12 -> 29
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            torch.nn.Dropout(p=0.25, inplace=True),
            # Defining another 2D convolution layer
            nn.Conv2d(128, 64, kernel_size=4, stride=1),  # 3 -> 31, 4 -> max 31, 6 -> 30, 8 -> 30, 12 -> 29
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            torch.nn.Dropout(p=0.25, inplace=True),
            # Defining another 2D convolution layer
            nn.Conv2d(64, 32, kernel_size=4, stride=1),  # 3 -> 31, 4 -> max 31, 6 -> 30, 8 -> 30, 12 -> 29
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            torch.nn.Dropout(p=0.25, inplace=True),
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(32 * 5 * 5, 256),  # Adjust input size based on the output size of the last convolutional layer
            nn.Linear(256, n_classes)
        )

    # Defining the forward pass
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn_layers(x)
        # After our convolutional layers which are 2D, we need to flatten our
        # input to be 1 dimensional, as the linear layers require this.
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x
