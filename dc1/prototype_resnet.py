import torch
import torch.nn as nn
import torch.nn.functional as F

class Bottleneck(nn.Module):
    """
    Bottleneck block for ResNet with residual connections.

    Args:
        in_planes (int): Number of input channels.
        planes (int): Number of output channels.
        stride (int, optional): Stride for the convolutional layers. Default is 1.
        is_last (bool, optional): Indicates whether this block is the last in the sequence. Default is False.
    """
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(Bottleneck, self).__init__()
        self.is_last = is_last

        # Three consecutive convolutional layers
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        # Shortcut connection for residual
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out

class BasicBlock(nn.Module):
    """
    Basic block for ResNet with residual connections.

    Args:
        in_planes (int): Number of input channels.
        planes (int): Number of output channels.
        stride (int, optional): Stride for the convolutional layers. Default is 1.
        is_last (bool, optional): Indicates whether this block is the last in the sequence. Default is False.
    """
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(BasicBlock, self).__init__()
        self.is_last = is_last

        # Two consecutive convolutional layers
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # Shortcut connection for residual
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out

class ResNet(nn.Module):
    """
    ResNet model with varying numbers of blocks in each layer.

    Args:
        block (nn.Module): Type of block to be used in the layers (Bottleneck or BasicBlock).
        num_blocks (list): List containing the number of blocks for each layer.
        in_channel (int, optional): Number of input channels. Default is 3.
        zero_init_residual (bool, optional): If True, initialize the residual weights to zero. Default is False.
    """
    def __init__(self, block, num_blocks, in_channel=3, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.in_planes = 64

        # Initial convolutional layer
        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # Four layers with varying number of blocks
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        # Global average pooling layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero initialize the residual weights if specified
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        """
        Create a ResNet layer.

        Args:
            block (nn.Module): Type of block to be used in the layer (Bottleneck or BasicBlock).
            planes (int): Number of output channels.
            num_blocks (int): Number of blocks in the layer.
            stride (int): Stride for the convolutional layers.

        Returns:
            nn.Sequential: ResNet layer.
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i in range(num_blocks):
            stride = strides[i]
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass through the ResNet model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return out

    def forward_with_intermediate(self, x, layer=100):
        """
        Forward pass with intermediate outputs.

        Args:
            x (torch.Tensor): Input tensor.
            layer (int): Layer at which to extract the intermediate output.

        Returns:
            torch.Tensor: Output tensor.
            torch.Tensor: Intermediate output tensor.
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        intermediate_output = self.layer4(out) if layer == 100 else None
        out = self.avgpool(intermediate_output) if layer == 100 else None
        out = torch.flatten(out, 1) if layer == 100 else None
        return out, intermediate_output
