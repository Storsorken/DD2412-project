import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

#https://debuggercafe.com/implementing-resnet18-in-pytorch-from-scratch/

class BasicBlock(nn.Module):
    def __init__(
            self, 
            inputChannels:int,
            outputChannels:int,
            stride:int = 1,
        ) -> None:

        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=inputChannels, 
            out_channels=outputChannels, 
            kernel_size=3, 
            stride=stride, 
            padding = 1,
            bias = False
            )
        self.bn1 = nn.BatchNorm2d(outputChannels)
        self.conv2 = nn.Conv2d(
            in_channels=outputChannels, 
            out_channels=outputChannels, 
            kernel_size=3, 
            stride=1, 
            padding = 1,
            bias = False
            )
        self.bn2 = nn.BatchNorm2d(outputChannels)

        self.shortcut = nn.Sequential()
        if stride != 1:
            self.shortcut = nn.Sequential(nn.Conv2d(
                in_channels=inputChannels, 
                out_channels=outputChannels, 
                kernel_size=1, 
                stride=stride, 
                bias=False
                ),
                nn.BatchNorm2d(outputChannels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        return F.relu(out + self.shortcut(x))
    

    
class Resnet18(nn.Module):
    def __init__(
            self,
            inputChannels:int,
            nClasses:int
        ) -> None:
        super().__init__()

        channels = 64

        self.conv1 = nn.Conv2d(
            in_channels=inputChannels, 
            out_channels=channels, 
            kernel_size=3, 
            stride=1, 
            padding = 1,
            bias = False
            )
        self.bn1 = nn.BatchNorm2d(channels)
        self.layer1 = self.make_layer(channels, channels, 1)
        self.layer2 = self.make_layer(channels, channels*2, 2)
        self.layer3 = self.make_layer(channels*2, channels*4, 2)
        self.layer4 = self.make_layer(channels*4, channels*8, 2)

        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.flatten = nn.Flatten(1)

        self.linear = nn.Linear(
            channels*8,
            nClasses
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.pool(out)
        out = self.flatten(out)
        return self.linear(out)

    def make_layer(self, in_channels, out_channels, stride):
        return nn.Sequential(
            BasicBlock(in_channels, out_channels, stride=stride),
            BasicBlock(out_channels, out_channels, stride=1)
        )
    
