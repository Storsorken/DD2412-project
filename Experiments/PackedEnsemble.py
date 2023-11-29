import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
#from Resnet_Implementation import BasicBlock, BottleNeck
import warnings


class PackedBasicBlock(nn.Module):
    def __init__(
            self, 
            inputChannels:int,
            outputChannels:int,
            stride:int = 1,
            alpha:int = 1,
            M:int = 1,
            gamma:int = 1,
            minChannels = 64,
        ) -> None:

        super(PackedBasicBlock, self).__init__()

        inputChannels *= alpha
        outputChannels *= alpha

        self.conv1 = nn.Conv2d(
            in_channels=inputChannels, 
            out_channels=outputChannels, 
            kernel_size=3, 
            stride=stride, 
            padding = 1,
            groups=M,
            bias = False
            )
        self.bn1 = nn.BatchNorm2d(outputChannels)
        self.conv2 = nn.Conv2d(
            in_channels=outputChannels, 
            out_channels=outputChannels, 
            kernel_size=3, 
            stride=1, 
            padding = 1,
            groups=self.get_groupsize(outputChannels, minChannels, gamma, M),
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
                groups=self.get_groupsize(inputChannels, minChannels, gamma, M),
                bias = False, 
                ),
                nn.BatchNorm2d(outputChannels)
            )

    def get_groupsize(self, channels, minChannels, gamma, M):
        final_gamma = gamma
        groups = gamma*M
        while (channels//groups < minChannels or channels%groups != 0) and final_gamma > 1:
            final_gamma -= 1
            groups = final_gamma*M
        
        if final_gamma != gamma:
            warnings.warn("Gamma was changed to meet the requirements for the number of channels")

        return groups


    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        return F.relu(out + self.shortcut(x))
    
    def __str__(self) -> str:
        return "PackedBasicBlock"
    

class PackedBottleNeck(nn.Module):
    def __init__(
            self, 
            inputChannels:int,
            outputChannels:int,
            stride:int = 1,
            alpha:int = 1,
            M:int = 1,
            gamma:int = 1,
            minChannels = 64,
        ) -> None:

        super(PackedBottleNeck, self).__init__()

        inputChannels *= alpha
        outputChannels *= alpha

        block_channels = outputChannels//4

        self.conv1 = nn.Conv2d(
            in_channels=inputChannels, 
            out_channels=block_channels, 
            kernel_size=1, 
            stride=1, 
            padding = 0,
            groups=M,
            bias = False
            )
        self.bn1 = nn.BatchNorm2d(block_channels)
        self.conv2 = nn.Conv2d(
            in_channels=block_channels, 
            out_channels=block_channels, 
            kernel_size=3, 
            stride=stride, 
            padding = 1,
            groups=self.get_groupsize(block_channels, minChannels, gamma, M),
            bias = False
            )
        self.bn2 = nn.BatchNorm2d(block_channels)
        self.conv3 = nn.Conv2d(
            in_channels=block_channels, 
            out_channels=outputChannels, 
            kernel_size=1, 
            stride=1, 
            padding = 0,
            groups=self.get_groupsize(block_channels, minChannels, gamma, M),
            bias = False
            )
        self.bn3 = nn.BatchNorm2d(outputChannels)

        self.shortcut = nn.Sequential()
        if stride != 1 or inputChannels!=outputChannels:
            self.shortcut = nn.Sequential(nn.Conv2d(
                in_channels=inputChannels, 
                out_channels=outputChannels, 
                kernel_size=1, 
                stride=stride, 
                groups=self.get_groupsize(inputChannels, minChannels, gamma, M),
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
        out = F.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        return F.relu(out + self.shortcut(x))
    
    def get_groupsize(self, channels, minChannels, gamma, M):
        final_gamma = gamma
        groups = gamma*M
        while (channels//groups < minChannels or channels%groups != 0) and final_gamma > 1:
            final_gamma -= 1
            groups = final_gamma*M
        
        if final_gamma != gamma:
            warnings.warn("Gamma was changed to meet the requirements for the number of channels")

        return groups
    
    def __str__(self) -> str:
        return "PackedBottleNeck"


class PackedEnsemble(nn.Module):
    #TODO kaming normal init?
    def __init__(
            self,
            inputChannels:int,
            nClasses:int,
            Block,
            output_sizes,
            layer_sizes,
            alpha:int,
            M:int,
            gamma:int,
        ) -> None:
        super().__init__()
        self.M = M
        self.alpha = alpha
        self.gamma = gamma
        self.nClasses = nClasses

        channels = 64

        self.conv1 = nn.Conv2d(
            in_channels=inputChannels, 
            out_channels=channels*alpha, 
            kernel_size=3, 
            stride=1, 
            padding = 1,
            bias = False
            )
        
        self.bn1 = nn.BatchNorm2d(channels*alpha)
        self.layer1 = self.make_layer(Block, channels, output_sizes[0], layer_sizes[0], 1, alpha, M, gamma)
        self.layer2 = self.make_layer(Block, output_sizes[0], output_sizes[1], layer_sizes[1], 2, alpha, M, gamma)
        self.layer3 = self.make_layer(Block, output_sizes[1], output_sizes[2], layer_sizes[2], 2, alpha, M, gamma)
        self.layer4 = self.make_layer(Block, output_sizes[2], output_sizes[3], layer_sizes[3], 2, alpha, M, gamma)

        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.flatten = nn.Flatten(1)

        self.grouped_linear = nn.Conv2d(
            in_channels = output_sizes[3]*alpha,
            out_channels = nClasses*M,
            kernel_size=1,
            groups=M
        )

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.pool(out)
        out = self.grouped_linear(out)
        out = self.flatten(out)
        return out
    
    def probabilities(self, x):
        out = self.forward(x)
        out = torch.reshape(out, (out.shape[0], self.M, self.nClasses))
        p = F.softmax(out, dim = -1)
        p = torch.mean(p, dim = 1)
        return p
    
    def make_layer(self, Block, in_channels, out_channels, layer_size, stride, alpha, M, gamma):
        od = OrderedDict()
        block = Block(in_channels, out_channels, stride,  alpha, M, gamma)
        od[str(block) + "0"] = block
        for i in range(1,layer_size):
            block = Block(out_channels, out_channels, 1,  alpha, M, gamma)
            od[str(block) + str(i)] = block
        return nn.Sequential(od)
    


class PackedResnet18(PackedEnsemble):
    def __init__(
            self,
            inputChannels:int,
            nClasses:int,
            alpha:int,
            M:int,
            gamma:int,
        ) -> None:

        super().__init__(
            inputChannels = inputChannels,
            nClasses = nClasses,
            Block = PackedBasicBlock,
            output_sizes = [64, 64*2, 64*4, 64*8],
            layer_sizes = [2, 2, 2, 2],
            alpha = alpha,
            M = M,
            gamma = gamma,
            )
        

class PackedResnet50(PackedEnsemble):
    def __init__(
            self,
            inputChannels:int,
            nClasses:int,
            alpha:int,
            M:int,
            gamma:int,
        ) -> None:

        super().__init__(
            inputChannels = inputChannels,
            nClasses = nClasses,
            Block = PackedBottleNeck,
            output_sizes = [256, 256*2, 256*4, 256*8],
            layer_sizes = [3, 4, 6, 3],
            alpha = alpha,
            M = M,
            gamma = gamma,
            )
