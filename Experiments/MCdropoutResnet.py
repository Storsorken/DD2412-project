import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
from typing import Any
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Dropout2dCustom(nn.Module):
    #ugly hack, MC-dropout only
    def __init__(self, p=0.5):
        super(Dropout2dCustom, self).__init__()
        self.p = p
        self.frozen_mask = None
        self.shape = None

    def forward(self, x):
        if not self.training or self.p == 0:
            if self.frozen_mask is not None:
                mask = self.frozen_mask.repeat(x.shape[0], 1)
                mask = mask.int()
                mask = mask.unsqueeze(-1).unsqueeze(-1).expand(*x.shape) / (1 - self.p)
                return x*mask
            else:
                return x

        v = torch.rand((x.shape[0], x.shape[1]), device=device)
        self.shape = x.shape
        mask = v > self.p
        mask = mask.int()
        mask = mask.unsqueeze(-1).unsqueeze(-1).expand(*x.shape)

        # Scale the remaining values to maintain expectation
        return x * mask / (1 - self.p)
    
    def set_frozen_mask(self,):
        # Used to subsample a model using dropout 
        # so that it can be applied over several datpoints
        self.frozen_mask = torch.rand((1, self.shape[1]), device = device)> self.p
        pass

#https://debuggercafe.com/implementing-resnet18-in-pytorch-from-scratch/

class BasicBlock(nn.Module):
    def __init__(
            self, 
            inputChannels:int,
            outputChannels:int,
            stride:int = 1,
            expansion:int = 1,
            dropout_prob:float = 0.1,
        ) -> None:

        super(BasicBlock, self).__init__()

        inputChannels*=expansion
        outputChannels*=expansion

        self.conv1 = nn.Conv2d(
            in_channels=inputChannels, 
            out_channels=outputChannels, 
            kernel_size=3, 
            stride=stride, 
            padding = 1,
            bias = False
            )
        self.bn1 = nn.BatchNorm2d(outputChannels)

        self.dropout = Dropout2dCustom(p = dropout_prob)

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
        out = self.dropout(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        return F.relu(out + self.shortcut(x))
    
    def __str__(self) -> str:
        return "BasicBlock"
    

class BottleNeck(nn.Module):
    def __init__(
            self, 
            inputChannels:int,
            outputChannels:int,
            stride:int = 1,
            expansion:int = 1,
            dropout_prob:float = 0.1,
        ) -> None:

        super(BottleNeck, self).__init__()

        inputChannels*=expansion
        outputChannels*=expansion
        block_channels = outputChannels//4

        self.conv1 = nn.Conv2d(
            in_channels=inputChannels, 
            out_channels=block_channels, 
            kernel_size=1, 
            stride=1, 
            padding = 0,
            bias = False
            )
        self.bn1 = nn.BatchNorm2d(block_channels)
        self.conv2 = nn.Conv2d(
            in_channels=block_channels, 
            out_channels=block_channels, 
            kernel_size=3, 
            stride=stride, 
            padding = 1,
            bias = False
            )
        self.bn2 = nn.BatchNorm2d(block_channels)

        self.dropout = Dropout2dCustom(p = dropout_prob)

        self.conv3 = nn.Conv2d(
            in_channels=block_channels, 
            out_channels=outputChannels, 
            kernel_size=1, 
            stride=1, 
            padding = 0,
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
        out = self.dropout(out)
        out = F.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        return F.relu(out + self.shortcut(x))
    
    def __str__(self) -> str:
        return "BottleNeck"
    

class MCResnet(nn.Module):
    #TODO kaming normal init?
    def __init__(
            self,
            inputChannels:int,
            n_classes:int,
            Block,
            output_sizes,
            layer_sizes,
            dropout_prob:float = 0.0,
        ) -> None:
        super().__init__()

        channels = 64
        self.n_classes = n_classes

        self.conv1 = nn.Conv2d(
            in_channels=inputChannels, 
            out_channels=channels, 
            kernel_size=3, 
            stride=1, 
            padding = 1,
            bias = False
            )
        
        self.bn1 = nn.BatchNorm2d(channels)
        self.layer1 = self.make_layer(Block, channels, output_sizes[0], layer_sizes[0], 1, dropout_prob)
        self.layer2 = self.make_layer(Block, output_sizes[0], output_sizes[1], layer_sizes[1], 2, dropout_prob)
        self.layer3 = self.make_layer(Block, output_sizes[1], output_sizes[2], layer_sizes[2], 2, dropout_prob)
        self.layer4 = self.make_layer(Block, output_sizes[2], output_sizes[3], layer_sizes[3], 2, dropout_prob)

        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.flatten = nn.Flatten(1)

        self.linear = nn.Linear(
            output_sizes[3],
            n_classes
        )

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.pool(out)
        out = self.flatten(out)
        return self.linear(out)
    
    def probabilities(self, x):
        return F.softmax(self.forward(x), dim = -1)
    
    def make_layer(self, Block, in_channels, out_channels, layer_size, stride, dropout_prob):
        od = OrderedDict()
        block = Block(in_channels, out_channels, stride=stride, dropout_prob=dropout_prob)
        od[str(block) + "0"] = block
        for i in range(1,layer_size):
            block = Block(out_channels, out_channels, stride=1, dropout_prob=dropout_prob)
            od[str(block) + str(i)] = block
        return nn.Sequential(od)
    

class MCResnet18(MCResnet):
    def __init__(
            self,
            inputChannels:int,
            n_classes:int,
            dropout_prob:float = 0.0,
        ) -> None:

        super().__init__(
            inputChannels = inputChannels,
            n_classes = n_classes,
            Block = BasicBlock,
            output_sizes = [64, 64*2, 64*4, 64*8],
            layer_sizes = [2, 2, 2, 2],
            dropout_prob=dropout_prob
            )
        

class MCResnet50(MCResnet):
    def __init__(
            self,
            inputChannels:int,
            n_classes:int,
            dropout_prob:float = 0.0,
        ) -> None:

        super().__init__(
            inputChannels = inputChannels,
            n_classes = n_classes,
            Block = BottleNeck,
            output_sizes = [256, 256*2, 256*4, 256*8],
            layer_sizes = [3, 4, 6, 3],
            dropout_prob=dropout_prob
            )
        

def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    copy_model = copy.deepcopy(model)
    for m in copy_model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.set_frozen_mask()
    return copy_model

class MCEnsemble:
    def __init__(
            self,
            model,
            n_models:int,
            device:torch.device = torch.device('cpu')
            ) -> None:
        
        self.models = [enable_dropout(model) for i in range(n_models)]
        self.n_classes = model.n_classes
        self.n_models = n_models
        self.device = device
        self.to(device)

    def forward(self, x):
        out = torch.zeros((self.n_models,x.size(dim=0),self.n_classes), device=self.device)
        idx = torch.arange(x.size(dim=0), device=self.device)
        for i, model in enumerate(self.models):
            out[i,idx,:] = model(x)
        return out
    
    def probabilities(self, x):
        out = self.forward(x)
        out = F.softmax(out, dim=-1)
        return torch.mean(out, dim = 0)
    
    def to(self, device:torch.device):
        self.device = device
        for model in self.models:
            model.to(device)
    
    def eval(self):
        for model in self.models:
            model.eval()

    def get_models(self):
        return self.models

    def train(self):
        for model in self.models:
            model.train()

    def ensemble_size(self):
        return self.n_models
    
    def ensemble_redictions(self, x):
        """
        Returns: the predictions of the different models in the ensemble
        if the number of members is M, and x is of length 1, return M predictions
        """
        preds = torch.zeros((x.shape[0], self.n_models), dtype=int, device=self.device)
        for i, model in enumerate(self.models):
            out = model(x)
            model_preds = torch.argmax(out, dim = -1)
            preds[:, i] = model_preds

        return preds

    def __call__(self, x) -> Any:
        return self.forward(x)
    
    def __getitem__(self, idx:int):
        return self.models[idx]
    
    def __iter__(self):
        self.current = -1
        return self
    
    def __next__(self):
        self.current += 1
        if self.current < len(self.models):
            return self.models[self.current]
        raise StopIteration
    
    def __len__(self):
        return len(self.models)
    
