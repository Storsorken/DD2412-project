from typing import Any
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
#from Resnet_Implementation import Resnet18

class DeepClassificationEnsemble:
    def __init__(
            self,
            Model,
            n_models:int,
            inputChannels:int, 
            n_classes:int,
            device:torch.device = torch.device('cpu')
            ) -> None:
        
        self.models = [Model(inputChannels, n_classes) for i in range(n_models)]
        self.n_classes = n_classes
        self.n_models = n_models
        self.device = device

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