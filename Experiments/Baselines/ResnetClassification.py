import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
import os
from torch import optim
from torch.utils.data import DataLoader


from ..utils import get_CIFAR10, train_model, evaluate_model, get_SVHN, get_dataloaders
from ..Resnet_Implementation import Resnet18
from ..Setup import Training_Setup

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device", device)
    file_path = "Models/test_Resnet.pth"

    training_setup = Training_Setup(
        lr = 0.05,
        momentum=0.9,
        weight_decay=5e-4,
        gamma=0.1,
        milestones=[25, 50],
    )


    batch_size = 128
    epochs = 75

    data = get_CIFAR10()
    dataloaders = get_dataloaders(data, batch_size, shuffle=True)

    # This dataset is used for OOD test on models trained on CIFAR
    ood_data = get_SVHN()
    ood_dataloaders = get_dataloaders(ood_data, batch_size)
    
    if os.path.exists(file_path):
        model = torch.load(file_path)
        model.to(device)
    else:
        model = Resnet18(inputChannels=3, nClasses=10)
        model.to(device)
        
        
        train_model(
            model, 
            epochs=epochs,
            training_setup=training_setup, 
            dataloaders = dataloaders,
            save_path=file_path,
            )

    
    model.eval()
    with torch.no_grad():
        metrics = evaluate_model(model, dataloaders["test"], ood_dataloaders["test"])
    print(metrics)


    


