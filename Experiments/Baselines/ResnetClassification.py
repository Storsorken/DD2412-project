import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
import os
from torch import optim
from torch.utils.data import DataLoader


from ..utils import get_CIFAR10, train_model, evaluate_model, get_SVHN
from ..Resnet_Implementation import Resnet18

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device", device)
    file_path = "Models/finalepoch_Single_Resnet18_9374.pth"

    lr = 0.05
    epochs = 75
    batch_size = 128
    momentum = 0.9
    weight_decay = 5e-4
    gamma = 0.1
    milestones = [25, 50]

    # Prepare CIFAR10 dataloaders
    data = get_CIFAR10()
    train_dataloader = DataLoader(data["train"], batch_size = batch_size, shuffle = True)
    test_dataloader = DataLoader(data["test"], batch_size = 512, shuffle = False)
    dataloaders = {
        "train":train_dataloader, 
        "test":test_dataloader
        }

    if "validation" in data:
        val_dataloader = DataLoader(data["validation"], batch_size = 512, shuffle = False)
        dataloaders["validation"] = val_dataloader
        file_path = "Models/Single_Resnet18.pth"

    # Prepare SVHN dataset
    # This dataset is used for OOD test on models trained on CIFAR
    ood_data = get_SVHN()
    ood_dataloader = DataLoader(ood_data["test"], batch_size = 512, shuffle = False)
    
    if os.path.exists(file_path):
        model = torch.load(file_path)
        model.to(device)
    else:
        model = Resnet18(inputChannels=3, nClasses=10)
        model.to(device)

        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum = momentum, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma)

        
        
        train_model(
            model, 
            epochs=epochs,
            loss_function=loss_function,
            optimizer=optimizer, 
            dataloaders = dataloaders,
            model_name="Single_Resnet18.pth",
            scheduler=scheduler
            )
        
        torch.save(model, "Models/finalepoch_Single_Resnet18.pth")


    
    model.eval()
    with torch.no_grad():
        metrics = evaluate_model(model, test_dataloader, ood_dataloader)
    print(
        "Accuracy:",metrics["accuracy"],
        "NLL:", metrics["NLL"]
    )


    



