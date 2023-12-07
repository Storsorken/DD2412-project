import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
import os
from torch import optim
from torch.utils.data import DataLoader


from ..utils import get_CIFAR10, get_CIFAR100, train_model, evaluate_model, get_SVHN, get_dataloaders, count_parameters
from ..Resnet_Implementation import Resnet18, Resnet50
from ..MCdropoutResnet import MCResnet18, MCResnet50
from ..Setup import Training_Setup

def Resnet_Single(result_path:str, resnet_name:str, dataset_name:str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device", device)

    if resnet_name == "Resnet18":
        training_setup = Training_Setup(
            lr = 0.05,
            momentum=0.9,
            weight_decay=5e-4,
            gamma=0.1,
            milestones=[25, 50],
        )

        batch_size = 128
        epochs = 1

        Network = Resnet18
    elif resnet_name == "Resnet50":
        training_setup = Training_Setup(
        lr = 0.1,
        momentum=0.9,
        weight_decay=5e-4,
        gamma=0.2,
        milestones=[60, 120, 160],
        )

        batch_size = 128
        epochs = 200

        Network = Resnet50

    if dataset_name == "CIFAR10":
        data = get_CIFAR10()
        dataloaders = get_dataloaders(data, batch_size, shuffle=True)

        # This dataset is used for OOD test on models trained on CIFAR
        ood_data = get_SVHN(in_dataset_name="CIFAR10")
        ood_dataloaders = get_dataloaders(ood_data, batch_size)
    elif dataset_name == "CIFAR100":
        data = get_CIFAR100()
        dataloaders = get_dataloaders(data, batch_size, shuffle=True)

        # This dataset is used for OOD test on models trained on CIFAR
        ood_data = get_SVHN(in_dataset_name="CIFAR100")
        ood_dataloaders = get_dataloaders(ood_data, batch_size)
    
    
    if os.path.exists(result_path):
        model = torch.load(result_path)
        model.to(device)
    else:
        model = Network(inputChannels=3, nClasses=10)
        #print(model)
        model.to(device)

        num_params = count_parameters(model)
        print(f"Number of parameters in the model: {num_params}")
        
        
        train_model(
            model, 
            epochs=epochs,
            training_setup=training_setup, 
            dataloaders = dataloaders,
            save_path=result_path,
            )

    
    model.eval()
    with torch.no_grad():
        metrics = evaluate_model(model, dataloaders["test"], ood_dataloaders["test"])
    print(metrics)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device", device)
    file_path = "Models/test_Resnet18.pth"

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
    ood_data = get_SVHN(in_dataset_name="CIFAR10")
    ood_dataloaders = get_dataloaders(ood_data, batch_size)
    
    if os.path.exists(file_path):
        model = torch.load(file_path)
        model.to(device)
    else:
        model = Resnet18(inputChannels=3, nClasses=10)
        print(model)
        model.to(device)

        num_params = count_parameters(model)
        print(f"Number of parameters in the model: {num_params}")
        
        
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



def main2():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device", device)
    file_path = "Models/test_Resnet50.pth"

    training_setup = Training_Setup(
        lr = 0.1,
        momentum=0.9,
        weight_decay=5e-4,
        gamma=0.2,
        milestones=[60, 120, 160],
    )


    batch_size = 128
    epochs = 200

    data = get_CIFAR10()
    dataloaders = get_dataloaders(data, batch_size, shuffle=True)

    # This dataset is used for OOD test on models trained on CIFAR
    ood_data = get_SVHN(in_dataset_name="CIFAR10")
    ood_dataloaders = get_dataloaders(ood_data, batch_size)
    
    if os.path.exists(file_path):
        model = torch.load(file_path)
        model.to(device)
    else:
        model = Resnet50(inputChannels=3, nClasses=10)
        print(model)
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



def MCDrop():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device", device)
    file_path = "Models/test_Resnet18_Dropout.pth"

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
    ood_data = get_SVHN(in_dataset_name="CIFAR10")
    ood_dataloaders = get_dataloaders(ood_data, batch_size)
    
    if os.path.exists(file_path):
        model = torch.load(file_path)
        model.to(device)
    else:
        model = MCResnet18(inputChannels=3, nClasses=10, dropout_prob=0.1)
        print(model)
        model.to(device)

        num_params = count_parameters(model)
        print(f"Number of parameters in the model: {num_params}")
        
        
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


    



