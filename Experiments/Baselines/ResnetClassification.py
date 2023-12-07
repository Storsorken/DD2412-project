import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
import os
from torch import optim
from torch.utils.data import DataLoader


from ..utils import get_CIFAR10, get_CIFAR100, train_model, evaluate_model, get_SVHN, get_dataloaders, count_parameters, get_dataset_stats
from ..Resnet_Implementation import Resnet18, Resnet50
from ..MCdropoutResnet import MCResnet18, MCResnet50
from ..Setup import Training_Setup
from ..FGSM import FGSM

def Resnet_Single(result_path:str, resnet_name:str, dataset_name:str, epsilon:float = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device", device)

    if resnet_name == "Resnet18":
        if dataset_name == "CIFAR10":
            gamma = 0.1
            weight_decay = 5e-4
        elif dataset_name == "CIFAR100":
            gamma = 0.2
            weight_decay = 1e-4

        training_setup = Training_Setup(
            lr = 0.05,
            momentum=0.9,
            weight_decay=weight_decay,
            gamma=gamma,
            milestones=[25, 50],
        )

        batch_size = 128
        epochs = 75

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


    if epsilon is not None:
        mean, std = get_dataset_stats(dataset_name)
        fgsm = FGSM(mean, std, epsilon=epsilon, loss_criterion=nn.CrossEntropyLoss())
    else:
        fgsm = None
    
    
    if os.path.exists(result_path):
        model = torch.load(result_path)
        model.to(device)
    else:
        model = Network(inputChannels=3, n_classes=10)
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
            attacker=fgsm
            )

    
    model.eval()
    with torch.no_grad():
        metrics = evaluate_model(model, dataloaders["test"], ood_dataloaders["test"])
    print(metrics)



def Resnet_Single_MC(result_path:str, resnet_name:str, dataset_name:str, dropout_prob:float):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device", device)

    if resnet_name == "Resnet18":
        if dataset_name == "CIFAR10":
            gamma = 0.1
            weight_decay = 5e-4
        elif dataset_name == "CIFAR100":
            gamma = 0.2
            weight_decay = 1e-4

        training_setup = Training_Setup(
            lr = 0.05,
            momentum=0.9,
            weight_decay=weight_decay,
            gamma=gamma,
            milestones=[25, 50],
        )

        batch_size = 128
        epochs = 75

        Network = MCResnet18
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

        Network = MCResnet50

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
        model = Network(inputChannels=3, n_classes=10, dropout_prob=dropout_prob)
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


    



