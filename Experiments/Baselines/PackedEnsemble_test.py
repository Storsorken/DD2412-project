import torch
import os
from torch import nn


from ..utils import get_CIFAR10, get_CIFAR100, evaluate_model, get_SVHN, get_dataloaders, train_packed_ensemble, count_parameters, get_dataset_stats
from ..PackedEnsemble import PackedResnet18, PackedResnet50
from ..Setup import Training_Setup
from ..FGSM import FGSM

def Resnet_PE(result_path:str, resnet_name:str, dataset_name:str, alpha:int, M:int, gam:int, epsilon:float = None):
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

        Network = PackedResnet18
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

        Network = PackedResnet50

    if dataset_name == "CIFAR10":
        data = get_CIFAR10()
        dataloaders = get_dataloaders(data, batch_size, shuffle=True)

        # This dataset is used for OOD test on models trained on CIFAR
        ood_data = get_SVHN(in_dataset_name="CIFAR10")
        ood_dataloaders = get_dataloaders(ood_data, batch_size)
        n_classes = 10
    elif dataset_name == "CIFAR100":
        data = get_CIFAR100()
        dataloaders = get_dataloaders(data, batch_size, shuffle=True)

        # This dataset is used for OOD test on models trained on CIFAR
        ood_data = get_SVHN(in_dataset_name="CIFAR100")
        ood_dataloaders = get_dataloaders(ood_data, batch_size)
        n_classes = 100


    if epsilon is not None:
        mean, std = get_dataset_stats(dataset_name)
        mean = torch.tensor(mean, device=device)
        std = torch.tensor(std, device=device)
        fgsm = FGSM(mean, std, epsilon=epsilon, loss_criterion=nn.CrossEntropyLoss())
    else:
        fgsm = None
    
    
    if os.path.exists(result_path):
        PE_model = torch.load(result_path)
        PE_model.to(device)
    else:
        PE_model = Network(
            inputChannels = 3,
            n_classes = n_classes,
            alpha = alpha,
            M = M,
            gamma = gam,
        )
        PE_model.to(device)

        num_params = count_parameters(PE_model)
        #print(PE_model)
        print(f"Number of parameters in the model: {num_params}")
        
        
        train_packed_ensemble(
            PE = PE_model, 
            epochs=epochs,
            training_setup=training_setup, 
            dataloaders = dataloaders,
            save_path=result_path,
            attacker=fgsm
            )

    
    PE_model.eval()
    with torch.no_grad():
        metrics = evaluate_model(PE_model, dataloaders["test"], ood_dataloaders["test"])
    print(metrics)

    



