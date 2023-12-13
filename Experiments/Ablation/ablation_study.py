import numpy as np
import torch
from torch import nn
from ..utils import get_CIFAR10, evaluate_model, get_SVHN, get_dataloaders, train_ensemble_standard, get_CIFAR100
from ..Ensemble import DeepClassificationEnsemble
from ..Resnet_Implementation import Resnet18
from ..Setup import Training_Setup
from ..MCdropoutResnet import MCEnsemble
import matplotlib.pyplot as plt
from ..FGSM import FGSM

#alpha M gamma

def ablation_results():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 256
    data = get_CIFAR100()
    dataloaders = get_dataloaders(data, batch_size, shuffle=True)

    # This dataset is used for OOD test on models trained on CIFAR
    ood_data = get_SVHN(in_dataset_name="CIFAR100")
    ood_dataloaders = get_dataloaders(ood_data, batch_size)

    #alpha ablation
    alphas = [1, 2, 3, 4]
    accuracies = []
    AUPRs = []
    for alpha in alphas:
        file_path = f'Results/Ablation/Resnet18_CIFAR100({alpha}41).pth'
        model = torch.load(file_path)
        model.to(device)

        model.eval()
        with torch.no_grad():
            metrics = evaluate_model(model, dataloaders["test"], ood_dataloaders["test"])
        accuracies.append(metrics["accuracy"])
        AUPRs.append(metrics["AUPR"])

    #accuracy plot
    plt.plot(np.array(alphas), np.array(accuracies), marker='o', linestyle='-')
    plt.xlabel(r'$\alpha$')
    plt.ylabel('Accuracy')
    plt.show()

    #aupr plot
    plt.plot(np.array(alphas), np.array(AUPRs), marker='o', linestyle='-')
    plt.xlabel(r'$\alpha$')
    plt.ylabel('AUPR')
    plt.show()


    #alpha ablation
    gammas = [1, 2, 4]
    accuracies = []
    AUPRs = []
    for gamma in gammas:
        file_path = f'Results/Ablation/Resnet18_CIFAR100(24{gamma}).pth'
        model = torch.load(file_path)
        model.to(device)

        model.eval()
        with torch.no_grad():
            metrics = evaluate_model(model, dataloaders["test"], ood_dataloaders["test"])
        accuracies.append(metrics["accuracy"])
        AUPRs.append(metrics["AUPR"])

    #accuracy plot
    plt.plot(np.array(gammas), np.array(accuracies), marker='o', linestyle='-')
    plt.xlabel(r'$\gamma$')
    plt.ylabel('Accuracy')
    plt.show()

    #aupr plot
    plt.plot(np.array(gammas), np.array(AUPRs), marker='o', linestyle='-')
    plt.xlabel(r'$\gamma$')
    plt.ylabel('AUPR')
    plt.show()

