import numpy as np
import torch
from ..utils import get_CIFAR10, evaluate_model, get_SVHN, get_dataloaders, train_ensemble_standard
from ..Ensemble import DeepClassificationEnsemble
from ..Resnet_Implementation import Resnet18
from ..Setup import Training_Setup
from ..MCdropoutResnet import MCEnsemble
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

""" # Your original array with 4 values
original_array = np.array([1, 2, 3, 2])

# Create a 4x4 matrix indicating whether pairs are equal
equality_matrix = (original_array[:, np.newaxis] == original_array[np.newaxis, :]).astype(int)

print(equality_matrix) """

def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.set_frozen_mask()

def dissagreement_matrix(ensemble, dataloader):
    n_ensembles = ensemble.ensemble_size()
    matrix = torch.zeros((n_ensembles, n_ensembles), device=device)
    n_total = 0.0
    accuracies = torch.zeros((n_ensembles,), device=device)
    for data, labels in dataloader:
        data = data.to(device)
        labels = labels.to(device)

        n_total += len(labels)
        predictions = ensemble.ensemble_redictions(data)
        v = (predictions[:, :, None] != predictions[:, None, :]).int()
        matrix += v.sum(dim=0)
        accuracies += torch.sum(predictions == labels[:,None], dim = 0)
    accuracies = accuracies.cpu()
    accuracies /= n_total

    matrix = matrix.cpu()
    matrix = matrix/n_total
    #result /= 1-accuracies
    return matrix, accuracies


def prepare_difference_pairs(matrix, accuracies):
    M = accuracies.shape[0]
    normalized_diversity = np.zeros(M)
    for i in range(M):
        normalized_diversity[i] = matrix[0, i]
    return normalized_diversity / (1 - accuracies)

def diversity_test():
    file_path = "Models/test_Resnet18_Dropout.pth"
    MC_model = torch.load(file_path)
    MCE_model = MCEnsemble(MC_model, 4, device=device)
    MCE_model.to(device)

    data = get_CIFAR10()
    dataloaders = get_dataloaders(data, 256, shuffle=False)

    MCE_model.eval()
    with torch.no_grad():
        MC_matrix, MC_accuracies = dissagreement_matrix(MCE_model, dataloaders["test"])

    MC_diff_pairs = prepare_difference_pairs(MC_matrix, MC_accuracies)


    file_path = "Models/test_PackedEnsemble_Resnet18.pth"
    PE_model = torch.load(file_path)
    PE_model.to(device)

    data = get_CIFAR10()
    dataloaders = get_dataloaders(data, 256, shuffle=False)

    PE_model.eval()
    with torch.no_grad():
        PE_matrix, PE_accuracues = dissagreement_matrix(PE_model, dataloaders["test"])

    PE_diff_pairs = prepare_difference_pairs(PE_matrix, PE_accuracues)


    file_path = "Models/test_Ensemble_Resnet.pth"
    DE_model = torch.load(file_path)
    DE_model.to(device)

    data = get_CIFAR10()
    dataloaders = get_dataloaders(data, 256, shuffle=False)

    DE_model.eval()
    with torch.no_grad():
        DE_matrix, DE_accuracues = dissagreement_matrix(DE_model, dataloaders["test"])

    DE_diff_pairs = prepare_difference_pairs(DE_matrix, DE_accuracues)

    plt.scatter(MC_accuracies, MC_diff_pairs, c='red', marker = "o", label='Dropout p=0.1')
    plt.scatter(PE_accuracues, PE_diff_pairs, c='green', marker = "s", label='Packed Ensemble')
    plt.scatter(DE_accuracues, DE_diff_pairs, c='blue', marker = "^", label='Deep Ensemble')
    # Add labels and legend
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()

    # Show the plot
    plt.show()


    


    



