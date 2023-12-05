import numpy as np
import torch
from ..utils import get_CIFAR10, evaluate_model, get_SVHN, get_dataloaders, train_ensemble_standard
from ..Ensemble import DeepClassificationEnsemble
from ..Resnet_Implementation import Resnet18
from ..Setup import Training_Setup

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

""" # Your original array with 4 values
original_array = np.array([1, 2, 3, 2])

# Create a 4x4 matrix indicating whether pairs are equal
equality_matrix = (original_array[:, np.newaxis] == original_array[np.newaxis, :]).astype(int)

print(equality_matrix) """

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
    result = matrix/n_total
    result /= 1-accuracies
    return result

def diversity_test():
    file_path = "Models/test_Ensemble_Resnet.pth"
    DE_model = torch.load(file_path)
    DE_model.to(device)

    data = get_CIFAR10()
    dataloaders = get_dataloaders(data, 256, shuffle=False)

    DE_model.eval()
    with torch.no_grad():
        metrics = dissagreement_matrix(DE_model, dataloaders["test"])



    



