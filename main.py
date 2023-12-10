import os
import torch
import numpy as np

from Experiments.Baselines.ResnetClassification import Resnet_Single, Resnet_Single_MC
from Experiments.Baselines.DeepEnsemble_test import Resnet_DE
from Experiments.Baselines.PackedEnsemble_test import Resnet_PE
from Experiments.DiversityTests.Diversity import diversity_test
from Experiments.FGSMTest.fgsm import fgsm_metrics_PE, fgsm_metrics_DE

from Experiments.utils import get_CIFAR10, get_CIFAR100, train_model, evaluate_model, get_SVHN, get_dataloaders, count_parameters

#TODO test
def summarize_metrics(folder_path:str, test_dataloader, ood_dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # List all files in the folder
    files = os.listdir(folder_path)

    final_metrics = {}
    final_metric_stats = {}

    # Iterate through each file
    for file_name in files:
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            model = torch.load(file_path)
            model.to(device)

            model.eval()
            with torch.no_grad():
                metrics = evaluate_model(model, test_dataloader, ood_dataloader)
                for metric, value in metrics.items():
                    if metric not in final_metrics:
                        final_metrics[metric] = []
                    final_metrics[metric].append(value)

    for metric, values in final_metrics.items():
        final_metric_stats[metric] = {"mean": np.array(values).mean(), "std": np.array(values).std()}

    return final_metrics, final_metric_stats

def test_summary():
    data = get_CIFAR10()
    dataloaders = get_dataloaders(data, 256, shuffle=True)

    # This dataset is used for OOD test on models trained on CIFAR
    ood_data = get_SVHN(in_dataset_name="CIFAR10")
    ood_dataloaders = get_dataloaders(ood_data, 256)
    final_metrics, final_metric_stats = summarize_metrics("Results/Resnet18_Single/CIFAR10", dataloaders["test"], ood_dataloaders["test"])
    print(final_metrics)
    print(final_metric_stats)

def get_metrics(model_path, dataset_name):

    if dataset_name == "CIFAR10":
        data = get_CIFAR10()
    elif dataset_name == "CIFAR100":
        data = get_CIFAR100()
        
    dataloaders = get_dataloaders(data, 256, shuffle=True)

    # This dataset is used for OOD test on models trained on CIFAR
    ood_data = get_SVHN(in_dataset_name="CIFAR10")
    ood_dataloaders = get_dataloaders(ood_data, 256)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_path)
    model.to(device)
    model.eval()
    with torch.no_grad():
        metrics = evaluate_model(model, dataloaders["test"], ood_dataloaders["test"])
    print(metrics)


def set_random_seed(seed):
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)


if __name__ == "__main__":
    # remaining cifar100
    set_random_seed(1)
    Resnet_DE("Results/Resnet50_DE/CIFAR100/Resnet50_CIFAR100_DE2.pth", "Resnet50", "CIFAR100")
    set_random_seed(2)
    Resnet_DE("Results/Resnet50_DE/CIFAR100/Resnet50_CIFAR100_DE3.pth", "Resnet50", "CIFAR100")
    set_random_seed(3)
    Resnet_DE("Results/Resnet50_DE/CIFAR100/Resnet50_CIFAR100_DE4.pth", "Resnet50", "CIFAR100")
    set_random_seed(4)
    Resnet_DE("Results/Resnet50_DE/CIFAR100/Resnet50_CIFAR100_DE5.pth", "Resnet50", "CIFAR100")

    # remaining cifar10
    set_random_seed(2)
    Resnet_DE("Results/Resnet50_DE/CIFAR10/Resnet50_CIFAR10_DE3.pth", "Resnet50", "CIFAR10")
    set_random_seed(3)
    Resnet_DE("Results/Resnet50_DE/CIFAR10/Resnet50_CIFAR10_DE4.pth", "Resnet50", "CIFAR10")
    set_random_seed(4)
    Resnet_DE("Results/Resnet50_DE/CIFAR10/Resnet50_CIFAR10_DE5.pth", "Resnet50", "CIFAR10")
