import os
import torch
import numpy as np

from Experiments.Baselines.ResnetClassification import main, main2, MCDrop, Resnet_Single
from Experiments.Baselines.DeepEnsemble import test1, Resnet_DE
from Experiments.Baselines.PackedEnsemble_test import PEResnet18_classification, PEResnet50_classification
from Experiments.DiversityTests.Diversity import diversity_test

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

                    



if __name__ == "__main__":
    #test_summary()
    """ Resnet_Single("Results/Resnet18_Single/CIFAR10/Resnet18_CIFAR10_Single1.pth", "Resnet18", "CIFAR10")
    Resnet_Single("Results/Resnet18_Single/CIFAR10/Resnet18_CIFAR10_Single2.pth", "Resnet18", "CIFAR10")
    Resnet_Single("Results/Resnet18_Single/CIFAR10/Resnet18_CIFAR10_Single3.pth", "Resnet18", "CIFAR10")
    Resnet_Single("Results/Resnet18_Single/CIFAR10/Resnet18_CIFAR10_Single4.pth", "Resnet18", "CIFAR10")
    Resnet_Single("Results/Resnet18_Single/CIFAR10/Resnet18_CIFAR10_Single5.pth", "Resnet18", "CIFAR10") """
    Resnet_DE("Results/Resnet50_DE/CIFAR10/Resnet50_CIFAR10_Single1.pth", "Resnet50", "CIFAR10")
    #Resnet_DE("Results/Resnet50_DE/CIFAR10/Resnet50_CIFAR10_Single2.pth", "Resnet50", "CIFAR10")
    #MCDrop()
    #diversity_test()
    #main()
    #main2()
    #test1()
    #PEResnet18_classification()
    #PEResnet50_classification()