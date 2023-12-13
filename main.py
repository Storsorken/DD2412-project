import os
import torch
import numpy as np

from Experiments.Baselines.ResnetClassification import Resnet_Single, Resnet_Single_MC
from Experiments.Baselines.DeepEnsemble_test import Resnet_DE
from Experiments.Baselines.PackedEnsemble_test import Resnet_PE
from Experiments.DiversityTests.Diversity import diversity_test
from Experiments.FGSMTest.fgsm import fgsm_metrics_PE, fgsm_metrics_DE
from Experiments.Ablation.ablation_study import ablation_results

from Experiments.utils import get_CIFAR10, get_CIFAR100, train_model, evaluate_model, get_SVHN, get_dataloaders, count_parameters

#TODO test
def summarize_metrics(folder_path:str, dataset_name:str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if dataset_name == "CIFAR10":
        data = get_CIFAR10()
    elif dataset_name == "CIFAR100":
        data = get_CIFAR100()
        
    dataloaders = get_dataloaders(data, 256, shuffle=True)

    ood_data = get_SVHN(in_dataset_name=dataset_name)
    ood_dataloaders = get_dataloaders(ood_data, 256)

    test_dataloader = dataloaders["test"]
    ood_dataloader = ood_dataloaders["test"]

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


if __name__ == "__main__":
    Resnet_Single_MC("Results/Resnet50_MC/CIFAR100/Resnet50_CIFAR100_MC_0.1.pth", "Resnet50", "CIFAR100", dropout_prob=0.1)
    Resnet_Single_MC("Results/Resnet50_MC/CIFAR100/Resnet50_CIFAR100_MC_0.25.pth", "Resnet50", "CIFAR100", dropout_prob=0.25)
    Resnet_Single_MC("Results/Resnet50_MC/CIFAR100/Resnet50_CIFAR100_MC_0.5.pth", "Resnet50", "CIFAR100", dropout_prob=0.5)

    Resnet_PE("Results/Ablation/Resnet18_CIFAR100(541).pth", "Resnet18", "CIFAR100", alpha=5, M=4, gam=1)
    Resnet_PE("Results/Ablation/Resnet18_CIFAR100(243).pth", "Resnet18", "CIFAR100", alpha=2, M=4, gam=3)

    Resnet_Single("Results/Resnet18_Single/CIFAR10/Resnet18_CIFAR10_1.pth", "Resnet18", "CIFAR10")
    Resnet_Single("Results/Resnet18_Single/CIFAR100/Resnet18_CIFAR100_1.pth", "Resnet18", "CIFAR100")

    Resnet_Single("Results/FGSM/Resnet18_CIFAR10_1.pth", "Resnet18", "CIFAR10", epsilon=0.01)
    Resnet_Single("Results/FGSM/Resnet18_CIFAR100_1.pth", "Resnet18", "CIFAR100", epsilon=0.01)



    """ final_metrics, final_metric_stats = summarize_metrics("Results/Resnet50_PE/CIFAR10", "CIFAR10")
    print(final_metric_stats)
    final_metrics, final_metric_stats = summarize_metrics("Results/Resnet50_PE/CIFAR100", "CIFAR100")
    print(final_metric_stats) """
    """ablation_results()
    diversity_test()
    fgsm_metrics_PE("Results/Resnet18_PE/CIFAR100/Resnet18_CIFAR100_PE1_(242).pth", "CIFAR100")
    fgsm_metrics_DE("Results/Resnet18_DE/CIFAR100/Resnet18_CIFAR100DE1.pth", "CIFAR100")
    fgsm_metrics_PE("Results/FGSM/Resnet18_CIFAR100(242)0.01.pth", "CIFAR100")
    fgsm_metrics_DE("Results/FGSM/Resnet18_CIFAR100DE0.01.pth", "CIFAR100")

    fgsm_metrics_PE("Results/Resnet18_PE/CIFAR10/Resnet18_CIFAR10_PE1_(242).pth", "CIFAR10")
    fgsm_metrics_DE("Results/Resnet18_DE/CIFAR10/Resnet18_CIFAR10DE1.pth", "CIFAR10")
    fgsm_metrics_PE("Results/FGSM/Resnet18_CIFAR10(242)0.01.pth", "CIFAR10")
    fgsm_metrics_DE("Results/FGSM/Resnet18_CIFAR10DE0.01.pth", "CIFAR10") """


    #Resnet_DE("Results/Resnet50_DE/CIFAR100/Resnet50_CIFAR100_DE2.pth", "Resnet50", "CIFAR100")


    #FGSM
    #Resnet_PE("test.pth", "Resnet18", "CIFAR10", alpha=2, M=4, gam=2)
    """ Resnet_DE("Results/FGSM/Resnet18_CIFAR10DE0.01.pth", "Resnet18", "CIFAR10", epsilon=0.01)
    Resnet_PE("Results/FGSM/Resnet18_CIFAR100(242)0.01.pth", "Resnet18", "CIFAR100", alpha=2, M=4, gam=2, epsilon=0.01)
    Resnet_DE("Results/FGSM/Resnet18_CIFAR100DE0.01.pth", "Resnet18", "CIFAR100", epsilon=0.01)

    Resnet_PE("Results/Resnet18_PE/CIFAR10/Resnet18_CIFAR10_PE1_(242).pth", "Resnet18", "CIFAR10", alpha=2, M=4, gam=2)
    Resnet_PE("Results/Resnet18_PE/CIFAR100/Resnet18_CIFAR100_PE1_(242).pth", "Resnet18", "CIFAR100", alpha=2, M=4, gam=2)
    Resnet_DE("Results/Resnet18_DE/CIFAR10/Resnet18_CIFAR10DE1.pth", "Resnet18", "CIFAR10")
    Resnet_DE("Results/Resnet18_DE/CIFAR100/Resnet18_CIFAR100DE1.pth", "Resnet18", "CIFAR100") """
    #Resnet_DE("Results/Ablation/Resnet18_CIFAR100(141).pth", "Resnet18", "CIFAR100", alpha=1, M=4, gam=1)

    #alpha ablation
    """ Resnet_PE("Results/Ablation/Resnet18_CIFAR100(141).pth", "Resnet18", "CIFAR100", alpha=1, M=4, gam=1)
    Resnet_PE("Results/Ablation/Resnet18_CIFAR100(241).pth", "Resnet18", "CIFAR100", alpha=2, M=4, gam=1)
    Resnet_PE("Results/Ablation/Resnet18_CIFAR100(341).pth", "Resnet18", "CIFAR100", alpha=3, M=4, gam=1)
    Resnet_PE("Results/Ablation/Resnet18_CIFAR100(441).pth", "Resnet18", "CIFAR100", alpha=4, M=4, gam=1) """

    """ Resnet_PE("Results/Ablation/Resnet18_CIFAR100(382).pth", "Resnet18", "CIFAR100", alpha=3, M=8, gam=2)
    Resnet_PE("Results/Ablation/Resnet18_CIFAR100(482).pth", "Resnet18", "CIFAR100", alpha=4, M=8, gam=2)
    Resnet_PE("Results/Ablation/Resnet18_CIFAR100(682).pth", "Resnet18", "CIFAR100", alpha=6, M=8, gam=2)
    Resnet_PE("Results/Ablation/Resnet18_CIFAR100(882).pth", "Resnet18", "CIFAR100", alpha=8, M=8, gam=2) """

    #gamma ablation
    """ Resnet_PE("Results/Ablation/Resnet18_CIFAR100(241).pth", "Resnet18", "CIFAR100", alpha=2, M=4, gam=1)
    Resnet_PE("Results/Ablation/Resnet18_CIFAR100(242).pth", "Resnet18", "CIFAR100", alpha=2, M=4, gam=2)
    Resnet_PE("Results/Ablation/Resnet18_CIFAR100(244).pth", "Resnet18", "CIFAR100", alpha=2, M=4, gam=4) """

    """ Resnet_PE("Results/Ablation/Resnet18_CIFAR100(381).pth", "Resnet18", "CIFAR100", alpha=3, M=8, gam=1)
    Resnet_PE("Results/Ablation/Resnet18_CIFAR100(382).pth", "Resnet18", "CIFAR100", alpha=3, M=8, gam=2)
    Resnet_PE("Results/Ablation/Resnet18_CIFAR100(383).pth", "Resnet18", "CIFAR100", alpha=3, M=8, gam=3)
    Resnet_PE("Results/Ablation/Resnet18_CIFAR100(384).pth", "Resnet18", "CIFAR100", alpha=3, M=8, gam=4)
    Resnet_PE("Results/Ablation/Resnet18_CIFAR100(385).pth", "Resnet18", "CIFAR100", alpha=3, M=8, gam=5) """


    """ get_metrics("Results/Resnet50_MC/CIFAR10/Resnet50_CIFAR10_MC_0.1.pth", "CIFAR10")
    get_metrics("Results/Resnet50_MC/CIFAR10/Resnet50_CIFAR10_MC_0.25.pth", "CIFAR10")
    get_metrics("Results/Resnet50_MC/CIFAR10/Resnet50_CIFAR10_MC_0.5.pth", "CIFAR10")
    get_metrics("Results/Resnet50_DE/CIFAR10/Resnet50_CIFAR10_DE1.pth", "CIFAR10")
    get_metrics("Results/Resnet50_PE/CIFAR10/Resnet50_CIFAR10_PE1_(242).pth", "CIFAR10")
    get_metrics("Results/Resnet50_DE/CIFAR100/Resnet50_CIFAR100_DE1.pth", "CIFAR100")
    get_metrics("Results/Resnet50_PE/CIFAR100/Resnet50_CIFAR100_PE1_(242).pth", "CIFAR100") """

    """ Resnet_Single_MC("Results/Resnet50_MC/CIFAR10/Resnet50_CIFAR10_MC_0.1.pth", "Resnet50", "CIFAR10", dropout_prob=0.1)
    Resnet_Single_MC("Results/Resnet50_MC/CIFAR10/Resnet50_CIFAR10_MC_0.25.pth", "Resnet50", "CIFAR10", dropout_prob=0.25)
    Resnet_Single_MC("Results/Resnet50_MC/CIFAR10/Resnet50_CIFAR10_MC_0.5.pth", "Resnet50", "CIFAR10", dropout_prob=0.5)

    Resnet_DE("Results/Resnet50_DE/CIFAR10/Resnet50_CIFAR10_DE1.pth", "Resnet50", "CIFAR10")
    Resnet_PE("Results/Resnet50_PE/CIFAR10/Resnet50_CIFAR10_PE1_(242).pth", "Resnet50", "CIFAR10", alpha=2, M=4, gam=2)

    Resnet_DE("Results/Resnet50_DE/CIFAR100/Resnet50_CIFAR100_DE1.pth", "Resnet50", "CIFAR100")
    Resnet_PE("Results/Resnet50_PE/CIFAR100/Resnet50_CIFAR100_PE1_(242).pth", "Resnet50", "CIFAR100", alpha=2, M=4, gam=2) """
