import torch
import os


from ..utils import get_CIFAR10, evaluate_model, get_SVHN, get_dataloaders, train_ensemble_standard
from ..Ensemble import DeepClassificationEnsemble
from ..Resnet_Implementation import Resnet18
from ..Setup import Training_Setup

def test1():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device", device)
    file_path = "Models/test_Ensemble_Resnet.pth"

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
        DE_model = torch.load(file_path)
        DE_model.to(device)
    else:
        DE_model = DeepClassificationEnsemble(
            Model=Resnet18, 
            n_models=4, 
            inputChannels=3, 
            n_classes=10)
        DE_model.to(device)
        
        
        train_ensemble_standard(
            DE = DE_model, 
            epochs=epochs,
            training_setup=training_setup, 
            dataloaders = dataloaders,
            save_path=file_path,
            save_each = True
            )

    
    DE_model.eval()
    with torch.no_grad():
        metrics = evaluate_model(DE_model, dataloaders["test"], ood_dataloaders["test"])
    print(metrics)


    



