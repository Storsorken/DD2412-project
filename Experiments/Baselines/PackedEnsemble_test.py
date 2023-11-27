import torch
import os


from ..utils import get_CIFAR10, evaluate_model, get_SVHN, get_dataloaders, train_packed_ensemble, count_parameters
from ..PackedEnsemble import PackedResnet18, PackedResnet50
from ..Setup import Training_Setup



def PEResnet18_classification():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device", device)
    file_path = "Models/test_PackedEnsemble_Resnet18.pth"

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
    ood_data = get_SVHN()
    ood_dataloaders = get_dataloaders(ood_data, batch_size)
    
    if os.path.exists(file_path):
        PE_model = torch.load(file_path)
        PE_model.to(device)
    else:
        PE_model = PackedResnet18(
            inputChannels = 3,
            nClasses = 10,
            alpha = 2,
            M = 4,
            gamma = 2,
        )
        PE_model.to(device)

        num_params = count_parameters(PE_model)
        print(PE_model)
        print(f"Number of parameters in the model: {num_params}")
        
        
        train_packed_ensemble(
            PE = PE_model, 
            epochs=epochs,
            training_setup=training_setup, 
            dataloaders = dataloaders,
            save_path=file_path,
            )

    
    PE_model.eval()
    with torch.no_grad():
        metrics = evaluate_model(PE_model, dataloaders["test"], ood_dataloaders["test"])
    print(metrics)



def PEResnet50_classification():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device", device)
    file_path = "Models/test_PackedEnsemble_Resnet50.pth"

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
    ood_data = get_SVHN()
    ood_dataloaders = get_dataloaders(ood_data, batch_size)
    
    if os.path.exists(file_path):
        PE_model = torch.load(file_path)
        PE_model.to(device)
    else:
        PE_model = PackedResnet50(
            inputChannels = 3,
            nClasses = 10,
            alpha = 2,
            M = 4,
            gamma = 2,
        )
        PE_model.to(device)

        num_params = count_parameters(PE_model)
        print(PE_model)
        print(f"Number of parameters in the model: {num_params}")
        
        
        train_packed_ensemble(
            PE = PE_model, 
            epochs=epochs,
            training_setup=training_setup, 
            dataloaders = dataloaders,
            save_path=file_path,
            )

    
    PE_model.eval()
    with torch.no_grad():
        metrics = evaluate_model(PE_model, dataloaders["test"], ood_dataloaders["test"])
    print(metrics)


    



