import numpy as np
import torch
from torch import nn
from ..utils import get_CIFAR10, get_CIFAR100, train_model, evaluate_model, get_SVHN
from ..utils import get_dataloaders, count_parameters, train_ensemble_standard, train_packed_ensemble
from ..Ensemble import DeepClassificationEnsemble
from ..Resnet_Implementation import Resnet18
from ..Setup import Training_Setup
from ..MCdropoutResnet import MCEnsemble
import matplotlib.pyplot as plt
from ..FGSM import FGSM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def accuracy(outputs, dataloader):
    correct = torch.zeros(1, device=device)
    N = torch.zeros(1, device=device)
    
    for i, (batch) in enumerate(dataloader):
        _, labels = batch
        output_probs = outputs[i]
        labels = labels.to(device)
        correct += torch.sum(torch.argmax(output_probs, dim =1) == labels)
        N += len(labels)

    accuracy = correct/N
    return accuracy.item()

def NLL(outputs, dataloader):
    nll = torch.zeros(1, device=device)
    N = torch.zeros(1, device=device)
    
    for i, (batch) in enumerate(dataloader):
        _, labels = batch
        output_probs = outputs[i]
        idx = torch.arange(len(labels), device=device)
        labels = labels.to(device)
        nll -= torch.sum(torch.log(output_probs[idx, labels]))
        N += len(labels)

    return (nll/N).item()

def ECE(outputs, dataloader, n_bins = 15):
    corrects = torch.zeros(n_bins)
    confidence_sums = torch.zeros(n_bins)
    N = 0.0
    
    for i, (batch) in enumerate(dataloader):
        _, labels = batch
        
        # Go back to cpu because histogram does not work with cuda
        output_probs = outputs[i].cpu()

        confidences, _ = torch.max(output_probs, dim = 1)

        corrects += torch.histogram(
            confidences, 
            bins = n_bins, 
            range=(0.0, 1.0),
            weight = torch.where(torch.argmax(output_probs, dim =1) == labels, 1.0, 0.0)
            )[0]
        confidence_sums += torch.histogram(
            confidences, 
            bins = n_bins, 
            range=(0.0, 1.0),
            weight = confidences
            )[0]
        
        N += len(labels)
        
    ece = torch.sum(torch.abs(corrects - confidence_sums))/N
    return ece.item()



def evaluate_model(outputs, dataloader):
    result = {
        "accuracy": accuracy(outputs, dataloader), 
        "NLL":NLL(outputs, dataloader),
        "ECE":ECE(outputs, dataloader),
    }  
    return result


def fgsm_metrics_DE():
    file_path = "Models/test_Ensemble_Resnet.pth"
    DE_model = torch.load(file_path)
    DE_model.to(device)

    data = get_CIFAR10()
    dataloaders = get_dataloaders(data, 256, shuffle=False)

    DE_model.eval()

    mean = torch.tensor([.4914, 0.4822, 0.4465], device=device)
    std = torch.tensor([0.2470, 0.2435, 0.2616], device=device)

    attacker = FGSM(mean, std, epsilon=0.01, loss_criterion=nn.CrossEntropyLoss())

    
    output_probs = []

    for images, labels in dataloaders["test"]:
        images = images.to(device)
        labels = labels.to(device)
        probs = torch.zeros((images.shape[0], 10), device=device)
        for model in DE_model:
            perturbed_images = attacker.attack(model, images, labels)
            with torch.no_grad():
                probs += model.probabilities(perturbed_images)
        with torch.no_grad():
            output_probs.append(probs/len(DE_model))

    with torch.no_grad():
        print(evaluate_model(output_probs, dataloaders["test"]))

def fgsm_metrics_PE():
    file_path = "Models/test_PackedEnsemble_Resnet18.pth"
    model = torch.load(file_path)
    model.to(device)

    data = get_CIFAR10()
    dataloaders = get_dataloaders(data, 256, shuffle=False)

    model.eval()

    mean = torch.tensor([.4914, 0.4822, 0.4465], device=device)
    std = torch.tensor([0.2470, 0.2435, 0.2616], device=device)

    attacker = FGSM(mean, std, epsilon=0.0, loss_criterion=nn.CrossEntropyLoss())
    
    output_probs = []

    for images, labels in dataloaders["test"]:
        images = images.to(device)
        labels = labels.to(device)
        labels = labels.repeat_interleave(model.M)
        probs = torch.zeros((images.shape[0], 10), device=device)
        perturbed_images = attacker.PEattack(model, images, labels)
        with torch.no_grad():
            probs = model.probabilities(perturbed_images)
            output_probs.append(probs)

    with torch.no_grad():
        print(evaluate_model(output_probs, dataloaders["test"]))



