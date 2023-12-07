import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from torchmetrics.classification import BinaryAveragePrecision, BinaryAUROC
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

class FGSM:
    def __init__(self, mean, std, epsilon, loss_criterion) -> None:
        self.mean = mean.view(1,-1,1,1)
        self.std = std.view(1,-1,1,1)
        self.eps = epsilon
        self.criterion = loss_criterion

    def attack(self, model, data, labels):
        model.eval()
        clone_data = data.clone()
        clone_labels = labels.clone()

        clone_data.requires_grad = True

        output = model(clone_data)
        loss = self.criterion(output, clone_labels)

        # Zero all existing gradients
        model.zero_grad()

        # calculate gradient
        loss.backward()

        data_grad = clone_data.grad.data

        data_denorm = self.denorm(clone_data)

        # fgsm on clean data
        sign_data_grad = data_grad.sign()
        perturbed_image = data_denorm + self.eps*sign_data_grad
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        perturbed_image = perturbed_image.detach()
        perturbed_image.requires_grad = False

        #image_array = perturbed_image[0].permute(1, 2, 0).cpu().numpy()
        # Plot the image
        #plt.imshow(image_array)
        #plt.show()

        # re-normalize data
        perturbed_data_normalized = self.norm(perturbed_image) 

        model.train()
        #perturbed_data_normalized = perturbed_data_normalized.detach()

        return perturbed_data_normalized
    
    def PEattack(self, model, data, labels):
        model.eval()
        clone_data = data.clone()
        clone_labels = labels.clone()

        clone_data.requires_grad = True

        output = model(clone_data)
        output = output.view(-1, model.n_classes)
        loss = self.criterion(output, clone_labels)

        # Zero all existing gradients
        model.zero_grad()

        # calculate gradient
        loss.backward()

        data_grad = clone_data.grad.data

        data_denorm = self.denorm(clone_data)

        # fgsm on clean data
        sign_data_grad = data_grad.sign()
        perturbed_image = data_denorm + self.eps*sign_data_grad
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        perturbed_image = perturbed_image.detach()
        perturbed_image.requires_grad = False

        #image_array = perturbed_image[0].permute(1, 2, 0).cpu().numpy()
        # Plot the image
        #plt.imshow(image_array)
        #plt.show()

        # re-normalize data
        perturbed_data_normalized = self.norm(perturbed_image) 

        model.train()
        #perturbed_data_normalized = perturbed_data_normalized.detach()

        return perturbed_data_normalized


    def denorm(self, norm_data):
        return norm_data * self.std + self.mean
    
    def norm(self, unormalized_data):
        return (unormalized_data - self.mean) / self.std