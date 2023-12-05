import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from torchmetrics.classification import BinaryAveragePrecision, BinaryAUROC
from torch.utils.data import DataLoader
from tqdm import tqdm



transform = transforms.Compose([
    transforms.ToTensor(),
])

# Load CIFAR-100 dataset
cifar_dataset = datasets.CIFAR100(root = "Data/CIFAR100", train=True, transform=transform, download=True)

dataset = DataLoader(dataset=cifar_dataset, batch_size=1, shuffle=False)


mean = np.zeros(3)
std = np.zeros(3)
std2 = np.zeros(3)
count = 0
for data, _ in dataset:
    count += len(data)
    for i in range(3):
        mean[i] += data[:, i, :, :].sum()
        std2[i] += data[:, i, :, :].std()


mean /= count*1024
std2 /= len(dataset)

for data, _ in dataset:
    for i in range(3):
        std[i] += torch.square(data[:, i, :, :] - mean[i]).sum()

std /= (count - 1)*1024
std = np.sqrt(std)
print(mean)
print(std)
print(std2)


transform = transforms.Compose([
    transforms.ToTensor(),
])

# Load CIFAR-10 dataset
cifar_dataset = datasets.CIFAR10(root = "Data/CIFAR10", train=True, transform=transform, download=True)

dataset = DataLoader(dataset=cifar_dataset, batch_size=1, shuffle=False)


mean = np.zeros(3)
std = np.zeros(3)
std2 = np.zeros(3)
count = 0
for data, _ in dataset:
    count += len(data)
    for i in range(3):
        mean[i] += data[:, i, :, :].sum()
        std2[i] += data[:, i, :, :].std()


mean /= count*1024
std2 /= 50000

for data, _ in dataset:
    for i in range(3):
        std[i] += torch.square(data[:, i, :, :] - mean[i]).sum()

std /= (count - 1)*1024
std = np.sqrt(std)
print(mean)
print(std)
print(std2)