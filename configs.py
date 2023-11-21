import torch
from torch import nn
from torchvision import datasets
from torchvision import transforms
import math

import models

base_resnet18_config = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "model": models.cifar_resnet18,
    "epochs": 75,
    "batch_size": 128,
    "learning_rate": 0.05,
    "loss_fn": nn.CrossEntropyLoss(),
    "optimizer": torch.optim.SGD,
    "optimizer_settings": {
        "momentum": 0.9,
        "weight_decay": 5e-4,
    },
    "scheduler": torch.optim.lr_scheduler.MultiStepLR,
    "scheduler_settings": {
        "milestones": [25, 50],
        "gamma": 0.1,
    },
    "dataset": datasets.CIFAR10,
    "num_dataloader_workers": 2,
    "train_transform":
        transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.491, 0.482, 0.446), (0.247, 0.243, 0.261)),
        ]),
    "test_transform":
        transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.491, 0.482, 0.446), (0.247, 0.243, 0.261)),
        ]),
}


base_resnet18_config_lightning = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "model": models.cifar_resnet18,
    "epochs": 30,
    "batch_size": 256,
    "learning_rate": 0.05,
    "loss_fn": nn.CrossEntropyLoss(),
    "optimizer": torch.optim.SGD,
    "optimizer_settings": {
        "momentum": 0.9,
        "weight_decay": 5e-4,
    },
    "scheduler": torch.optim.lr_scheduler.OneCycleLR,
    "scheduler_settings": {
        "max_lr": 0.1,
        "epochs": 30,
        "steps_per_epoch": math.ceil(50000 / 256),
    },
    "dataset": datasets.CIFAR10,
    "num_dataloader_workers": 2,
    "train_transform":
        transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.491, 0.482, 0.446), (0.247, 0.243, 0.261)),
        ]),
    "test_transform":
        transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.491, 0.482, 0.446), (0.247, 0.243, 0.261)),
        ]),
}