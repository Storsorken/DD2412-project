import torch
from torch import nn
from torchvision import datasets
from torchvision import transforms

import models

base_resnet18_config = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "model": models.resnet18,
    "epochs": 10,
    "batch_size": 64,
    "learning_rate": 1e-3,
    "loss_fn": nn.CrossEntropyLoss(),
    "optimizer": torch.optim.SGD,
    "scheduler": torch.optim.lr_scheduler.StepLR,
    "scheduler_settings": {
        "step_size": 1,
        "gamma": 0.7,
    },
    "dataset": datasets.CIFAR10,
    "num_dataloader_workers": 2,
    "train_transform":
        transforms.Compose([
            transforms.ToTensor(),
        ]),
    "test_transform":
        transforms.Compose([
            transforms.ToTensor(),
        ]),
}