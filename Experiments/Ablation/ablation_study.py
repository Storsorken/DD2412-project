import numpy as np
import torch
from torch import nn
from ..utils import get_CIFAR10, evaluate_model, get_SVHN, get_dataloaders, train_ensemble_standard
from ..Ensemble import DeepClassificationEnsemble
from ..Resnet_Implementation import Resnet18
from ..Setup import Training_Setup
from ..MCdropoutResnet import MCEnsemble
import matplotlib.pyplot as plt
from ..FGSM import FGSM

#alpha M gamma

