import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from torchmetrics.classification import BinaryAveragePrecision, BinaryAUROC
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_curve

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_dataset_stats(dataset_name:str):
    if dataset_name == "CIFAR10":
        mean = (0.4914, 0.4822, 0.4465)
        std=(0.2470, 0.2435, 0.2616)
    elif dataset_name == "CIFAR100":
        mean = (0.5071, 0.4865, 0.4409)
        std=(0.2673, 0.2564, 0.2761)
    return mean, std

def init_transform(random_crop:bool=False, hflip_prob:float = 0.5, dataset_name:str = None):
    test_transform_list = []
    train_transform_list = []
    
    if random_crop:
        train_transform_list.append(transforms.RandomCrop(32, padding=4))
    if hflip_prob>0.0:
        train_transform_list.append(transforms.RandomHorizontalFlip(p = hflip_prob))

    train_transform_list.append(transforms.ToTensor())
    train_transform_list.append(transforms.ConvertImageDtype(torch.float32))
    test_transform_list.append(transforms.ToTensor())

    if dataset_name is not None:
        mean, std = get_dataset_stats(dataset_name)
        train_transform_list.append(transforms.Normalize(mean = mean, std = std))
        test_transform_list.append(transforms.Normalize(mean = mean, std = std))

    train_transform = transforms.Compose(train_transform_list)
    test_transform = transforms.Compose(test_transform_list)

    return {
        "train transform":train_transform,
        "test transform": test_transform
    }

def calculate_data_split(dataset_size:int, trainsplit:float):
    training_size = int(dataset_size*trainsplit)
    validation_size = dataset_size - training_size
    return [training_size, validation_size]

def training_validation_idx(dataset_size:int, trainsplit:float):
    indices = list(range(dataset_size))
    split_idx = int(np.floor(trainsplit * dataset_size))

    train_idx, valid_idx = indices[:split_idx], indices[split_idx:]
    assert len(train_idx) != 0 and len(valid_idx) != 0
    return (train_idx, valid_idx)


def get_CIFAR10(trainsplit:float=1.0, normalize:bool = True, random_crop:bool=False, hflip_prob:float = 0.5):
    data = {}
    data_transforms = init_transform(random_crop, hflip_prob, 
                                     dataset_name = "CIFAR10" if normalize else None)
    
    trainData = datasets.CIFAR10(root = "Data/CIFAR10", train=True, transform=data_transforms["train transform"], download=True)

    if trainsplit < 1.0:
        train_idx, valid_idx = training_validation_idx(len(trainData), trainsplit)
        trainData = torch.utils.data.Subset(trainData, train_idx)
        
        valData = datasets.CIFAR10(root = "Data/CIFAR10", train=True, transform=data_transforms["test transform"], download=True)
        valData = torch.utils.data.Subset(valData, valid_idx)
        data["validation"] = valData

    testData = datasets.CIFAR10(root = "Data/CIFAR10", train=False, transform=data_transforms["test transform"], download=True)
    data["train"] = trainData
    data["test"] = testData
    return data

def get_CIFAR100(trainsplit:float=1.0, normalize:bool = True, random_crop:bool=False, hflip_prob:float = 0.5):
    data = {}
    data_transforms = init_transform(random_crop, hflip_prob, 
                                     dataset_name = "CIFAR100" if normalize else None)
    trainData = datasets.CIFAR100(root = "Data/CIFAR100", train=True, transform=data_transforms["train transform"], download=True)

    if trainsplit < 1.0:
        train_idx, valid_idx = training_validation_idx(len(trainData), trainsplit)
        trainData = torch.utils.data.Subset(trainData, train_idx)
        
        valData = datasets.CIFAR100(root = "Data/CIFAR100", train=True, transform=data_transforms["test transform"], download=True)
        valData = torch.utils.data.Subset(valData, valid_idx)
        data["validation"] = valData

    testData = datasets.CIFAR100(root = "Data/CIFAR100", train=False, transform=data_transforms["test transform"], download=True)
    data["train"] = trainData
    data["test"] = testData
    return data

def get_SVHN(in_dataset_name:str, trainsplit:float=1.0, random_crop:bool=False, hflip_prob:float = 0.0):
    data = {}
    data_transforms = init_transform(random_crop, hflip_prob, 
                                     dataset_name = in_dataset_name)
    trainData = datasets.SVHN(root = "Data/SVHN", split="train", transform=data_transforms["train transform"], download=True)

    if trainsplit < 1.0:
        train_idx, valid_idx = training_validation_idx(len(trainData), trainsplit)
        trainData = torch.utils.data.Subset(trainData, train_idx)
        
        valData = datasets.SVHN(root = "Data/SVHN", split="train", transform=data_transforms["test transform"], download=True)
        valData = torch.utils.data.Subset(valData, valid_idx)
        data["validation"] = valData

    testData = datasets.SVHN(root = "Data/SVHN", split="test", transform=data_transforms["test transform"], download=True)
    data["train"] = trainData
    data["test"] = testData
    return data

def get_dataloaders(data, batch_size, shuffle = False):
    dataloaders = {}
    if "train" in data:
        train_dataloader = DataLoader(data["train"], batch_size = batch_size, shuffle = shuffle)
        dataloaders["train"] = train_dataloader
        data.pop("train")

    for key, value in data.items():
        dataloaders[key] = DataLoader(value, batch_size = batch_size, shuffle = False)

    return dataloaders

def accuracy(model, dataloader):
    correct = torch.zeros(1, device=device)
    N = torch.zeros(1, device=device)
    
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        output_probs = model.probabilities(images)
        correct += torch.sum(torch.argmax(output_probs, dim =1) == labels)
        N += len(labels)

    accuracy = correct/N
    return accuracy.item()

def NLL(model, dataloader):
    nll = torch.zeros(1, device=device)
    N = torch.zeros(1, device=device)
    
    for images, labels in dataloader:
        idx = torch.arange(len(labels), device=device)
        images = images.to(device)
        labels = labels.to(device)
        output_probs = model.probabilities(images)
        nll -= torch.sum(torch.log(output_probs[idx, labels]))
        N += len(labels)

    return (nll/N).item()

def ECE(model, dataloader, n_bins = 15):
    corrects = torch.zeros(n_bins)
    confidence_sums = torch.zeros(n_bins)
    N = 0.0
    
    for images, labels in dataloader:
        images = images.to(device)

        # Go back to cpu because histogram does not work with cuda
        output_probs = model.probabilities(images).cpu()

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


def fpr_and_fdr_at_recall(labels, preds, recall_threshold = 0.95, pos_label=1):
    fpr, tpr, _ = roc_curve(labels, preds, pos_label=pos_label)
    return np.interp(recall_threshold, tpr, fpr)


def OODMetrics(model, dataloader, ood_dataloader):
    """
    Reurns: AUPR, AUC, FPR95
    """
    N = len(dataloader.dataset) + len(ood_dataloader.dataset)

    confs = torch.zeros(N, device=device)
    targets = torch.zeros(N,dtype=torch.int64, device=device)

    mAUPR = BinaryAveragePrecision()
    mAUC = BinaryAUROC()
    
    i = 0
    for images, labels in dataloader:
        idx = torch.arange(i, i + len(labels))
        images = images.to(device)
        labels = labels.to(device)
        output_probs = model.probabilities(images)
        conf = -torch.max(output_probs, axis = 1)[0]
        confs[idx] = conf
        targets[idx] = 0
        i += len(labels)

    for images, labels in ood_dataloader:
        idx = torch.arange(i, i + len(labels))
        images = images.to(device)
        labels = labels.to(device)
        output_probs = model.probabilities(images)
        conf = -torch.max(output_probs, axis = 1)[0]
        confs[idx] = conf
        targets[idx] = 1
        i += len(labels)


    return {"AUPR":mAUPR(confs.cpu(), targets.cpu()).item(),
            "AUC":mAUC(confs.cpu(), targets.cpu()).item(),
            "FPR95":fpr_and_fdr_at_recall(np.array(targets.cpu()), np.array(confs.cpu()), pos_label=1)}



def evaluate_model(model, dataloader, ood_dataloader):
    ood_results = OODMetrics(model, dataloader, ood_dataloader)
    result = {
        "accuracy": accuracy(model, dataloader), 
        "NLL":NLL(model, dataloader),
        "ECE":ECE(model, dataloader),
    }  
    result.update(ood_results)
    return result

def train_model(model, epochs, training_setup, dataloaders, save_path = "Models/model.pth", attacker = None):
    model = model.to(device)

    loss_function, optimizer, scheduler = training_setup.create_training_setup(model)
    
    if "validation" in dataloaders:
        perform_val = True
        valDataLoader = dataloaders["validation"]
    else:
        perform_val = False
        valDataLoader = None

    trainDataLoader = dataloaders["train"]

    best_accuracy = 0.0

    for epoch in tqdm(range(epochs)):
        for i, (images, labels) in enumerate(trainDataLoader):
            images = images.to(device)
            labels = labels.to(device)

            if attacker is not None:
                images = attacker.attack(model, images, labels)

            optimizer.zero_grad()
            output = model(images)
            loss = loss_function(output, labels)
            loss.backward()
            optimizer.step()

        if perform_val:
            model.eval()
            with torch.no_grad():
                acc = accuracy(model, valDataLoader)
                if acc > best_accuracy:
                    best_accuracy = acc
                    #print("New best validation accuracy: ", best_accuracy)
                    torch.save(model, save_path)
            model.train()
            
        if scheduler is not None:
            scheduler.step()

    if not perform_val:
        torch.save(model, save_path)
    model = torch.load(save_path)

def train_ensemble_standard(DE, epochs, training_setup, dataloaders, save_path = "Models/model.pth", save_each:bool = False, attacker = None):
    def get_sub_path(model_nr):
        return save_path[:save_path.rfind(".")] + "_" + str(model_nr) + ".pth"

    for i, model in enumerate(DE):
        print("Training sub model nr", i)
        if save_each:
            train_model(model, epochs, training_setup, dataloaders, save_path=get_sub_path(i), attacker = attacker)
        else:
            train_model(model, epochs, training_setup, dataloaders, attacker = attacker)
    torch.save(DE, save_path)


def train_packed_ensemble(PE, epochs, training_setup, dataloaders, save_path = "Models/model.pth", attacker = None):
    PE = PE.to(device)

    loss_function, optimizer, scheduler = training_setup.create_training_setup(PE)
    M = PE.M
    model = PE
    
    if "validation" in dataloaders:
        perform_val = True
        valDataLoader = dataloaders["validation"]
    else:
        perform_val = False
        valDataLoader = None

    trainDataLoader = dataloaders["train"]

    best_accuracy = 0.0

    for epoch in tqdm(range(epochs)):
        for i, (images, labels) in enumerate(trainDataLoader):
            images = images.to(device)
            labels = labels.to(device)
            labels = labels.repeat_interleave(M)

            if attacker is not None:
                images = attacker.PEattack(model, images, labels)
                
            optimizer.zero_grad()            
            output = model(images)
            output = output.view(-1, PE.n_classes)
            loss = loss_function(output, labels)
            loss.backward()
            optimizer.step()

        if perform_val:
            model.eval()
            with torch.no_grad():
                acc = accuracy(model, valDataLoader)
                if acc > best_accuracy:
                    best_accuracy = acc
                    #print("New best validation accuracy: ", best_accuracy)
                    torch.save(model, save_path)
            model.train()
            
        if scheduler is not None:
            scheduler.step()

    if not perform_val:
        torch.save(model, save_path)
    model = torch.load(save_path)



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    

def measure_throughput_on_gpu(model_path, batch_size, img_size):
    """ Example: measure_throughput_on_gpu(model_path, 128, 32) for some model on CIFAR10/100 or similarly sized images"""
    if device != torch.device("cuda"):
        print("No GPU available when calling measure_throughput_on_gpu. Skipping test.")
        return

    model = torch.load(model_path)
    model.to(device)
    throughput_measurements = []
    for _ in range(3):
        # Source for benchmarking approach: https://deci.ai/blog/measure-inference-time-deep-neural-networks/
        dummy_input = torch.randn(batch_size, 3, img_size, img_size, dtype=torch.float).to(device)
        batch_repetitions = 100
        total_time = 0
        with torch.inference_mode():
            for rep in range(batch_repetitions):
                starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                starter.record()
                _ = model(dummy_input)
                ender.record()
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender) / 1000
                total_time += curr_time
        throughput_measurements.append((batch_repetitions * batch_size) / total_time)
    print("Throughput (imgs/s):", np.mean(throughput_measurements))
    print("Standard deviation:", np.std(throughput_measurements))


if __name__ == "__main__":
    output_probs = torch.tensor([[0.75, 0.05, 0.05, 0.05, 0.05],
                    [0.05, 0.75, 0.05, 0.05, 0.05],
                    [0.05, 0.05, 0.75, 0.05, 0.05],
                    [0.05, 0.05, 0.05, 0.75, 0.05]], device=device)
    labels = torch.tensor([0, 1, 3, 2], device=device)
    datasetCifar10 = get_CIFAR10(0.9)
    datasetCifar100 = get_CIFAR100(0.7)
    pass