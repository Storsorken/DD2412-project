import torch
from tqdm.auto import tqdm
import data


def init_optimizer_and_scheduler(config, model):
    optimizer = config["optimizer"](model.parameters(), lr=config["learning_rate"])
    scheduler = config["scheduler"](optimizer, **config["scheduler_settings"])
    return optimizer, scheduler


def init_model(config):
    return config["model"]().to(config["device"])


def train(dataloader, model, loss_fn, optimizer, scheduler, device):
    "Train the model for one epoch"
    num_data_points = len(dataloader.dataset)
    model.train()
    running_loss = 0.0
    running_correct = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        running_loss += loss.item() * X.size(0)
        running_correct += (pred.argmax(1) == y).sum().item()
    return {
        "train_loss": running_loss / num_data_points,
        "train_acc": running_correct / num_data_points,
    }


def test(dataloader, model, loss_fn, device):
    "Evaluate the model's performance using the dataloader"
    num_data_points = len(dataloader.dataset)
    model.eval()
    running_loss = 0.0
    running_correct = 0
    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            running_loss += loss_fn(pred, y).item() * X.size(0)
            running_correct += (pred.argmax(1) == y).sum().item()
    return {
        "test_loss": running_loss / num_data_points,
        "test_acc": running_correct / num_data_points,
    }


def train_loop(config):
    train_dataloader, test_dataloader = data.get_dataloaders(
        pytorch_dataset=config["dataset"],
        batch_size=config["batch_size"],
        num_workers=config["num_dataloader_workers"],
        train_transform=config["train_transform"],
        test_transform=config["test_transform"],
    )
    model = init_model(config)
    optimizer, scheduler = init_optimizer_and_scheduler(config, model)

    train_metrics = []
    test_metrics = []
    for epoch in tqdm(range(config["epochs"])):
        train_metrics.append(
            train(
                dataloader=train_dataloader,
                model=model,
                loss_fn=config["loss_fn"],
                optimizer=optimizer,
                scheduler=scheduler,
                device=config["device"],
            )
        )
        test_metrics.append(
            test(
                dataloader=test_dataloader,
                model=model,
                loss_fn=config["loss_fn"],
                device=config["device"],
            )
        )

    return model, train_metrics, test_metrics
