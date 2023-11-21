from torch.utils.data import DataLoader


def get_dataloaders(pytorch_dataset, batch_size, num_workers, train_transform, test_transform):
    """
    Returns the dataloaders for the training, validation, and test sets.
    """
    train_data = pytorch_dataset(
        root="local-dataset",
        train=True,
        download=True,
        transform=train_transform,
    )

    test_data = pytorch_dataset(
        root="local-dataset",
        train=False,
        download=True,
        transform=test_transform,
    )

    train_dataloader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    test_dataloader = DataLoader(
        test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    return train_dataloader, test_dataloader
