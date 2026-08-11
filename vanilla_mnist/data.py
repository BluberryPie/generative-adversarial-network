from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST


mnist_transforms = transforms.Compose(
    [transforms.ToTensor(), transforms.Lambda(lambda x: torch.flatten(x))]
)


def load_mnist(data_dir: str, batch_size: int) -> tuple[DataLoader, DataLoader]:
    data_path = Path(__file__).parent / data_dir
    train_dataset = MNIST(
        root=data_path, train=True, download=True, transform=mnist_transforms
    )
    test_dataset = MNIST(
        root=data_path, train=False, download=True, transform=mnist_transforms
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    return train_loader, test_loader
