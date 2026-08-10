import torch

from config import Config
from data import load_mnist


def main():
    config = Config()
    torch.manual_seed(seed=config.random_seed)

    train_loader, test_loader = load_mnist(
        data_dir=config.data_dir, batch_size=config.batch_size
    )


if __name__ == "__main__":
    main()
