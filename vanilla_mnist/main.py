import torch

from config import Config
from data import load_mnist
from model import Generator


def main():
    config = Config()
    torch.manual_seed(seed=config.random_seed)

    train_loader, test_loader = load_mnist(
        data_dir=config.data_dir, batch_size=config.batch_size
    )

    generator = Generator(latent_dim=config.latent_dim, hidden_dims=config.hidden_dims, data_dim=784)


if __name__ == "__main__":
    main()
