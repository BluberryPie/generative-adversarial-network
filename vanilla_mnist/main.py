import torch

from config import Config
from data import load_mnist
from model import Generator, Discriminator


def main():
    config = Config()
    torch.manual_seed(seed=config.random_seed)

    train_loader, test_loader = load_mnist(
        data_dir=config.data_dir, batch_size=config.batch_size
    )

    generator = Generator(
        latent_dim=config.latent_dim,
        hidden_dims=config.G_hidden_dims,
        data_dim=config.data_dim,
    )
    discriminator = Discriminator(
        data_dim=config.data_dim, hidden_dims=config.D_hidden_dims, k=config.maxout_k
    )


if __name__ == "__main__":
    main()
