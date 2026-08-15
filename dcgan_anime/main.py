import logging
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import trange

from config import Config
from data import load_anime
from model import Discriminator, Generator


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    config = Config()
    torch.manual_seed(config.random_seed)

    anime_loader = load_anime(
        root_dir=Path(__file__).parent / config.image_dir, batch_size=config.batch_size
    )
    anime_loader_iter = iter(anime_loader)

    device = get_device()
    G = Generator(latent_dim=config.latent_dim).to(device)
    D = Discriminator().to(device)

    bce_loss = nn.BCELoss()
    optim_G = torch.optim.Adam(
        G.parameters(), lr=config.learning_rate, betas=(config.adam_beta_1, 0.999)
    )
    optim_D = torch.optim.Adam(
        D.parameters(), lr=config.learning_rate, betas=(config.adam_beta_1, 0.999)
    )

    for _ in trange(config.num_train_iterations):
        try:
            X = next(anime_loader_iter)
        except StopIteration:
            anime_loader_iter = iter(anime_loader)
            X = next(anime_loader_iter)
        X = X.to(device)
        # Update D
        Z = torch.rand(size=(config.batch_size, config.latent_dim)).to(device)
        loss_D = bce_loss(D(X), torch.ones(config.batch_size, 1).to(device))
        loss_D += bce_loss(D(G(Z).detach()), torch.zeros(config.batch_size, 1).to(device))
        optim_D.zero_grad()
        loss_D.backward()
        optim_D.step()
        # Update G
        Z = torch.rand(size=(config.batch_size, config.latent_dim)).to(device)
        loss_G = bce_loss(D(G(Z)), torch.ones(config.batch_size, 1).to(device))
        optim_G.zero_grad()
        loss_G.backward()
        optim_G.step()


if __name__ == "__main__":
    main()
