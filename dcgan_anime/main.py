import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision
from tqdm import trange

from config import Config
from data import load_anime
from model import Discriminator, Generator
from visualize import make_animation, plot_D_probs, plot_loss_curves


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def sample_grid(G: Generator, Z: torch.Tensor, nrows: int) -> np.ndarray:
    G.eval()
    generated_images = G(Z)
    grid = torchvision.utils.make_grid(
        generated_images, nrow=nrows, normalize=True, value_range=(-1, 1)
    )
    grid = grid.permute(1, 2, 0).cpu().numpy()
    G.train()
    return grid


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

    G_losses: list[float] = []
    D_losses: list[float] = []
    D_real_probs: list[float] = []
    D_fake_probs: list[float] = []
    Z_fixed = torch.rand(size=(config.nrows_per_grid**2, config.latent_dim)).to(device)
    frames: list[np.ndarray] = []

    for i in trange(config.num_train_iterations):
        try:
            X = next(anime_loader_iter)
        except StopIteration:
            anime_loader_iter = iter(anime_loader)
            X = next(anime_loader_iter)
        X = X.to(device)
        # Update D
        Z = torch.rand(size=(config.batch_size, config.latent_dim)).to(device)
        D_real_prob = D(X)
        D_fake_prob = D(G(Z).detach())
        loss_D = bce_loss(D_real_prob, torch.full((config.batch_size, 1), 0.9).to(device))
        loss_D += bce_loss(
            D_fake_prob, torch.zeros(config.batch_size, 1).to(device)
        )
        D_losses.append(loss_D.item())
        D_real_probs.append(D_real_prob.mean().item())
        D_fake_probs.append(D_fake_prob.mean().item())
        optim_D.zero_grad()
        loss_D.backward()
        optim_D.step()
        # Update G
        Z = torch.rand(size=(config.batch_size, config.latent_dim)).to(device)
        loss_G = bce_loss(D(G(Z)), torch.ones(config.batch_size, 1).to(device))
        G_losses.append(loss_G.item())
        optim_G.zero_grad()
        loss_G.backward()
        optim_G.step()

        if (i + 1) % (config.num_train_iterations // config.num_anim_frames) == 0:
            frames.append(sample_grid(G, Z_fixed, nrows=config.nrows_per_grid))

    plot_loss_curves(G_losses, D_losses)
    plot_D_probs(D_real_probs, D_fake_probs)
    make_animation(
        frames, Path(__file__).parent / config.result_dir / "animation.gif", fps=20
    )


if __name__ == "__main__":
    main()
