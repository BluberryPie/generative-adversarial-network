from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision
from tqdm import trange

from config import Config
from data import load_mnist
from model import Generator, Discriminator
from visualize import plot_loss_curves, plot_D_probs, make_animation


@torch.no_grad()
def sample_grid(G: Generator, Z: torch.Tensor, nrows: int) -> np.ndarray:
    generated_images = G(Z).reshape(Z.shape[0], 1, 28, 28)  # Assumes MNIST shape
    grid = torchvision.utils.make_grid(generated_images, nrow=nrows)
    grid = grid.permute(1, 2, 0).numpy()
    return grid


def main():
    config = Config()
    torch.manual_seed(seed=config.random_seed)

    train_loader, test_loader = load_mnist(
        data_dir=config.data_dir, batch_size=config.batch_size
    )

    G = Generator(
        latent_dim=config.latent_dim,
        hidden_dims=config.G_hidden_dims,
        data_dim=config.data_dim,
    )
    D = Discriminator(
        data_dim=config.data_dim,
        hidden_dims=config.D_hidden_dims,
        k=config.maxout_k,
    )

    optim_G = torch.optim.SGD(
        params=G.parameters(), lr=config.learning_rate, momentum=config.momentum
    )
    optim_D = torch.optim.SGD(
        params=D.parameters(), lr=config.learning_rate, momentum=config.momentum
    )

    bce_loss_fn = nn.BCELoss(reduction="mean")
    train_data_iter = iter(train_loader)
    if config.use_non_saturating_loss:
        target_G = torch.ones(size=(config.batch_size, 1))
        sign_G = 1
    else:
        target_G = torch.zeros(size=(config.batch_size, 1))
        sign_G = -1

    D_losses: list[float] = []
    G_losses: list[float] = []
    D_real_probs: list[float] = []
    D_fake_probs: list[float] = []
    G_grad_norms: list[float] = []
    frames: list[np.ndarray] = []
    Z_fixed = torch.randn(size=(config.nrows_per_grid**2, config.latent_dim))

    for i in trange(config.num_train_iterations):
        # Update D
        for _ in range(config.D_steps):
            try:
                X, _ = next(train_data_iter)
            except StopIteration:
                train_data_iter = iter(train_loader)
                X, _ = next(train_data_iter)
            Z = torch.randn(size=(config.batch_size, config.latent_dim))
            loss_D = bce_loss_fn(D(X), torch.ones(size=(config.batch_size, 1)))
            loss_D += bce_loss_fn(
                D(G(Z).detach()), torch.zeros(size=(config.batch_size, 1))
            )
            optim_D.zero_grad()
            loss_D.backward()
            optim_D.step()

        with torch.no_grad():
            D_real_probs.append(D(X).mean().item())
            D_fake_probs.append(D(G(Z)).mean().item())

        # Update G
        Z = torch.randn(size=(config.batch_size, config.latent_dim))
        loss_G = bce_loss_fn(D(G(Z)), target_G) * sign_G
        optim_G.zero_grad()
        loss_G.backward()
        G_grad_norms.append(
            torch.nn.utils.get_total_norm(
                [p.grad for p in G.parameters() if p.grad is not None]
            ).item()
        )
        optim_G.step()

        D_losses.append(loss_D.item())
        G_losses.append(loss_G.item())
        if (i + 1) % (config.num_train_iterations // config.num_anim_frames) == 0:
            frames.append(sample_grid(G, Z_fixed, nrows=config.nrows_per_grid))

    plot_loss_curves(
        D_losses,
        G_losses,
        title=f"D/G loss curves ({'non-saturating' if config.use_non_saturating_loss else 'minimax'} G loss)",
        stride=10,
    )
    plot_D_probs(D_real_probs, D_fake_probs, G_grad_norms)
    make_animation(frames, Path(__file__).parent / config.result_dir / "animation.gif")


if __name__ == "__main__":
    main()
