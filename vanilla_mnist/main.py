import torch
import torch.nn as nn
from tqdm import trange

from config import Config
from data import load_mnist
from model import Generator, Discriminator
from visualize import plot_loss_curves


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

    for _ in trange(config.num_train_iterations):
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
        Z = torch.randn(size=(config.batch_size, config.latent_dim))
        loss_G = bce_loss_fn(D(G(Z)), target_G) * sign_G
        optim_G.zero_grad()
        loss_G.backward()
        optim_G.step()

        D_losses.append(loss_D.item())
        G_losses.append(loss_G.item())

    plot_loss_curves(
        D_losses, G_losses, title=f"{config.use_non_saturating_loss = }", stride=10
    )


if __name__ == "__main__":
    main()
