import torch
import torch.nn as nn


def build_g_conv_layers(layer_dims: list[int]) -> list[nn.Module]:
    layers: list[nn.Module] = []
    for d_in, d_out in zip(layer_dims, layer_dims[1:]):
        layers.append(
            nn.ConvTranspose2d(
                in_channels=d_in,
                out_channels=d_out,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            )
        )
        layers.append(nn.BatchNorm2d(num_features=d_out))
        layers.append(nn.ReLU())
    return layers


def build_d_conv_layers(layer_dims: list[int]) -> list[nn.Module]:
    layers: list[nn.Module] = []
    for d_in, d_out in zip(layer_dims, layer_dims[1:]):
        layers.append(
            nn.Conv2d(
                in_channels=d_in,
                out_channels=d_out,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            )
        )
        layers.append(nn.BatchNorm2d(num_features=d_out))
        layers.append(nn.LeakyReLU(0.2))
    return layers


class Generator(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=latent_dim,
                out_channels=1024,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=1024),
            nn.ReLU(),
            *build_g_conv_layers(layer_dims=[1024, 512, 256, 128]),
            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=3,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=True,
            ),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = z.unsqueeze(-1).unsqueeze(-1)
        return self.layers(z)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=128,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=True,
            ),
            nn.LeakyReLU(0.2),
            *build_d_conv_layers(layer_dims=[128, 256, 512, 1024]),
            nn.Conv2d(
                in_channels=1024,
                out_channels=1,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=True,
            ),
            nn.Sigmoid(),
        )

    def forward(self, X: torch.Tensor):
        p = self.layers(X).squeeze(-1).squeeze(-1)
        return p
