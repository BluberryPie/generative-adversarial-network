import torch
import torch.nn as nn


def build_alternating(layer_dims: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for d_in, d_out in zip(layer_dims, layer_dims[1:]):
        layers.append(nn.Linear(in_features=d_in, out_features=d_out))
        layers.append(nn.ReLU())
    layers = layers[:-1] + [nn.Sigmoid()]
    return nn.Sequential(*layers)


def build_discriminator_layers(layer_dims: list[int], k: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    for d_in, d_out in zip(layer_dims, layer_dims[1:]):
        layers.append(nn.Linear(in_features=d_in, out_features=d_out * k))
        layers.append(MaxOut(k=k))
        layers.append(nn.Dropout())
    layers.append(nn.Linear(in_features=layer_dims[-1], out_features=1))
    layers.append(nn.Sigmoid())
    return nn.Sequential(*layers)


class MaxOut(nn.Module):
    def __init__(self, k: int):
        super().__init__()
        self.k = k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.unflatten(input=x, dim=-1, sizes=(-1, self.k))
        return torch.max(x, dim=-1).values


class Generator(nn.Module):
    def __init__(self, latent_dim: int, hidden_dims: list[int], data_dim: int):
        super().__init__()
        layer_dims = [latent_dim, *hidden_dims, data_dim]
        self.layers = build_alternating(layer_dims)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.layers(z)


class Discriminator(nn.Module):
    def __init__(self, data_dim: int, hidden_dims: list[int], k: int):
        super().__init__()
        layer_dims = [data_dim, *hidden_dims]
        self.layers = build_discriminator_layers(layer_dims, k=k)

    def forward(self, x: torch.Tensor):
        return self.layers(x)
