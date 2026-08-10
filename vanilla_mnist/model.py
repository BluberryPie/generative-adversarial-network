import torch
import torch.nn as nn


def build_alternating(layer_dims: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for d_in, d_out in zip(layer_dims, layer_dims[1:]):
        layers.append(nn.Linear(in_features=d_in, out_features=d_out))
        layers.append(nn.ReLU())
    layers = layers[:-1] + [nn.Sigmoid()]
    return nn.Sequential(*layers)


class Generator(nn.Module):
    def __init__(self, latent_dim: int, hidden_dims: list[int], data_dim: int):
        super().__init__()
        layer_dims = [latent_dim, *hidden_dims, data_dim]
        self.layers = build_alternating(layer_dims)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.layers(z)
