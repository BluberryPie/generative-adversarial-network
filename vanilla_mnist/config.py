from dataclasses import dataclass, field


@dataclass
class Config:
    random_seed: int = 42
    data_dir: str = "data"
    batch_size: int = 32
    latent_dim: int = 16
    hidden_dims: list[int] = field(default_factory=lambda: [64, 256])
