from dataclasses import dataclass, field


@dataclass
class Config:
    random_seed: int = 42
    data_dir: str = "data"
    data_dim: int = 784  # MNIST: 28x28
    batch_size: int = 32
    latent_dim: int = 16
    G_hidden_dims: list[int] = field(default_factory=lambda: [64, 256])
    D_hidden_dims: list[int] = field(default_factory=lambda: [256, 64])
    maxout_k: int = 3
    use_non_saturating_loss: bool = False
    num_train_iterations: int = 10_000
    D_steps: int = 1  # Original paper's `k` in Algorithm 1
    learning_rate: float = 1e-3
    momentum: float = 0.9
