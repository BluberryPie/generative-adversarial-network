from dataclasses import dataclass


@dataclass
class Config:
    random_seed: int = 42
    image_dir: str = "data/images"
    result_dir: str = "result"
    batch_size: int = 32
    latent_dim: int = 100
    num_train_iterations: int = 100
    learning_rate: float = 2e-4
    adam_beta_1: float = 0.5
