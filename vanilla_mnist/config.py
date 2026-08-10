from dataclasses import dataclass


@dataclass
class Config:
    random_seed: int = 42
    data_dir: str = "data"
    batch_size: int = 32
