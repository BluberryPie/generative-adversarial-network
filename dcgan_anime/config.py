from dataclasses import dataclass


@dataclass
class Config:
    random_seed: int = 42
    image_dir: str = "data/images"
    batch_size: int = 32
