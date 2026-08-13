import logging
from pathlib import Path

import torch

from config import Config
from data import load_anime


def main():
    config = Config()
    torch.manual_seed(config.random_seed)

    anime_loader = load_anime(
        root_dir=Path(__file__).parent / config.image_dir, batch_size=config.batch_size
    )


if __name__ == "__main__":
    main()
