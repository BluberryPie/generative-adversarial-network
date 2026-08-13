from pathlib import Path
from typing import Callable

from PIL import Image
from torch.utils.data import Dataset


class AnimeFaceDataset(Dataset):
    def __init__(self, root_dir: Path, transform: Callable | None = None):
        self.image_paths = list(root_dir.glob("*.jpg"))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image
