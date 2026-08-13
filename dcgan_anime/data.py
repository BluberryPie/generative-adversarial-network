from pathlib import Path
from typing import Callable

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


anime_transforms = transforms.Compose(
    [
        transforms.Resize(size=(64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ]
)


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


def load_anime(root_dir: Path, batch_size: int) -> DataLoader:
    anime_dataset = AnimeFaceDataset(root_dir=root_dir, transform=anime_transforms)
    anime_loader = DataLoader(
        dataset=anime_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    return anime_loader
