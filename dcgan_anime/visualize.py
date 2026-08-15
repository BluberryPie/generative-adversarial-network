from pathlib import Path

import imageio
import numpy as np


def make_animation(frames: list[np.ndarray], path: Path, fps: int) -> None:
    # Dtype conversion from float32 to uint8
    converted_frames = [
        np.clip(np.round(frame * 255), 0, 255).astype(np.uint8) for frame in frames
    ]
    imageio.mimsave(path, converted_frames, fps=fps)
