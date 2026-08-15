from pathlib import Path

import imageio
import matplotlib.pyplot as plt
import numpy as np


def make_animation(frames: list[np.ndarray], path: Path, fps: int) -> None:
    # Dtype conversion from float32 to uint8
    converted_frames = [
        np.clip(np.round(frame * 255), 0, 255).astype(np.uint8) for frame in frames
    ]
    imageio.mimsave(path, converted_frames, fps=fps)


def plot_D_probs(D_real_probs: list[float], D_fake_probs: list[float], stride: int = 1) -> None:
    plt.figure(figsize=(10, 5))
    plt.title("Discriminator Real and Fake Probabilities During Training")
    plt.plot(D_real_probs[::stride], label="D(X)")
    plt.plot(D_fake_probs[::stride], label="D(G(Z))")
    plt.xlabel("iterations")
    plt.ylabel("Probability")
    plt.legend()
    plt.show()


def plot_loss_curves(G_losses: list[float], D_losses: list[float], stride: int = 1) -> None:
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses[::stride], label="G")
    plt.plot(D_losses[::stride], label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
