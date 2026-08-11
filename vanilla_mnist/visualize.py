from pathlib import Path

import imageio
import matplotlib.pyplot as plt
import numpy as np


def plot_loss_curves(
    D_loss: list[float], G_loss: list[float], title: str, stride: int = 1
) -> None:
    x_vals = range(0, len(D_loss), stride)
    plt.plot(x_vals, D_loss[::stride], label="loss_D", color="tab:blue")
    plt.plot(x_vals, G_loss[::stride], label="loss_G", color="tab:orange")
    plt.xlabel("Training iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.title(title)
    plt.show()


def plot_D_probs(
    D_real_probs: list[float],
    D_fake_probs: list[float],
    G_grad_norms: list[float],
    stride: int = 1,
) -> None:
    _, axis_1 = plt.subplots(figsize=(10, 6))
    x_vals = range(0, len(D_real_probs), stride)
    axis_1.plot(x_vals, D_real_probs[::stride], label="D(real)", color="tab:blue")
    axis_1.plot(x_vals, D_fake_probs[::stride], label="D(fake)", color="tab:orange")
    axis_1.set_xlabel("Training iteration")
    axis_1.set_ylabel("D(x) output (mean over batch)")

    axis_2 = axis_1.twinx()
    axis_2.plot(x_vals, G_grad_norms[::stride], label="G grad norms", color="tab:green")
    axis_2.set_ylabel("G grad norms")

    handles_1, labels_1 = axis_1.get_legend_handles_labels()
    handles_2, labels_2 = axis_2.get_legend_handles_labels()
    axis_1.legend(handles_1 + handles_2, labels_1 + labels_2)

    plt.title("D confidence & G gradient norm over training")
    plt.show()


def make_animation(frames: list[np.ndarray], path: Path, fps: int = 5) -> None:
    # Dtype conversion from float32 to uint8
    converted_frames = [
        np.clip(np.round(frame * 255), 0, 255).astype(np.uint8) for frame in frames
    ]
    imageio.mimsave(path, converted_frames, fps=fps)
