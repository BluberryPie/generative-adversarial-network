import matplotlib.pyplot as plt


def plot_loss_curves(
    D_loss: list[float], G_loss: list[float], title: str, stride: int = 1
) -> None:
    x_vals = range(0, len(D_loss), stride)
    plt.plot(x_vals, D_loss[::stride], label="loss_D", color="tab:blue")
    plt.plot(x_vals, G_loss[::stride], label="loss_G", color="tab:orange")
    plt.legend()
    plt.title(title)
    plt.show()


def plot_D_probs(D_real_probs: list[float], D_fake_probs: list[float], stride: int = 1) -> None:
    x_vals = range(0, len(D_real_probs), stride)
    plt.plot(x_vals, D_real_probs[::stride], label="D(real)", color="tab:blue")
    plt.plot(x_vals, D_fake_probs[::stride], label="D(fake)", color="tab:orange")
    plt.legend()
    plt.title("D(real)/D(fake) mean values")
    plt.show()
