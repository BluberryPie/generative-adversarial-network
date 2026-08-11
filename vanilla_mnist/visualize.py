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
