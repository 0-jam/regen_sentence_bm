import matplotlib.pyplot as plt

# Default: (6.4, 4.8) = 640x480
FIGSIZE = (16, 10)


def plot_result(losses):
    fig, ax = plt.subplots()

    ax.set(xlabel='epoch', ylabel='loss', xticks=range(len(losses)))
    ax.plot(losses)

    fig.set_size_inches(FIGSIZE)
    fig.tight_layout()

    return fig


def save_result(losses, save_to='result.png'):
    plot_result(losses).savefig(save_to)


def show_result(losses):
    plot_result(losses)
    plt.show()
