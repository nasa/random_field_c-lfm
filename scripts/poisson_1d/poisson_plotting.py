import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch


def compare_samples(
    x_grid,
    u_true,
    h_true,
    u_gen,
    h_gen,
    x_sensor=None,
    samples_to_plot=250,
    plot_name=None,
    show_plot=True,
):

    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle("VAE Results")
    ax[0].set_title("Real: u(x)")
    ax[0].set_xlabel("x")
    ax[1].set_title("Generated: u(x)")
    ax[1].set_xlabel("x")
    ax[2].set_title("Real: h(x)")
    ax[2].set_xlabel("x")
    ax[3].set_title("Generated: h(x)")
    ax[3].set_xlabel("x")

    alpha = 0.05

    ax[0].set_xlim(0.0, np.pi)
    ax[1].set_xlim(0.0, np.pi)
    ax[2].set_xlim(0.0, np.pi)
    ax[3].set_xlim(0.0, np.pi)

    ax[0].set_ylim(0, 1.0)
    ax[1].set_ylim(0, 1.0)
    ax[2].set_ylim(0.9, 4.0)
    ax[3].set_ylim(0.9, 4.0)

    for i in range(u_true.shape[0]):
        if i < samples_to_plot:
            ax[0].plot(x_grid[:, 0].cpu(), u_true[i], color="blue", alpha=alpha)
            ax[1].plot(x_grid[:, 0].cpu(), u_gen[i, :].cpu(), color="blue", alpha=alpha)

            ax[2].plot(
                x_grid[:, 0].cpu(), h_true[i], color="green", alpha=alpha, label="Real"
            )
            ax[3].plot(
                x_grid[:, 0].cpu(),
                h_gen[i, :].cpu(),
                color="green",
                alpha=alpha,
                label="Inferred",
            )

    if x_sensor is not None:
        sns.rugplot(
            x=x_sensor[0, :, 0].cpu(),
            ax=ax[0],
            label="Sensors",
            color="red",
            height=0.2,
        )
        sns.rugplot(
            x=x_sensor[0, :, 0].cpu(),
            ax=ax[1],
            label="Sensors",
            color="red",
            height=0.2,
        )

    ax[0].legend()
    ax[1].legend()
    if plot_name is not None:
        plt.savefig(plot_name)
    if show_plot:
        plt.show()


def compare_mean_std_vs_x(
    x_grid, y_true, y_gen, variable="h", plot_name=None, show_plot=True
):
    mean_gen = torch.mean(y_gen, axis=0)
    mean_true = torch.mean(y_true, axis=0)
    var_gen = torch.var(y_gen, axis=0)
    var_true = torch.var(y_true, axis=0)
    print(variable, "MEAN MSE:", torch.mean(torch.square(mean_gen - mean_true)))
    print(variable, "VAR MSE:", torch.mean(torch.square(var_gen - var_true)))

    plt.figure(figsize=(8, 5))
    std_gen = torch.sqrt(var_gen).cpu().flatten()
    std_true = torch.sqrt(var_true).cpu().flatten()
    plt.plot(x_grid[:, 0].cpu(), mean_true, label="True", color="blue", linestyle="-")
    plt.fill_between(
        x_grid[:, 0].cpu(),
        mean_true - std_true,
        mean_true + std_true,
        color="blue",
        alpha=0.2,
    )
    plt.plot(
        x_grid[:, 0].cpu(), mean_gen, label="Generated", color="red", linestyle="-"
    )
    plt.fill_between(
        x_grid[:, 0].cpu(),
        mean_gen - std_gen,
        mean_gen + std_gen,
        color="red",
        alpha=0.2,
    )
    plt.xlabel("x")
    plt.ylabel(f"{variable}(x)")
    plt.legend()
    plt.grid(True)
    if plot_name is not None:
        plt.savefig(plot_name)
    if show_plot:
        plt.show()


def plot_loss(metrics, plot_name=None, show_plot=True):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].plot(np.log(metrics["reconstruction"]))
    ax[1].plot(np.log(metrics["residual"]))
    ax[2].plot(np.log(metrics["kld"]))

    fig.suptitle("Log Losses")
    ax[0].set_title("Reconstruction")
    ax[1].set_title("Residual")
    ax[2].set_title("KLD")
    if plot_name is not None:
        plt.savefig(plot_name)
    if show_plot:
        plt.show()


def compare_pt_wise_histograms(
    true_samples, gen_samples, variable, plot_name=None, show_plot=True
):

    fig, ax = plt.subplots()
    plt.hist(
        true_samples, bins=20, color="blue", alpha=0.3, edgecolor="k", label="True"
    )
    plt.hist(
        gen_samples, bins=20, color="red", alpha=0.3, edgecolor="k", label="Generated"
    )
    plt.xlabel(variable)
    plt.ylabel("Frequency")
    plt.legend()

    if plot_name is not None:
        plt.savefig(plot_name)
    if show_plot:
        plt.show()
