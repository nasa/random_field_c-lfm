import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import pandas as pd

from clfm.problems.gaussian_process import cov_func


def plot_convergence(result_path, plot_name=None, show_plot=True):
    metrics = pd.read_csv(result_path / "metrics.csv")
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


def compare_samples(
    x_true,
    y_true,
    x_gen,
    y_gen,
    x_sensor=None,
    num_samples_to_plot=250,
    plot_name=None,
    show_plot=True,
):
    _, ax = plt.subplots(1, 2, figsize=(10, 3))

    ax[0].set_title("Real Samples")
    ax[1].set_title("Generated Samples")
    ax[0].set_ylim(-3.0, 5.0)
    ax[1].set_ylim(-3.0, 5.0)
    for i in range(num_samples_to_plot):
        ax[0].plot(x_true[:, 0], y_true[i, :], color="blue", alpha=0.1)
        ax[1].plot(x_gen[:, 0].cpu(), y_gen[i, :].cpu(), color="blue", alpha=0.1)
    if x_sensor is not None:
        sns.rugplot(
            x=x_sensor[0, :, 0].cpu(),
            ax=ax[0],
            label="Sensors",
            color="red",
            height=0.2,
        )
    if plot_name is not None:
        plt.savefig(plot_name)
    if show_plot:
        plt.show()


def compare_covariance(
    x_grid,
    y_gen,
    gp_var=0.5,
    gp_cov_len=0.1,
    plot_name=None,
    show_plot=True,
):

    cov_true = cov_func(x_grid[:, :], gp_cov_len, gp_var).cpu()
    cov_est = torch.cov(y_gen.squeeze().T).cpu()

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].set_title("True Covariance")
    c1 = ax[0].pcolor(x_grid[:, 0].cpu(), x_grid[:, 0].cpu(), cov_true)
    fig.colorbar(c1, ax=ax[0])
    ax[1].set_title("Generated Covariance")
    c2 = ax[1].pcolor(x_grid[:, 0].cpu(), x_grid[:, 0].cpu(), cov_est)
    fig.colorbar(c2, ax=ax[1])

    ax[2].set_title("Error")
    c3 = ax[2].pcolor(x_grid[:, 0].cpu(), x_grid[:, 0].cpu(), (cov_est - cov_true))
    fig.colorbar(c3, ax=ax[2])
    if plot_name is not None:
        plt.savefig(plot_name)
    print("Covariance MSE: ", torch.mean(torch.square(cov_true - cov_est)))
    if show_plot:
        plt.show()


def compare_correlation(
    x_grid, y_gen, gp_var=0.5, gp_cov_len=0.1, plot_name=None, show_plot=True
):
    corr_true = cov_func(x_grid[:, :].cpu(), gp_cov_len, gp_var).cpu() / gp_var
    corr_est = torch.corrcoef(y_gen.squeeze().T).cpu()

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].set_title("True Correlation")
    c1 = ax[0].pcolor(x_grid[:, 0].cpu().cpu(), x_grid[:, 0].cpu(), corr_true)
    fig.colorbar(c1, ax=ax[0])

    ax[1].set_title("Generated Correlation")
    c2 = ax[1].pcolor(x_grid[:, 0].cpu(), x_grid[:, 0].cpu(), corr_est)
    fig.colorbar(c2, ax=ax[1])

    ax[2].set_title("Error")
    c3 = ax[2].pcolor(x_grid[:, 0].cpu(), x_grid[:, 0].cpu(), (corr_est - corr_true))
    fig.colorbar(c3, ax=ax[2])
    if plot_name is not None:
        plt.savefig(plot_name)
    if show_plot:
        plt.show()


def compare_mean_std(
    x_grid, y_gen, gp_var=0.5, gp_mean=1.0, plot_name=None, show_plot=True
):
    y_gen_mean = torch.mean(y_gen, 0)
    y_gen_var = torch.var(y_gen, 0)

    if type(gp_mean) is str and gp_mean.upper() == "LINEAR":
        mean_func = x_grid[:, 0].cpu().flatten()
    else:
        mean_func = np.ones(len(x_grid[:, 0])) * gp_mean

    x_plot = x_grid[:, 0].cpu()

    plt.figure(figsize=(8, 5))
    std_gen = torch.sqrt(y_gen_var).cpu().flatten()
    std_true = np.sqrt(np.ones(len(x_grid[:, 0])) * gp_var)

    plt.plot(x_plot, mean_func, label="True", color="blue", linestyle="-")
    plt.fill_between(
        x_plot, mean_func - std_true, mean_func + std_true, color="blue", alpha=0.2
    )

    y_gen_mean = y_gen_mean.cpu().flatten()
    plt.plot(x_plot, y_gen_mean, label="Generated", color="red", linestyle="-")
    plt.fill_between(
        x_plot, y_gen_mean - std_gen, y_gen_mean + std_gen, color="red", alpha=0.2
    )

    print("Mean MSE: ", torch.mean(torch.square(y_gen_mean - mean_func)))
    print("Variance MSE: ", torch.mean(torch.square(y_gen_mean - gp_var)))

    plt.xlabel("X")
    plt.ylabel("Mean Function")
    plt.legend()
    plt.title("Mean Function with ±1 Std Dev. Shaded Region")
    plt.grid(True)
    if plot_name is not None:
        plt.savefig(plot_name)
    if show_plot:
        plt.show()
