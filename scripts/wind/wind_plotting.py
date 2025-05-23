import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd

from clfm.problems.wind import get_true_coherence, v_coherence


def compare_coherence(x, u_gen, disc, dt, nperseg=None, plot_name=None, show_plot=True):
    # Reshape the data
    u_bar = u_gen.reshape(-1, disc[1] * disc[2], disc[3]).cpu()
    x_bar = x.reshape(-1, disc[1] * disc[2], disc[3], 4)
    x_bar = x_bar[:, :, :, :3].cpu()

    if nperseg is None:
        nperseg = disc[3]

    # Create a randomly permuted version of u_bar
    # This will shuffle the points (second dimension) while keeping the time series intact
    permutation = torch.randperm(u_bar.shape[1])
    u_bar_permuted = u_bar[:, permutation, :]

    # Get the grid locations of the original and permuted points
    x_hat_original = x_bar[0, :, 0, :]  # Original grid positions
    x_hat_permuted = x_bar[0, permutation, 0, :]  # Permuted grid positions

    # Calculate coherence between original and permuted time series
    coh = v_coherence(u_bar, u_bar_permuted, nperseg=nperseg)

    # Get frequency grid
    freq_grid = torch.fft.rfftfreq(nperseg, d=dt)

    # Calculate true coherence for all 100 pairs
    true_coh = []
    for i in range(u_bar.shape[1]):  # Loop through all points (disc[1]*disc[2])
        true_coh.append(
            torch.tensor(
                [
                    get_true_coherence(
                        f, x_hat_original[i, :].cpu(), x_hat_permuted[i, :].cpu()
                    )
                    ** 2
                    for f in freq_grid
                ]
            )
        )

    fig, ax = plt.subplots(4, 4, figsize=(9, 6), sharex=True, sharey=True)
    for i in range(ax.shape[0]):
        for j in range(ax.shape[1]):
            idx = np.random.randint(x_hat_original.shape[0])
            ax[i, j].plot(freq_grid, true_coh[idx], label="True")
            ax[i, j].plot(freq_grid, coh[idx, :], label="Generated")
            if j == 0:
                ax[i, j].set_ylabel("Coherence")
            if i == ax.shape[0] - 1:
                ax[i, j].set_xlabel("Frequency")
    true_coh = torch.stack(true_coh).to(coh.device)
    residual = torch.mean(torch.square(true_coh - coh))
    print("Coherence Residual:", residual)

    fig.suptitle("Coherence Comparison")
    plt.legend()
    if plot_name is not None:
        plt.savefig(plot_name)
    if show_plot:
        plt.show()


def compare_energy(u_true, u_gen, disc, time_range, plot_name=None, show_plot=True):
    u_gen = u_gen.reshape(-1, *disc)
    generated_energy = (1 / (disc[1] * disc[2])) * torch.square(u_gen).sum(
        (1, 2, 3)
    ).cpu()
    true_energy = (1 / u_true.shape[1]) * torch.square(u_true).sum(1).cpu()
    t_gen = torch.linspace(time_range[0], time_range[1], disc[3])
    t_true = torch.linspace(time_range[0], time_range[1], u_true.shape[2])

    _, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].set_title("Energy of True Samples")
    ax[0].set_xlabel("Time (s)")
    ax[0].set_ylabel("Energy")
    ax[1].set_title("Energy of Generated Samples")
    ax[1].set_xlabel("Time (s)")
    ax[1].set_ylabel("Energy")
    for i in range(generated_energy.shape[0]):
        ax[0].plot(t_true, true_energy[i, :], color="red", alpha=0.05)
        ax[1].plot(t_gen, generated_energy[i, :], color="red", alpha=0.05)

    if plot_name is not None:
        plt.savefig(plot_name)
    if show_plot:
        plt.show()


def compare_wind_stat(
    u_true, u_gen, x, disc, stat="Mean", plot_name=None, show_plot=True
):
    u_bar = u_gen.reshape(-1, disc[1] * disc[2], disc[3]).cpu()
    x_bar = x.reshape(-1, disc[1] * disc[2], disc[3], 4)
    x_bar = x_bar[0, :, 0, :3].cpu()
    u_true = u_true.cpu()

    if stat.upper() == "MEAN":
        u_gen_stat = torch.mean(u_bar, [0, 2])
        u_true_stat = torch.mean(u_true, [0, 2])
        print("MEAN MSE:", torch.mean(torch.square(u_gen_stat - u_true_stat)))
    elif stat.upper() == "VARIANCE":
        u_gen_stat = torch.var(u_bar, [0, 2])
        u_true_stat = torch.var(u_true, [0, 2])
        print("VARIANCE MSE:", torch.mean(torch.square(u_gen_stat - u_true_stat)))
    else:
        raise ValueError("Invalid Stat")

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].set_title(f"True {stat} Velocity")
    ax[0].set_xlabel("y")
    ax[0].set_ylabel("z")
    ax[1].set_title(f"Generated {stat} Velocity")
    ax[1].set_xlabel("y")
    ax[1].set_ylabel("z")
    ax[2].set_title("Absolute Error (True-Gen)")
    ax[2].set_xlabel("y")
    ax[2].set_ylabel("z")
    cntr0 = ax[0].tricontourf(x_bar[:, 1], x_bar[:, 2], u_true_stat, cmap="viridis")
    cntr1 = ax[1].tricontourf(x_bar[:, 1], x_bar[:, 2], u_gen_stat, cmap="viridis")
    error = u_true_stat - u_gen_stat
    cntr2 = ax[2].tricontourf(x_bar[:, 1], x_bar[:, 2], error, cmap="viridis")
    fig.colorbar(cntr0, ax=ax[0])
    fig.colorbar(cntr1, ax=ax[1])
    fig.colorbar(cntr2, ax=ax[2])
    if plot_name is not None:
        plt.savefig(plot_name)
    if show_plot:
        plt.show()


def compare_coherence_contours(
    x, u_gen, N, T, dt, freq_index=None, nperseg=None, plot_name=None, show_plot=True
):

    u_bar = u_gen.reshape(-1, N, T).cpu()
    x_bar = x.reshape(-1, N, T, 4)
    x_bar = x_bar[0, :, 0, :3].cpu()
    if nperseg is None:
        nperseg = T

    u_hat_1 = torch.repeat_interleave(u_bar, N, dim=1)
    u_hat_2 = u_bar.tile((1, N, 1))
    coh = v_coherence(u_hat_1, u_hat_2, nperseg=nperseg)

    freq_grid = torch.fft.rfftfreq(nperseg, d=dt)
    true_coh = []
    for i in range(N):
        for j in range(N):
            true_coh.append(
                torch.tensor(
                    [
                        get_true_coherence(f, x_bar[i, :].cpu(), x_bar[j, :].cpu()) ** 2
                        for f in freq_grid
                    ]
                )
            )

    if freq_index is None:
        freq_index = 20
    # NOTE: hardcoded grid for now
    x_plot = torch.linspace(1.0, 100.0, N)
    cov_true = torch.zeros(N * N, dtype=torch.float32)
    cov_est = torch.zeros(N * N, dtype=torch.float32)
    for i in range(N * N):
        cov_est[i] = coh[i, freq_index]
        cov_true[i] = true_coh[i][freq_index]
    cov_true = cov_true.reshape(N, N)
    cov_est = cov_est.reshape(N, N)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    title = "True Coherence " + "(f={0: 0.2f}hz)".format(freq_grid[freq_index])
    ax[0].set_title(title)
    c1 = ax[0].pcolor(x_plot, x_plot, cov_true)
    fig.colorbar(c1, ax=ax[0])

    title = "Generated Coherence " + "(f={0: 0.2f}hz)".format(freq_grid[freq_index])
    ax[1].set_title(title)
    c2 = ax[1].pcolor(x_plot, x_plot, cov_est)
    fig.colorbar(c2, ax=ax[1])

    ax[2].set_title("Error")
    c3 = ax[2].pcolor(x_plot, x_plot, (cov_est - cov_true))
    fig.colorbar(c3, ax=ax[2])
    if plot_name is not None:
        plt.savefig(plot_name)
    if show_plot:
        plt.show()


def plot_train_loss(metrics_file, plot_name=None, show_plot=True):

    val_metrics = ["val_coherence_mse", "val_mean_mse", "val_var_mse"]
    metrics = pd.read_csv(metrics_file).drop(val_metrics, axis=1).dropna()

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Log scale losses")

    ax[0].set_title("Reconstruction")
    ax[0].plot(np.log10(metrics["reconstruction"]))
    ax[0].set_xlabel("Epoch")

    ax[1].set_title("Residual")
    ax[1].plot(np.log10(metrics["residual"]))
    ax[1].set_xlabel("Epoch")

    ax[2].set_title("KL Divergence")
    ax[2].plot(np.log10(metrics["kld"]))
    ax[2].set_xlabel("Epoch")
    if plot_name is not None:
        plt.savefig(plot_name)
    if show_plot:
        plt.show()


def plot_val_loss(metrics_file, plot_name=None, show_plot=True):
    val_coherence_mse = pd.read_csv(metrics_file)["val_coherence_mse"].dropna()
    val_mean_mse = pd.read_csv(metrics_file)["val_mean_mse"].dropna()
    val_var_mse = pd.read_csv(metrics_file)["val_var_mse"].dropna()

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].set_ylabel("Log Validation Error")
    ax[0].set_xlabel("Epoch")
    ax[0].plot(np.log10(val_coherence_mse))
    ax[0].set_title("Coherence Error")

    ax[1].set_title("Mean Error")
    ax[1].plot(np.log10(val_mean_mse))

    ax[2].set_title("Variance Error")
    ax[2].plot(np.log10(val_var_mse))
    if plot_name is not None:
        plt.savefig(plot_name)
    if show_plot:
        plt.show()
