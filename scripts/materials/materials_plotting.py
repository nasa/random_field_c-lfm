import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.stats import gaussian_kde


def plot_field_errors(
    e_true, e_hat, figsize=(12, 5), cmap="viridis", plot_name=None, show_plot=True
):
    """
    Plot the errors between true and generated fields for both mean and standard deviation.

    Parameters:
    -----------
    e_true : torch.Tensor or numpy.ndarray
        True data tensor, expected shape: (n_samples, height, width)
    e_hat : torch.Tensor or numpy.ndarray
        Generated data tensor, expected shape: (n_samples, height, width)
    figsize : tuple, optional
        Size of the output figure, default: (12, 5)
    cmap : str, optional
        Colormap to use for the plots, default: 'viridis'

    """

    # Convert torch tensors to numpy if needed
    if hasattr(e_true, "cpu") and hasattr(e_true, "numpy"):
        # Calculate statistics before converting to numpy to use torch's efficient implementations
        e_true_mean = e_true.mean(dim=0).cpu().numpy()
        e_true_std = e_true.std(dim=0).cpu().numpy()
    else:
        e_true_mean = np.mean(e_true, axis=0)
        e_true_std = np.std(e_true, axis=0)

    if hasattr(e_hat, "cpu") and hasattr(e_hat, "numpy"):
        e_hat_mean = e_hat.mean(dim=0).cpu().numpy()
        e_hat_std = e_hat.std(dim=0).cpu().numpy()
    else:
        e_hat_mean = np.mean(e_hat, axis=0)
        e_hat_std = np.std(e_hat, axis=0)

    # Calculate errors
    mean_error = e_true_mean - e_hat_mean
    std_error = e_true_std - e_hat_std

    # Create figure and axes
    fig, ax = plt.subplots(1, 2, figsize=figsize)

    # Plot mean error
    c0 = ax[0].pcolor(mean_error, cmap=cmap)
    cbar0 = fig.colorbar(c0, ax=ax[0])
    cbar0.set_label("Error magnitude")

    # Plot std error
    c1 = ax[1].pcolor(std_error, cmap=cmap)
    cbar1 = fig.colorbar(c1, ax=ax[1])
    cbar1.set_label("Error magnitude")

    # Set titles and labels
    ax[0].set_title("Error of Mean Fields")
    ax[0].set_xlabel("x1")
    ax[0].set_ylabel("x2")

    ax[1].set_title("Error of Std Fields")
    ax[1].set_xlabel("x1")
    ax[1].set_ylabel("x2")

    # Improve layout
    fig.tight_layout()
    if plot_name is not None:
        plt.savefig(plot_name)
    if show_plot:
        plt.show()


def plot_samples(
    ux_hat,
    uy_hat,
    e_hat,
    num_samples=3,
    figsize=None,
    cmap="viridis",
    plot_name=None,
    show_plot=True,
):
    """
    Plot reconstructed samples of ux, uy, and e fields.

    Parameters:
    -----------
    ux_hat : torch.Tensor or numpy.ndarray
        Generated ux field tensor, expected shape: (batch_size, height, width)
    uy_hat : torch.Tensor or numpy.ndarray
        Generated uy field tensor, expected shape: (batch_size, height, width)
    e_hat : torch.Tensor or numpy.ndarray
        Generated e field tensor, expected shape: (batch_size, height, width)
    num_samples : int, optional
        Number of samples to plot, default: 3
    figsize : tuple, optional
        Size of the output figure, default: (15, num_samples * 5)
    cmap : str, optional
        Colormap to use for the plots, default: 'viridis'

    """

    # Set default figure size if not provided
    if figsize is None:
        figsize = (15, num_samples * 5)

    # Convert torch tensors to numpy if needed
    if hasattr(ux_hat, "cpu") and hasattr(ux_hat, "numpy"):
        ux_hat = ux_hat.cpu().numpy()
    if hasattr(uy_hat, "cpu") and hasattr(uy_hat, "numpy"):
        uy_hat = uy_hat.cpu().numpy()
    if hasattr(e_hat, "cpu") and hasattr(e_hat, "numpy"):
        e_hat = e_hat.cpu().numpy()

    # Ensure we don't try to plot more samples than we have
    num_samples = min(num_samples, len(ux_hat))

    # Create figure and axes
    fig, ax = plt.subplots(num_samples, 3, figsize=figsize)

    # Handle the case where num_samples is 1
    if num_samples == 1:
        ax = np.array([ax])

    # Set column titles
    ax[0, 0].set_title("Generated ux")
    ax[0, 1].set_title("Generated uy")
    ax[0, 2].set_title("Generated e")

    # Plot each sample
    for i in range(num_samples):
        # Plot ux field
        c0 = ax[i, 0].pcolor(ux_hat[i], cmap=cmap)
        cbar0 = fig.colorbar(c0, ax=ax[i, 0])
        cbar0.set_label("ux magnitude")

        # Plot uy field
        c1 = ax[i, 1].pcolor(uy_hat[i], cmap=cmap)
        cbar1 = fig.colorbar(c1, ax=ax[i, 1])
        cbar1.set_label("uy magnitude")

        # Plot e field
        c2 = ax[i, 2].pcolor(e_hat[i], cmap=cmap)
        cbar2 = fig.colorbar(c2, ax=ax[i, 2])
        cbar2.set_label("e magnitude")

        # Add row labels if desired
        ax[i, 0].set_ylabel(f"Sample {i+1}")

    # Improve layout
    fig.tight_layout()
    if plot_name is not None:
        plt.savefig(plot_name)
    if show_plot:
        plt.show()


def compare_pdfs(
    e_hat,
    e_true,
    test_x,
    n_grid,
    points=None,
    figure_size=(10, 6),
    plot_name=None,
    show_plot=True,
):
    """
    Compare probability density functions (PDFs) of generated and true data at specific points.

    Parameters:
    -----------
    e_hat : numpy.ndarray or torch.Tensor
        Generated data tensor, expected shape: (n_samples, grid_size, grid_size)
    e_true : numpy.ndarray or torch.Tensor
        True data tensor, expected shape: (n_samples, grid_size, grid_size)
    test_x : numpy.ndarray or torch.Tensor
        Coordinates of test data points
    n_grid : int
        Size of the grid (assuming square grid)
    points : dict, optional
        Dictionary of points to analyze, format: {'name': (x,y)}
        Default: {'A': (0.25, 0.75), 'B': (0.5, 0.5), 'C': (0.25, 0.25)}
    figure_size : tuple, optional
        Size of the output figure, default: (10, 6)

    """

    # Convert torch tensors to numpy if needed
    if hasattr(e_hat, "cpu") and hasattr(e_hat, "numpy"):
        e_hat = e_hat.cpu().numpy()
    if hasattr(e_true, "cpu") and hasattr(e_true, "numpy"):
        e_true = e_true.cpu().numpy()
    if hasattr(test_x, "cpu") and hasattr(test_x, "numpy"):
        test_x = test_x.cpu().numpy()

    # Default points of interest if none provided
    if points is None:
        points = {"A": (0.25, 0.75), "B": (0.5, 0.5), "C": (0.25, 0.25)}

    # Helper function to find the nearest grid point to a target point
    def find_nearest_index(xg, point):
        distances = np.sqrt((xg[0] - point[0]) ** 2 + (xg[1] - point[1]) ** 2)
        idx = np.unravel_index(np.argmin(distances), distances.shape)
        return idx

    # Reshape test_x to grid format
    xg = test_x.T.reshape(-1, n_grid, n_grid)

    # Find indices of nearest grid points to our points of interest
    indices = {}
    for name, point in points.items():
        indices[name] = find_nearest_index(xg, point)

    # Extract e data at these points across all samples
    e_values = {}
    for name, idx in indices.items():
        e_values[name] = e_hat[:, idx[0], idx[1]]

    # Create KDE for each point
    kde_values = {}
    for name, e_val in e_values.items():
        kde_values[name] = gaussian_kde(e_val)

    # Reference KDE (truth at point B or the middle point)
    reference_point = "B"  # Using point B as reference
    ref_idx = indices[reference_point]
    e_ref = e_true[:, ref_idx[0], ref_idx[1]]
    kde_ref = gaussian_kde(e_ref)

    # Create a range of x values to evaluate the KDEs
    all_e_values = list(e_values.values())
    min_e = min(v.min() for v in all_e_values) - 0.5
    max_e = max(v.max() for v in all_e_values) + 0.5

    x_eval = np.linspace(min_e, max_e, 1000)

    # Evaluate KDEs
    pdfs = {}
    for name, kde in kde_values.items():
        pdfs[name] = kde(x_eval)
    pdf_ref = kde_ref(x_eval)

    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=figure_size)

    # Plot reference (truth)
    ax.plot(x_eval, pdf_ref, label="True", color="black", linewidth=2)

    # Plot generated PDFs
    colors = {"A": "red", "B": "blue", "C": "green"}
    for name, pdf in pdfs.items():
        color = colors.get(name, None)  # Use predefined color or let matplotlib decide
        ax.plot(x_eval, pdf, label=f"Generated - Pt. {name}", color=color)

    # Add labels and legend
    ax.set_xlabel("E")
    ax.set_ylabel("p(E)")
    ax.legend()
    ax.set_title("PDF Comparison at Different Points")
    if plot_name is not None:
        plt.savefig(plot_name)
    if show_plot:
        plt.show()


def plot_correlation_comparison_from_vae(
    vae, test_E, test_x, z, device, plot_name=None, show_plot=True
):
    """
    Generate and plot correlation comparison between true data and generated data
    at three horizontal lines at different y-values.

    Parameters:
    -----------
    vae : torch.nn.Module
        The VAE model used for decoding
    test_E : torch.Tensor
        Test data tensor for true correlation calculation
    test_x : torch.Tensor
        Coordinates of test data points
    z : torch.Tensor
        Latent vectors for generation
    device : torch.device
        Device to perform computations on

    """
    # Number of points along each line
    NUM_POINTS = 25

    # Define y-coordinates for horizontal lines
    y_lines = {"A": 0.75, "B": 0.50, "C": 0.25}

    # Calculate true correlation for line A
    E_corr_A = torch.corrcoef(test_E[:, (test_x[:, 1] == y_lines["A"])].T)

    # Generate points and correlations for each horizontal line
    points = {}
    e_values = {}
    e_corr = {}

    for line_id, y_value in y_lines.items():
        # Create a horizontal line of points
        points[line_id] = (
            torch.stack(
                [
                    torch.linspace(0.0, 1.0, NUM_POINTS),
                    torch.full((NUM_POINTS,), y_value),
                ],
                dim=1,
            )
            .unsqueeze(0)
            .repeat(z.shape[0], 1, 1)
            .to(device)
        )

        # Evaluate model at these points
        with torch.no_grad():
            # Assuming vae.decode returns a tuple with e as the third element
            _, _, e_values[line_id] = (
                vae.decode(z, points[line_id]).cpu().chunk(3, dim=2)
            )

        # Squeeze to remove extra dimensions
        e_values[line_id] = e_values[line_id].squeeze()

        # Compute empirical correlation
        e_corr[line_id] = torch.corrcoef(e_values[line_id].T)

    # Extract x-coordinates for plotting
    points_x = points["A"][0, :, 0].cpu()

    # Find the midpoint index
    idx = int(NUM_POINTS / 2)

    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Plot true correlation
    ax.plot(points_x, E_corr_A[idx, :], label="True", color="black")

    # Plot generated correlations
    line_colors = {"A": "red", "B": "purple", "C": "green"}

    for line_id, color in line_colors.items():
        ax.plot(
            points_x,
            e_corr[line_id][idx, :],
            label=f"Generated ({line_id})",
            color=color,
            linestyle="dashed",
        )

    # Add legend and labels
    ax.legend()
    ax.set_xlabel("x1")
    ax.set_ylabel("C(x1, 0.5)")
    ax.set_title("Correlation Comparison at Different y-values")
    if plot_name is not None:
        plt.savefig(plot_name)
    if show_plot:
        plt.show()


def plot_train_loss(metrics_file, plot_name=None, show_plot=True):

    val_metrics = [
        "val_loss",
        "val_loss_observable",
        "val_loss_unobservable",
        "val_corr_loss",
    ]
    metrics = pd.read_csv(metrics_file).drop(val_metrics, axis=1).dropna()

    fig, ax = plt.subplots(1, 4, figsize=(15, 5))
    fig.suptitle("Log scale losses")

    ax[0].set_title("Reconstruction")
    ax[0].plot(np.log(metrics["reconstruction"]))
    ax[0].set_xlabel("Epoch")

    ax[1].set_title("Residual")
    ax[1].plot(np.log(metrics["residual"]))
    ax[1].set_xlabel("Epoch")

    ax[2].set_title("Boundary")
    ax[2].plot(np.log(metrics["boundary"]))
    ax[2].set_xlabel("Epoch")

    ax[3].set_title("KL Divergence")
    ax[3].plot(np.log(metrics["kld"]))
    ax[3].set_xlabel("Epoch")
    if plot_name is not None:
        plt.savefig(plot_name)
    if plot_name is not None:
        plt.savefig(plot_name)
    if show_plot:
        plt.show()


def plot_val_loss(metrics_file, plot_name=None, show_plot=True):
    val_loss = pd.read_csv(metrics_file)["val_loss"].dropna()
    val_loss_observable = pd.read_csv(metrics_file)["val_loss_observable"].dropna()
    val_loss_unobservable = pd.read_csv(metrics_file)["val_loss_unobservable"].dropna()
    val_corr_loss = pd.read_csv(metrics_file)["val_corr_loss"].dropna()

    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    ax[0].set_ylabel("Log Validation Error")
    ax[0].set_xlabel("Epoch")
    ax[0].plot(np.log(val_loss))
    ax[0].set_title("Log Validation Loss")

    ax[1].set_title("Log Validation Observable Loss")
    ax[1].plot(np.log(val_loss_observable))

    ax[2].set_title("Log Validation Unobservable Loss")
    ax[2].plot(np.log(val_loss_unobservable))

    ax[3].set_title("Log Validation Correlation Loss")
    ax[3].plot(np.log(val_corr_loss))
    if plot_name is not None:
        plt.savefig(plot_name)
    if plot_name is not None:
        plt.savefig(plot_name)
    if show_plot:
        plt.show()


def plot_correlation_bands(
    results, figsize=(10, 6), y_limits=None, plot_name=None, show_plot=True
):
    """
    Plot correlation bands from the results of generate_correlations_at_bands.

    Parameters:
    -----------
    results : dict
        Results dictionary from generate_correlations_at_bands
    figsize : tuple, optional
        Size of the figure, default: (10, 6)
    y_limits : tuple or None, optional
        Y-axis limits (min, max), default: None (auto-scaling)

    """

    # Extract data from results
    points_x = results["points_x"]
    true_corr = results["true_corr"]
    e_corr = results["e_corr"]
    line_ids = results["line_ids"]

    # Finding the midpoint on the x axis
    idx = int(len(points_x) / 2)

    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Plot true correlation
    ax.plot(points_x, true_corr[idx, :], label="True", color="black")

    # Plot generated correlations
    line_colors = {"A": "red", "B": "purple", "C": "green", "D": "blue", "E": "orange"}

    for line_id in line_ids:
        color = line_colors.get(
            line_id, None
        )  # Use predefined color or let matplotlib decide
        ax.plot(
            points_x,
            e_corr[line_id][idx, :],
            label=f"Generated ({line_id})",
            color=color,
            linestyle="dashed",
        )

    # Set y-limits if provided
    if y_limits is not None:
        ax.set_ylim(y_limits)

    # Add legend and labels
    ax.legend()
    ax.set_xlabel("x1")
    ax.set_ylabel("C(x1, 0.5)")
    ax.set_title("Correlation at Different y-values")
    if plot_name is not None:
        plt.savefig(plot_name)
    if show_plot:
        plt.show()
