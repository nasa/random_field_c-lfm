from argparse import ArgumentParser
from pathlib import Path
import numpy as np
import torch

from clfm.nn.vae import FunctionalVAE
from clfm.nn.fully_connected_nets import FlowModel
from clfm.problems.materials import MaterialsTrain, MaterialsVal
from clfm.utils.latent_fm import sample_lfm
from materials_utils import get_result_and_model_path
from scripts.materials.materials_plotting import (
    plot_correlation_bands,
    compare_pdfs,
    plot_field_errors,
    plot_samples,
)


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    NUM_TEST_SAMPLES = 1000
    SHOW_PLOTS = True

    result_dir = Path(args.result_dir)
    case_params = vars(args)
    result_path, model_path = get_result_and_model_path(case_params, result_dir)

    vae = FunctionalVAE.load_from_checkpoint(model_path).to(args.device)

    N_GRID = 25
    train_data = MaterialsTrain()
    test_data = MaterialsVal(x_sensor=train_data.X_u)
    e_test = test_data.E
    x_test = test_data.X

    # Load VAE & flow matching models and generate samples
    vae = FunctionalVAE.load_from_checkpoint(model_path).to(args.device)

    fm_model_path = Path(result_path) / "fm.pth"
    checkpoint = torch.load(fm_model_path)
    flow_model = FlowModel(**checkpoint["architecture"])
    flow_model.load_state_dict(checkpoint["state_dict"])

    u_e_gen = sample_lfm(
        NUM_TEST_SAMPLES, x_test, flow_model, vae, args.num_time_steps_fm, args.device
    )

    ux_hat, uy_hat, e_hat = u_e_gen.chunk(3, dim=2)
    ux_hat = ux_hat.squeeze().reshape(-1, N_GRID, N_GRID).detach().cpu()
    uy_hat = uy_hat.squeeze().reshape(-1, N_GRID, N_GRID).detach().cpu()
    e_hat = e_hat.squeeze().reshape(-1, N_GRID, N_GRID).detach().cpu()
    e_test_reshape = e_test.reshape(-1, N_GRID, N_GRID).squeeze().cpu()

    # Plot convergence and results
    # optionally add plot_names to save the figures
    plot_name = None
    plot_samples(ux_hat, uy_hat, e_hat, plot_name=plot_name, show_plot=SHOW_PLOTS)
    plot_field_errors(e_test_reshape, e_hat, plot_name=plot_name, show_plot=SHOW_PLOTS)
    compare_pdfs(
        e_hat, e_test_reshape, x_test, N_GRID, plot_name=plot_name, show_plot=SHOW_PLOTS
    )
    results = generate_lfm_correlations_at_bands(
        e_test,
        test_data,
        flow_model,
        vae,
        NUM_TEST_SAMPLES,
        N_GRID,
        args.num_time_steps_fm,
        args.device,
    )
    plot_correlation_bands(results, plot_name=plot_name, show_plot=SHOW_PLOTS)


def generate_lfm_correlations_at_bands(
    test_E,
    test_data,
    flow_model,
    vae,
    num_samples=1000,
    num_points=25,
    num_time_steps=100,
    device="cuda",
    y_lines=None,
):
    """
    Generate samples and calculate correlations at horizontal bands across the domain.
    Returns:
    --------
    dict
        Dictionary containing points_x, true correlation, and generated correlations for each line
    """

    # Default y-coordinates if not provided
    if y_lines is None:
        y_lines = [0.75, 0.50, 0.25]

    # Create identifiers for the lines
    line_ids = [chr(65 + i) for i in range(len(y_lines))]  # A, B, C, ...
    # Dictionary to store results
    results = {}
    points_dict = {}
    e_values = {}
    e_corr = {}

    # Calculate true correlation for first line (line A)
    E_corr_A = torch.corrcoef(test_E[:, (test_data.X[:, 1] == y_lines[0])].T)
    results["true_corr"] = E_corr_A

    # Generate horizontal bands of points
    for i, (line_id, y_value) in enumerate(zip(line_ids, y_lines)):
        # Create points along a horizontal line
        points = torch.stack(
            [
                torch.linspace(0.0, 1.0, num_points),
                torch.full((num_points,), y_value),
            ],
            dim=1,
        )
        points_dict[line_id] = points

        # Sample using the provided sample_lfm function
        u_samples = sample_lfm(
            num_samples, points, flow_model, vae, num_time_steps, device
        )

        # Extract the e component (assuming u_samples has shape [samples, points, 3])
        _, _, e = u_samples.chunk(3, dim=2)
        e = e.squeeze()  # Remove extra dimensions if any
        e_values[line_id] = e

        # Compute empirical correlation
        e_corr[line_id] = torch.corrcoef(e.T)

    # Store results
    results["points_x"] = points_dict[line_ids[0]][:, 0].cpu()
    results["e_corr"] = e_corr
    results["line_ids"] = line_ids

    return results


if __name__ == "__main__":
    parser = ArgumentParser()
    # flow matching paramters:
    parser.add_argument("--num_time_steps_fm", type=int, default=100)
    # NN architecture / capacity
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--p_deeponet", type=int, default=128)
    parser.add_argument("--h_encoder", type=int, default=128)
    parser.add_argument("--h_branch", type=int, default=128)
    parser.add_argument("--h_trunk", type=int, default=128)
    parser.add_argument("--nhl_encoder", type=int, default=3)
    parser.add_argument("--nhl_branch", type=int, default=2)
    parser.add_argument("--nhl_trunk", type=int, default=2)
    parser.add_argument("--grad_clip", type=float, default=0.5)
    parser.add_argument("--bs", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    # method / run specifics
    parser.add_argument("--N_data", type=int, default=1000)
    parser.add_argument("--num_test_data", type=int, default=500)
    parser.add_argument("--num_colloc", type=int, default=100)
    parser.add_argument("--res_weight", type=float, default=1e-6)
    parser.add_argument("--lbc_weight", type=float, default=1.0)
    parser.add_argument("--rbc_weight", type=float, default=1.0)
    parser.add_argument("--kld_weight", type=float, default=1e-6)
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    # misc:
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--result_dir", type=str, default="results/materials")
    parser.add_argument("--version", type=int, default=0)
    main(parser.parse_args())
