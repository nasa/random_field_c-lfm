from argparse import ArgumentParser
import torch
import numpy as np
from pathlib import Path

from clfm.problems.poisson_1d import Poisson1DDataset
from clfm.nn.fully_connected_nets import FlowModel
from clfm.nn.vae import FunctionalVAE
from clfm.utils.latent_fm import sample_lfm
from poisson_utils import get_result_and_model_path
from poisson_plotting import (
    compare_samples,
    compare_mean_std_vs_x,
    compare_pt_wise_histograms,
)


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    NUM_SAMPLES = 1000
    NUM_SAMPLES_PLOT = 500
    SHOW_PLOTS = True
    X_TEST_GRID = 101

    result_dir = Path(args.result_dir)
    case_params = vars(args)
    result_path, model_path = get_result_and_model_path(case_params, result_dir)

    # Get true u/v samples and x coords for sensors and plotting
    test_data = Poisson1DDataset(NUM_SAMPLES, X_TEST_GRID)
    u_true = test_data.u_samples
    v_true = test_data.v_samples
    x_sensor = (
        torch.linspace(0, torch.pi, case_params["num_sensors"])
        .reshape(1, -1, 1)
        .to(args.device)
    )
    x_grid = torch.linspace(0, torch.pi, X_TEST_GRID).reshape(-1, 1).to(args.device)

    # Load VAE & flow matching models and generate samples
    vae = FunctionalVAE.load_from_checkpoint(model_path).to(args.device)

    fm_model_path = Path(result_path) / "fm.pth"
    checkpoint = torch.load(fm_model_path)
    flow_model = FlowModel(**checkpoint["architecture"])
    flow_model.load_state_dict(checkpoint["state_dict"])

    u_h_gen = sample_lfm(
        NUM_SAMPLES, x_grid, flow_model, vae, args.num_time_steps_fm, args.device
    )
    u_hat, v_hat = u_h_gen.chunk(2, dim=2)
    u_hat, v_hat = u_hat.squeeze(2).cpu(), v_hat.squeeze(2).cpu()

    # optionally add plot_names to save figures
    plot_name = None
    compare_samples(
        x_grid,
        u_true,
        v_true,
        u_hat,
        v_hat,
        x_sensor,
        samples_to_plot=NUM_SAMPLES_PLOT,
        plot_name=plot_name,
        show_plot=SHOW_PLOTS,
    )
    compare_mean_std_vs_x(
        x_grid, u_true, u_hat, variable="u", plot_name=plot_name, show_plot=SHOW_PLOTS
    )
    compare_mean_std_vs_x(
        x_grid, v_true, v_hat, variable="v", plot_name=plot_name, show_plot=SHOW_PLOTS
    )

    # Compare histograms for u at x=pi/2
    u_true_samples = u_true[:, 50]
    u_gen_samples = u_hat[:, 50]
    compare_pt_wise_histograms(
        u_true_samples,
        u_gen_samples,
        variable="u(pi/2)",
        plot_name=plot_name,
        show_plot=SHOW_PLOTS,
    )
    # Compare histograms for v at x=pi
    v_true_samples = v_true[:, -1]
    v_gen_samples = v_hat[:, -1]
    compare_pt_wise_histograms(
        v_true_samples,
        v_gen_samples,
        variable="v(pi)",
        plot_name=plot_name,
        show_plot=SHOW_PLOTS,
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    # flow matching paramters:
    parser.add_argument("--num_time_steps_fm", type=int, default=100)
    # problem set up
    parser.add_argument("--N_data", type=int, default=500)
    # NN architecture / capacity
    parser.add_argument("--latent_dim", type=int, default=4)
    parser.add_argument("--p_deeponet", type=int, default=64)
    parser.add_argument("--h_encoder", type=int, default=128)
    parser.add_argument("--h_trunk", type=int, default=128)
    parser.add_argument("--h_branch", type=int, default=128)
    parser.add_argument("--nhl_trunk", type=int, default=2)
    parser.add_argument("--nhl_branch", type=int, default=2)
    parser.add_argument("--nhl_encoder", type=int, default=3)
    parser.add_argument("--grad_clip", type=float, default=0.5)
    parser.add_argument("--bs", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    # method / run specifics
    parser.add_argument("--num_sensors", type=int, default=25)
    parser.add_argument("--num_colloc", type=int, default=50)
    parser.add_argument("--res_weight", type=float, default=0.001)
    parser.add_argument("--kld_weight", type=float, default=1e-6)
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    # misc:
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--result_dir", type=str, default="results/poisson_1d")
    parser.add_argument("--version", type=int, default=0)
    main(parser.parse_args())
