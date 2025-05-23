from argparse import ArgumentParser, BooleanOptionalAction
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from clfm.problems.wind import WindDataset
from clfm.nn.vae import FunctionalVAE
from clfm.nn.fully_connected_nets import FlowModel
from clfm.utils.latent_fm import sample_lfm
from clfm.utils.utils import dense_grid_eval
from wind_utils import get_result_and_model_path
from wind_plotting import (
    plot_train_loss,
    plot_val_loss,
    compare_coherence,
    compare_wind_stat,
    compare_energy,
)

# NOTE: must generate train/test data using data/wind_data_generation/generate_wind_data.py to run this script
TEST_DATA_FILE = Path(__file__).parent / "../../data/wind/wind_test_data.hdf5"


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    NUM_SAMPLES = 500
    SHOW_PLOTS = True
    DISC = (1, 10, 10, 256)  # discretization for sampling (x1 x x2 x x3 x t)
    result_dir = Path(args.result_dir)
    case_params = vars(args)
    result_path, model_path = get_result_and_model_path(case_params, result_dir)

    data_test = WindDataset(TEST_DATA_FILE)
    loader_test = DataLoader(data_test, batch_size=NUM_SAMPLES, shuffle=False)
    u_test, _ = next(iter(loader_test))
    u_test = u_test.to(args.device)
    x_grid = dense_grid_eval(data_test.grid, DISC)

    # load vae, flow matching models & generate samples on x_grid
    vae = FunctionalVAE.load_from_checkpoint(model_path).to(args.device)
    fm_model_path = Path(result_path) / "fm.pth"
    checkpoint = torch.load(fm_model_path)
    flow_model = FlowModel(**checkpoint["architecture"])
    flow_model.load_state_dict(checkpoint["state_dict"])
    u_hat = sample_lfm(
        NUM_SAMPLES, x_grid, flow_model, vae, args.num_time_steps_fm, args.device
    )

    # Unnormalize data for plotting
    u_hat = u_hat * (data_test.v1_max - data_test.v1_min) + data_test.v1_min
    u = u_test * (data_test.v1_max - data_test.v1_min) + data_test.v1_min

    # optionall add plot_names to save figures
    plot_name = None
    plot_train_loss(
        result_path / "metrics.csv", plot_name=plot_name, show_plot=SHOW_PLOTS
    )
    plot_val_loss(
        result_path / "metrics.csv", plot_name=plot_name, show_plot=SHOW_PLOTS
    )
    x = x_grid[None, :, :].repeat(NUM_SAMPLES, 1, 1)
    compare_coherence(
        x, u_hat, DISC, data_test.dt, plot_name=plot_name, show_plot=SHOW_PLOTS
    )
    compare_wind_stat(
        u, u_hat, x, DISC, stat="Mean", plot_name=plot_name, show_plot=SHOW_PLOTS
    )
    compare_wind_stat(
        u, u_hat, x, DISC, stat="Variance", plot_name=plot_name, show_plot=SHOW_PLOTS
    )
    t_grid = np.array(data_test.f["t_grid"])
    time_range = [t_grid[0], t_grid[-1]]
    compare_energy(
        u, u_hat, DISC, time_range, plot_name=plot_name, show_plot=SHOW_PLOTS
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    # flow matching paramters:
    parser.add_argument("--num_time_steps_fm", type=int, default=100)
    # NN architecture / capacity
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--p_deeponet", type=int, default=128)
    parser.add_argument("--h_branch", type=int, default=128)
    parser.add_argument("--h_trunk", type=int, default=128)
    parser.add_argument("--nhl_branch", type=int, default=2)
    parser.add_argument("--nhl_trunk", type=int, default=3)
    parser.add_argument("--bs", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--grad_clip", type=float, default=0.5)
    # method / run specifics
    parser.add_argument("--sparse_sensors", action=BooleanOptionalAction, default=False)
    parser.add_argument("--T_colloc", type=int, default=128)
    parser.add_argument("--num_colloc", type=int, default=16)
    parser.add_argument("--res_weight", type=float, default=0.01)
    parser.add_argument("--kld_weight", type=float, default=1e-7)
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    # misc / hpc:
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--result_dir", type=str, default="results/wind")
    parser.add_argument("--version", type=int, default=0)
    main(parser.parse_args())
