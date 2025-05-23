from argparse import ArgumentParser, BooleanOptionalAction
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from clfm.utils.utils import reparameterize
from clfm.problems.wind import WindDataset
from clfm.utils import dense_grid_eval
from clfm.nn.vae import FunctionalVAE
from wind_utils import get_result_and_model_path
from wind_plotting import (
    plot_train_loss,
    plot_val_loss,
    compare_coherence,
    compare_wind_stat,
    compare_energy,
)

# NOTE: must generate train/test data using data/wind_data_generation/generate_wind_data.py to run this script
TRAIN_DATA_FILE = Path(__file__).parent / "../../data/wind/wind_train_data.hdf5"
TEST_DATA_FILE = Path(__file__).parent / "../../data/wind/wind_test_data.hdf5"


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    NUM_SAMPLES = 250
    SHOW_PLOTS = True
    DISC = (1, 10, 10, 256)  # discretization for sampling (x1 x x2 x x3 x t)

    result_dir = Path(args.result_dir)
    case_params = vars(args)
    result_path, model_path = get_result_and_model_path(case_params, result_dir)
    vae = FunctionalVAE.load_from_checkpoint(model_path).to(args.device)

    data = WindDataset(TRAIN_DATA_FILE, args.sparse_sensors)
    loader = DataLoader(data, batch_size=NUM_SAMPLES, shuffle=True)
    data_test = WindDataset(TEST_DATA_FILE)
    loader_test = DataLoader(data_test, batch_size=NUM_SAMPLES, shuffle=False)

    u, _ = next(iter(loader))
    u = u.to(args.device)
    u_test, _ = next(iter(loader_test))
    u_test = u_test.to(args.device)
    dt = data.dt
    x = dense_grid_eval(vae.grid, DISC)
    x = x[None, :, :].repeat(NUM_SAMPLES, 1, 1)

    with torch.no_grad():
        mu, logvar = vae.encode(u)
        z = reparameterize(mu, logvar)
        u_hat = vae.decode(z, x).cpu()

    # Unnormalize data for plotting
    u_hat = u_hat * (data.v1_max - data.v1_min) + data.v1_min
    u_test = u_test * (data.v1_max - data.v1_min) + data.v1_min

    # optionall add plot_names to save figures
    plot_name = None
    plot_train_loss(
        result_path / "metrics.csv", plot_name=plot_name, show_plot=SHOW_PLOTS
    )
    plot_val_loss(
        result_path / "metrics.csv", plot_name=plot_name, show_plot=SHOW_PLOTS
    )
    compare_coherence(x, u_hat, DISC, dt, plot_name=None)
    compare_wind_stat(
        u_test, u_hat, x, DISC, stat="Mean", plot_name=plot_name, show_plot=SHOW_PLOTS
    )
    compare_wind_stat(
        u_test,
        u_hat,
        x,
        DISC,
        stat="Variance",
        plot_name=plot_name,
        show_plot=SHOW_PLOTS,
    )
    time_range = [data.grid.low[3], data.grid.high[3]]
    compare_energy(
        u_test, u_hat, DISC, time_range, plot_name=plot_name, show_plot=SHOW_PLOTS
    )


if __name__ == "__main__":
    parser = ArgumentParser()
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
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--result_dir", type=str, default="results/wind")
    parser.add_argument("--version", type=int, default=0)
    main(parser.parse_args())
