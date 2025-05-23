from argparse import ArgumentParser
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

from clfm.problems.poisson_1d import Poisson1DDataset
from clfm.nn.vae import FunctionalVAE
from clfm.utils import reparameterize
from poisson_utils import get_result_and_model_path
from poisson_plotting import (
    compare_samples,
    compare_mean_std_vs_x,
    plot_loss,
    compare_pt_wise_histograms,
)


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    NUM_TEST_SAMPLES = 1000
    NUM_SAMPLES_PLOT = 500
    SHOW_PLOTS = True
    result_dir = Path(args.result_dir)

    case_params = vars(args)
    result_path, model_path = get_result_and_model_path(case_params, result_dir)
    vae = FunctionalVAE.load_from_checkpoint(model_path).to(args.device)

    train_data = Poisson1DDataset(NUM_TEST_SAMPLES, case_params["num_sensors"])
    test_data = Poisson1DDataset(NUM_TEST_SAMPLES, num_sensors=100)

    metrics = pd.read_csv(result_path / "metrics.csv")
    train_loader = DataLoader(train_data, batch_size=len(train_data), shuffle=True)
    u_train, x_sensor = next(iter(train_loader))
    u_train, x_sensor = u_train.to(args.device), x_sensor.to(args.device)

    _, x_dense = test_data.dense_eval(0)
    x_dense = x_dense.reshape(1, -1, 1).to(args.device)

    # Sample VAE reconstruction
    with torch.no_grad():
        mu, logvar = vae.encode(u_train)
        z = reparameterize(mu, logvar)
        # evaluate function
        u_hat, v_hat = vae.decode(z, x_dense).chunk(2, dim=2)
        u_hat, v_hat = u_hat.squeeze(2).cpu(), v_hat.squeeze(2).cpu()

    x_grid = x_dense[0]
    u_true = test_data.u_samples
    v_true = test_data.v_samples

    # optionally add plot names to save figures
    plot_name = None
    plot_loss(metrics, plot_name=plot_name, show_plot=SHOW_PLOTS)

    compare_samples(
        x_grid,
        u_true,
        v_true,
        u_hat,
        v_hat,
        x_sensor,
        samples_to_plot=NUM_SAMPLES_PLOT,
        plot_name=None,
    )
    compare_mean_std_vs_x(
        x_grid, u_true, u_hat, variable="u", plot_name=plot_name, show_plot=SHOW_PLOTS
    )
    compare_mean_std_vs_x(
        x_grid, v_true, v_hat, variable="u", plot_name=plot_name, show_plot=SHOW_PLOTS
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
    # compare histograms for v at x=pi
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
