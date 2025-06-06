from argparse import ArgumentParser, BooleanOptionalAction
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

from clfm.problems.gaussian_process import GPDataset
from clfm.nn.fully_connected_nets import FlowModel
from clfm.nn.vae import FunctionalVAE
from clfm.utils.latent_fm import sample_lfm
from gp_utils import get_result_and_model_path
from gp_plotting import (
    compare_samples,
    compare_covariance,
    compare_correlation,
    compare_mean_std,
)


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    NUM_SAMPLES = 1000
    NUM_SAMPLES_PLOT = 250
    SHOW_PLOTS = True

    result_dir = Path(args.result_dir)
    case_params = vars(args)
    result_path, model_path = get_result_and_model_path(case_params, result_dir)

    test_data = GPDataset(
        NUM_SAMPLES,
        case_params["num_sensors"],
        cov_len=case_params["gp_cov_len"],
        variance=case_params["gp_var"],
        mean="LINEAR",
    )
    test_loader = DataLoader(test_data, batch_size=len(test_data), shuffle=False)
    _, x_sensor = next(iter(test_loader))
    x_sensor = x_sensor.to(args.device)
    y_true, x_true = test_data.dense_eval(range(len(test_data)))

    # load vae, flow matching models & generate samples on x_grid
    vae = FunctionalVAE.load_from_checkpoint(model_path).to(args.device)
    fm_model_path = Path(result_path) / "fm.pth"
    checkpoint = torch.load(fm_model_path)
    flow_model = FlowModel(**checkpoint["architecture"])
    flow_model.load_state_dict(checkpoint["state_dict"])

    x_grid = torch.linspace(0.0, 1.0, 100).reshape(-1, 1)
    y_gen = sample_lfm(
        NUM_SAMPLES, x_grid, flow_model, vae, args.num_time_steps_fm, args.device
    )

    # optionally add plot names to save the figures
    plot_name = None
    compare_samples(
        x_true,
        y_true,
        x_grid,
        y_gen,
        x_sensor,
        NUM_SAMPLES_PLOT,
        plot_name=plot_name,
        show_plot=SHOW_PLOTS,
    )
    compare_covariance(
        x_grid,
        y_gen,
        vae.loss.variance,
        vae.loss.cov_len,
        plot_name=plot_name,
        show_plot=SHOW_PLOTS,
    )
    compare_correlation(
        x_grid,
        y_gen,
        vae.loss.variance,
        vae.loss.cov_len,
        plot_name=plot_name,
        show_plot=SHOW_PLOTS,
    )
    compare_mean_std(
        x_grid,
        y_gen,
        vae.loss.variance,
        "LINEAR",
        plot_name=plot_name,
        show_plot=SHOW_PLOTS,
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    # flow matching paramters:
    parser.add_argument("--num_time_steps_fm", type=int, default=100)
    # problem set up
    parser.add_argument("--N_data", type=int, default=1000)
    parser.add_argument("--gp_var", type=float, default=0.5)
    parser.add_argument("--gp_cov_len", type=float, default=0.1)
    parser.add_argument("--use_cov", action=BooleanOptionalAction, default=False)
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
    parser.add_argument("--num_sensors", type=int, default=5)
    parser.add_argument("--num_colloc", type=int, default=50)
    parser.add_argument("--res_weight", type=float, default=0.1)
    parser.add_argument("--kld_weight", type=float, default=1e-6)
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    # misc:
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--result_dir", type=str, default="results/gaussian_process")
    parser.add_argument("--version", type=int, default=0)
    main(parser.parse_args())
