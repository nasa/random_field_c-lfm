from argparse import ArgumentParser
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import numpy as np

from clfm.problems.materials import MaterialsTrain, MaterialsVal
from clfm.nn.vae import FunctionalVAE
from clfm.utils import reparameterize
from materials_utils import get_result_and_model_path
from scripts.materials.materials_plotting import (
    plot_train_loss,
    plot_val_loss,
    plot_correlation_comparison_from_vae,
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
    test_loader = DataLoader(test_data, batch_size=NUM_TEST_SAMPLES, shuffle=True)

    u, _, _ = next(iter(test_loader))
    u = u.to(args.device)
    e_test = test_data.E
    x_test = test_data.X
    x_gen = x_test.unsqueeze(0).repeat(u.shape[0], 1, 1).to(args.device)

    with torch.no_grad():
        mu, logvar = vae.encode(u)
        z = reparameterize(mu, logvar)
        ux_hat, uy_hat, e_hat = vae.decode(z, x_gen).chunk(3, dim=2)

    ux_hat = ux_hat.squeeze().reshape(-1, N_GRID, N_GRID).detach().cpu()
    uy_hat = uy_hat.squeeze().reshape(-1, N_GRID, N_GRID).detach().cpu()
    e_hat = e_hat.squeeze().reshape(-1, N_GRID, N_GRID).detach().cpu()
    e_test_reshape = e_test.reshape(-1, N_GRID, N_GRID).squeeze()

    # Plot convergence and results
    # optionally add plot_names to save the figures
    plot_name = None
    plot_train_loss(
        result_path / "metrics.csv", plot_name=plot_name, show_plot=SHOW_PLOTS
    )
    plot_val_loss(
        result_path / "metrics.csv", plot_name=plot_name, show_plot=SHOW_PLOTS
    )
    plot_samples(ux_hat, uy_hat, e_hat, plot_name=plot_name, show_plot=SHOW_PLOTS)
    plot_field_errors(e_test_reshape, e_hat, plot_name=plot_name, show_plot=SHOW_PLOTS)
    compare_pdfs(
        e_hat, e_test_reshape, x_test, N_GRID, plot_name=plot_name, show_plot=SHOW_PLOTS
    )
    plot_correlation_comparison_from_vae(
        vae, e_test, x_test, z, args.device, plot_name=plot_name, show_plot=SHOW_PLOTS
    )


if __name__ == "__main__":
    parser = ArgumentParser()

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
