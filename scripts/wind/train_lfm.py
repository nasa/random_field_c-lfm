from argparse import ArgumentParser, BooleanOptionalAction
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt

from clfm.problems.wind import WindDataset
from clfm.nn.vae import FunctionalVAE
from clfm.utils.latent_fm import train_lfm
from wind_utils import get_result_and_model_path

# NOTE: must generate train/test data using data/wind_data_generation/generate_wind_data.py to run this script
TRAIN_DATA_FILE = Path(__file__).parent / "../../data/wind/wind_train_data.hdf5"


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    result_dir = Path(args.result_dir)
    case_params = vars(args)
    result_path, model_path = get_result_and_model_path(case_params, result_dir)

    vae = FunctionalVAE.load_from_checkpoint(model_path).to(args.device)
    train_data = WindDataset(TRAIN_DATA_FILE, args.sparse_sensors)

    flow, loss = train_lfm(
        vae,
        train_data,
        case_params["latent_dim"],
        args.h_fm,
        args.nhl_fm,
        args.epochs_fm,
        args.sigma_min_fm,
        args.bs_fm,
        args.lr_fm,
        args.device,
        args.num_workers,
    )

    checkpoint = {
        "architecture": {
            "latent_dim": case_params["latent_dim"],
            "hidden_size": args.h_fm,
            "num_hidden_layers": args.nhl_fm,
        },
        "state_dict": flow.state_dict(),
    }
    torch.save(checkpoint, Path(result_path) / "fm.pth")

    plt.figure()
    plt.plot(loss)
    plt.xlabel("FM Epoch")
    plt.ylabel("FM Loss")
    plt.show()


if __name__ == "__main__":
    parser = ArgumentParser()
    # flow matching paramters:
    parser.add_argument("--epochs_fm", type=int, default=1000)
    parser.add_argument("--nhl_fm", type=int, default=3)  # num hidden layers
    parser.add_argument("--h_fm", type=int, default=128)  # hidden layer size
    parser.add_argument("--bs_fm", type=int, default=128)  # batch size
    parser.add_argument("--lr_fm", type=float, default=0.001)  # learning rate
    parser.add_argument("--sigma_min_fm", type=float, default=0.01)
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
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--version", type=int, default=0)
    main(parser.parse_args())
