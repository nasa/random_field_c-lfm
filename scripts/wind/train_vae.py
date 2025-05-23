from argparse import ArgumentParser, BooleanOptionalAction
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from clfm.problems.wind import WindDataset, WindLoss
from clfm.nn.unet1d import Encoder1d
from clfm.nn.vae import FunctionalVAE
from clfm.nn.fully_connected_nets import FCTrunk, EnhancedBranchNetwork
from wind_utils import get_logging_path_and_name

# NOTE: must generate train/test data using data/wind_data_generation/generate_wind_data.py to run this script
TRAIN_DATA_FILE = Path(__file__).parent / "../../data/wind/wind_train_data.hdf5"
TEST_DATA_FILE = Path(__file__).parent / "../../data/wind/wind_test_data.hdf5"


def main(args):
    torch.manual_seed(args.seed)

    train_data = WindDataset(TRAIN_DATA_FILE, args.sparse_sensors)
    validation_data = WindDataset(TEST_DATA_FILE)

    train_loader = DataLoader(
        train_data,
        batch_size=args.bs,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        validation_data,
        batch_size=args.bs,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    loss = WindLoss(args.num_colloc, args.T_colloc, train_data)

    encoder = Encoder1d(in_channels=train_data.num_sensors, latent_dim=args.latent_dim)
    branch = EnhancedBranchNetwork(
        latent_dim=args.latent_dim,
        hidden_dim=args.h_branch,
        output_dim=args.p_deeponet,
        num_blocks=args.nhl_branch,
    )
    trunk = FCTrunk(
        input_size=train_data.grid.ndim,
        hidden_size=args.h_trunk,
        output_size=args.p_deeponet,
        num_outputs=train_data.num_fields,
        num_hidden_layers=args.nhl_trunk,
    )
    vae = FunctionalVAE(
        encoder=encoder,
        branch=branch,
        trunk=trunk,
        num_fields=train_data.num_fields,
        grid=train_data.grid,
        lr=args.lr,
        res_weight=args.res_weight,
        kld_weight=args.kld_weight,
        loss=loss,
    )

    save_dir, name = get_logging_path_and_name(
        root_dir=args.result_dir,
        params_dict=vars(args),
    )
    logger = CSVLogger(save_dir=save_dir, name=name)
    checkpoint = ModelCheckpoint(monitor="total_loss", mode="min", save_top_k=1)

    # grad clipping turned off (=None) if grad_clip_val is neg (default=-1)
    gradient_clip_val = args.grad_clip if args.grad_clip > 0 else None

    trainer = L.Trainer(
        accelerator=args.accelerator,
        devices=args.num_devices,
        max_epochs=args.epochs,
        logger=logger,
        callbacks=[checkpoint],
        deterministic=True,
        gradient_clip_val=gradient_clip_val,
    )

    trainer.fit(vae, train_loader, val_loader)


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
    parser.add_argument("--accelerator", type=str, default="gpu")
    parser.add_argument("--result_dir", type=str, default="results/wind")
    parser.add_argument("--num_devices", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=16)
    main(parser.parse_args())
