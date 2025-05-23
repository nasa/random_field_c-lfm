from argparse import ArgumentParser
import torch
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from clfm.problems.materials import MaterialsTrain, MaterialsVal, MaterialsLoss
from clfm.nn.vae import FunctionalVAE
from clfm.nn.fully_connected_nets import FCEncoder, FCTrunk
from clfm.nn.fully_connected_nets import EnhancedBranchNetwork
from materials_utils import get_logging_path_and_name


def main(args):

    torch.manual_seed(args.seed)
    train_data = MaterialsTrain(args.N_data)
    val_data = MaterialsVal(train_data.X_u, num_samples=args.num_test_data)

    batch_size = args.N_data if args.N_data < args.bs else args.bs
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=128,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    loss = MaterialsLoss(
        train_data,
        num_colloc=args.num_colloc,
        lbc_weight=args.lbc_weight,
        rbc_weight=args.rbc_weight,
    )

    encoder = FCEncoder(
        input_size=train_data.num_sensors,
        hidden_size=args.h_encoder,
        output_size=args.latent_dim,
        num_hidden_layers=args.nhl_encoder,
    )
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

    gradient_clip_val = args.grad_clip if args.grad_clip > 0 else None
    trainer = L.Trainer(
        accelerator=args.accelerator,
        devices=args.num_devices,
        max_epochs=args.epochs,
        logger=logger,
        callbacks=[checkpoint],
        gradient_clip_val=gradient_clip_val,
    )

    trainer.fit(vae, train_loader, val_loader)
    return


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
    # misc / hpc:
    parser.add_argument("--accelerator", type=str, default="gpu")
    parser.add_argument("--result_dir", type=str, default="results/materials")
    parser.add_argument("--num_devices", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=16)
    main(parser.parse_args())
