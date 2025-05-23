from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, StochasticWeightAveraging

from fgm.problems.materials import MaterialsTrain, MaterialsVal, MaterialsLoss
from fgm.vae import FunctionalVAE

def main(args):
    train_data = MaterialsTrain()
    val_data = MaterialsVal(train_data.X_u)

    train_loader = DataLoader(train_data, batch_size = args.bs, shuffle = True, drop_last = True)
    val_loader = DataLoader(val_data, batch_size = args.bs, shuffle = True)
    loss = MaterialsLoss(train_data, num_colloc = args.num_colloc)

    vae = FunctionalVAE(
        branch_in_size = train_data.num_sensors, 
        trunk_in_size = train_data.grid.ndim,
        h_size = args.h_size, 
        interact_dim = args.h_size,
        fields = train_data.fields,
        grid = train_data.grid,
        lr = args.lr,
        res_weight = args.res_weight,
        kld_weight = args.kld_weight,
        emb_dim = args.emb_dim,
        theta = args.theta,
        loss = loss).cuda()
    
    name = f'epochs={args.epochs}-bs={args.bs}-lr={args.lr}-h_size={args.h_size}-res_weight={args.res_weight}-kld_weight={args.kld_weight}-emb_dim={args.emb_dim}-theta={args.theta}'
    logger = CSVLogger(save_dir = 'results/materials/vae', name = name)
    checkpoint = ModelCheckpoint(monitor = 'val_corr_loss', mode = 'min', save_top_k = 1)
    
    trainer = L.Trainer(
        accelerator = 'gpu',
        devices = 1,
        max_epochs = args.epochs,
        logger = logger,
        callbacks = [
            checkpoint,
            # StochasticWeightAveraging(swa_lrs=1e-2)
        ],
        gradient_clip_val = 0.5,
        check_val_every_n_epoch = 10
    )
    
    trainer.fit(vae, train_loader, val_loader)

    # u, x, f = next(iter(val_loader))
    # u, x, f = u.cuda(), x.cuda(), f.cuda()

    # with torch.no_grad():
    #     vae.loss.validate(vae, u, x, f)

    # ux, uy = u.chunk(2, dim = 2)
    # ux, uy = ux.squeeze(), uy.squeeze()
    # ux = ux.reshape(-1, 10, 9)
    # uy = uy.reshape(-1, 10, 9)

    # import matplotlib.pyplot as plt

    # plt.scatter(val_data.X[:, 0], val_data.X[:, 1])
    # plt.scatter(val_data.x_sensor[:, 0], val_data.x_sensor[:, 1])
    # plt.show()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--bs',             type = int,     default = 128)
    parser.add_argument('--lr',             type = float,   default = 0.001)
    parser.add_argument('--h_size',         type = int,     default = 128)
    parser.add_argument('--num_colloc',     type = int,     default = 500)
    parser.add_argument('--res_weight',     type = float,   default = 1e-6)
    parser.add_argument('--kld_weight',     type = float,   default = 5e-6)
    parser.add_argument('--epochs',         type = int,     default = 5000)
    parser.add_argument('--sensor_noise',   type = float,   default = 0.0)
    parser.add_argument('--emb_dim',        type = int,     default = 0)
    parser.add_argument('--theta',          type = float,   default = 0.0)
    main(parser.parse_args())

