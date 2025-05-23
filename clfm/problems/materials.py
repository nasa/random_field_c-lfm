from pathlib import Path
import h5py
import torch
from torch import nn, Tensor
from torch.utils.data import Dataset
import numpy as np
from scipy.interpolate import NearestNDInterpolator

from clfm.utils import grad, reparameterize
from clfm.utils.grid import RectangularGrid


class MaterialsTrain(Dataset):
    """
    Assumes the existance of a dataset located one directory outside of the clfm library in data/materials/
    """

    def __init__(
        self,
        num_samples: int = 1000,
        path: str = Path(__file__).parent / "../../data/materials/dataset_train.hdf5",
    ):
        super().__init__()
        if num_samples > 10000:
            raise ValueError("Only 1000 samples in dataset, choose num_samples <= 1000")

        with h5py.File(path, "r") as f:
            data = f["data"]
            self.X_f = torch.tensor(np.array(data["X_f"]))
            self.X_u = torch.tensor(np.array(data["X_u"]))
            self.bc = dict(data["boundary_conditions"])

            self.bc = {}
            for k, v in data["boundary_conditions"].items():
                self.bc[k] = torch.tensor(np.array(v))

            # data is always stored in vectors of length 180. flattened [ux, uy] fields stacked.
            self.snapshots = torch.tensor(np.array(data["snapshots"])).reshape(
                -1, 90, 2
            )[:num_samples, :, :]

        # required attributes
        self.grid = RectangularGrid(torch.zeros(2), torch.ones(2))
        self.num_fields = 3  # ux, uy, E
        self.num_sensors = self.X_u.numel()

    def __len__(self):
        return self.snapshots.shape[0]

    def __getitem__(self, idx: int):
        return self.snapshots[idx, :], self.X_u


class MaterialsVal(Dataset):
    """
    Assumes the existance of a dataset located one directory outside of the clfm library in data/materials/
    """

    def __init__(
        self,
        x_sensor: Tensor,
        num_samples: int = 10000,
        path: str = Path(__file__).parent / "../../data/materials/dataset_test.hdf5",
    ):
        super().__init__()

        if num_samples > 10000:
            raise ValueError(
                "Only 10000 samples in dataset, choose num_samples <= 10000"
            )

        with h5py.File(path, "r") as f:
            data = f["testing"]
            E = torch.tensor(np.array(data["E_test"]))[:num_samples, :]
            X = torch.tensor(np.array(data["X_test"]))
            u = torch.tensor(np.array(data["u_test"]))[:num_samples, ::]

        self.E = E
        self.X = X
        self.u = u
        self.x_sensor = x_sensor
        self.grid = RectangularGrid(torch.zeros(2), torch.ones(2))
        self.num_fields = 3  # ux, uy, E
        self.num_sensors = x_sensor.numel()

    def __getitem__(self, idx):
        interp = NearestNDInterpolator(self.X, self.u[idx, :, :])

        u_sensor = torch.tensor(interp(self.x_sensor), dtype=torch.float)
        return (
            u_sensor,
            self.X,
            torch.cat([self.u[idx, :, :], self.E[idx, :, None]], dim=1),
        )

    def __len__(self):
        return self.u.shape[0]


class MaterialsLoss(nn.Module):
    """
    Residual for the 2d elasticity problem.
    f: (batch, points, fields)
    colloc: (batch, points, dimention)
    """

    def __init__(
        self,
        data: MaterialsTrain,
        num_colloc: int,
        sigma: float = 1.5,
        nu: float = 0.3,
        lbc_weight: float = 1.0,
        rbc_weight: float = 1.0,
    ):
        super().__init__()

        Gamma1 = data.bc["X_ux_bc"].unsqueeze(0)
        Gamma2 = data.bc["X_sigma_bc"].unsqueeze(0)

        self.register_buffer("Gamma1", Gamma1)
        self.register_buffer("Gamma2", Gamma2)
        self.register_buffer("origin", torch.zeros(1, 1, 2))

        self.num_colloc = num_colloc
        self.sigma = sigma
        self.nu = nu
        self.lbc_weight = lbc_weight
        self.rbc_weight = rbc_weight

    def reconstruction(self, vae, z: Tensor, x: Tensor, u_true: Tensor):
        f_pred = vae.decode(z, x)
        u_pred = f_pred[:, :, :2]  # strip off ux and uy fields from e field.
        return torch.mean(torch.square(u_pred - u_true))

    def residual(self, vae, z: Tensor, colloc: Tensor = None):

        if colloc is None:
            colloc = vae.grid.sample(z.shape[0] * self.num_colloc)
            colloc = colloc.reshape(z.shape[0], self.num_colloc, -1)
            colloc.requires_grad_()

        ux, uy, e = vae.decode(z, colloc).chunk(3, dim=2)

        # Compute all required partial derivatives:
        e_x, e_y = grad(e, colloc).chunk(2, dim=2)

        ux_x, ux_y = grad(ux, colloc).chunk(2, dim=2)
        uy_x, uy_y = grad(uy, colloc).chunk(2, dim=2)

        ux_xx, ux_xy = grad(ux_x, colloc).chunk(2, dim=2)
        uy_yx, uy_yy = grad(uy_y, colloc).chunk(2, dim=2)

        ux_yx, ux_yy = grad(ux_y, colloc).chunk(2, dim=2)
        uy_xx, uy_xy = grad(uy_x, colloc).chunk(2, dim=2)

        """
        Calculate PDE residual MSE at collocation points; follows equations from PIGANs elasticity paper (https://arxiv.org/abs/2006.05791)
        """
        # Equation 26
        r1 = (
            e_x * (ux_x + self.nu * uy_y)
            + e * (ux_xx + self.nu * uy_yx)
            + (1.0 - self.nu) / 2.0 * ((e_y * (ux_y + uy_x)) + (e * (ux_yy + uy_xy)))
        )
        # Equation 27
        r2 = (
            (1.0 - self.nu) / 2.0 * ((e_x * (ux_y + uy_x)) + (e * (ux_yx + uy_xx)))
            + e_y * (self.nu * ux_x + uy_y)
            + e * (self.nu * ux_xy + uy_yy)
        )
        R = torch.mean(torch.square(r1)) + torch.mean(torch.square(r2))

        """
        Computes the boundary residual on a differentiable callable function.
        Function should take in points as a 3d tensor: (batch_size, num_points, dim_point).
        func should output a tensor with 3 fields: (batch_size, num_points, 3)
        """

        device = vae.loss.Gamma1.device
        batch_size = z.shape[0]
        num_boundary = 20

        # left boundary loss
        Gamma1 = torch.cat(
            [
                torch.zeros(1, num_boundary, 1, device=device),
                torch.rand(1, num_boundary, 1, device=device),
            ],
            dim=2,
        )  # random sampled Gamma1
        Gamma1 = Gamma1.repeat(batch_size, 1, 1)
        origin = self.origin.repeat(batch_size, 1, 1)

        # z = (batch x interact_dim)
        # collocation x = (batch x num_points x dim point)
        # Gamma1 = (1 x num_boundary x dim_point)
        # (batch x num_boundary x 3)

        ux_Gamma1, _, _ = vae.decode(z, Gamma1).chunk(3, dim=2)
        _, uy_origin, _ = vae.decode(z, origin).chunk(3, dim=2)
        left_boundary_residual = torch.mean(torch.square(ux_Gamma1)) + torch.mean(
            torch.square(uy_origin)
        )

        # right boundary loss
        Gamma2 = torch.cat(
            [
                torch.ones(1, num_boundary, 1, device=device),
                torch.rand(1, num_boundary, 1, device=device),
            ],
            dim=2,
        )  # random sampled Gamma2

        Gamma2 = Gamma2.repeat(batch_size, 1, 1)
        Gamma2.requires_grad_()

        ux_Gamma2, uy_Gamma2, e_Gamma2 = vae.decode(z, Gamma2).chunk(3, dim=2)
        ux_x_Gamma2, ux_y_Gamma2 = grad(ux_Gamma2, Gamma2).chunk(2, dim=2)
        uy_x_Gamma2, uy_y_Gamma2 = grad(uy_Gamma2, Gamma2).chunk(2, dim=2)
        sigma_xx = (
            (1.0 / (1.0 - self.nu**2))
            * e_Gamma2
            * (ux_x_Gamma2 + self.nu * uy_y_Gamma2)
        )
        sigma_xy = (
            (1.0 / (1.0 + self.nu)) * e_Gamma2 * 0.5 * (ux_y_Gamma2 + uy_x_Gamma2)
        )
        right_boundary_residual = torch.mean(
            torch.square(sigma_xx - self.sigma)
        ) + torch.mean(torch.square(sigma_xy))

        B = (
            self.lbc_weight * left_boundary_residual
            + self.rbc_weight * right_boundary_residual
        )

        return R + B, {"residual": R.item(), "boundary": B.item()}

    def validate(self, vae, u: Tensor, x: Tensor, f: Tensor):

        z = reparameterize(*vae.encode(u))
        ux_pred, uy_pred, e_pred = vae.decode(z, x).chunk(3, dim=2)
        ux_true, uy_true, e_true = f.chunk(3, dim=2)

        val_loss_observable = torch.mean(torch.square(ux_pred - ux_true)) + torch.mean(
            torch.square(uy_pred - uy_true)
        )
        val_loss_unobservable = torch.mean(torch.square(e_pred - e_true))

        # # hardcoding three lines on the y axis to evaluate correlation along.
        y_line_A = 0.75
        y_line_B = 0.50
        y_line_C = 0.25

        _, _, e_true = f.chunk(3, dim=2)
        e_true = e_true.squeeze()

        e_true_corr_A = torch.corrcoef(e_true[:, (x[0, :, 1] == y_line_A)].T)
        e_true_corr_B = torch.corrcoef(e_true[:, (x[0, :, 1] == y_line_B)].T)
        e_true_corr_C = torch.corrcoef(e_true[:, (x[0, :, 1] == y_line_C)].T)

        # # constructing a band of points at y = 0.75, 0.50, 0.25
        A_points = (
            torch.stack(
                [
                    torch.linspace(0.0, 1.0, 25),
                    torch.full((25,), y_line_A),
                ],
                dim=1,
            )
            .unsqueeze(0)
            .repeat(x.shape[0], 1, 1)
            .to(x.device)
        )

        B_points = (
            torch.stack(
                [torch.linspace(0.0, 1.0, 25), torch.full((25,), y_line_B)], dim=1
            )
            .unsqueeze(0)
            .repeat(x.shape[0], 1, 1)
            .to(x.device)
        )

        C_points = (
            torch.stack(
                [torch.linspace(0.0, 1.0, 25), torch.full((25,), y_line_C)], dim=1
            )
            .unsqueeze(0)
            .repeat(x.shape[0], 1, 1)
            .to(x.device)
        )

        # z = reparameterize(*vae.encode(u))
        _, _, e_pred_A = vae.decode(z, A_points).chunk(3, dim=2)
        _, _, e_pred_B = vae.decode(z, B_points).chunk(3, dim=2)
        _, _, e_pred_C = vae.decode(z, C_points).chunk(3, dim=2)

        e_pred_A = e_pred_A.squeeze()
        e_pred_corr_A = torch.corrcoef(e_pred_A.T)

        e_pred_B = e_pred_B.squeeze()
        e_pred_corr_B = torch.corrcoef(e_pred_B.T)

        e_pred_C = e_pred_C.squeeze()
        e_pred_corr_C = torch.corrcoef(e_pred_C.T)

        A_error = torch.mean(torch.square(e_pred_corr_A - e_true_corr_A))
        B_error = torch.mean(torch.square(e_pred_corr_B - e_true_corr_B))
        C_error = torch.mean(torch.square(e_pred_corr_C - e_true_corr_C))
        val_corr_loss = A_error + B_error + C_error

        val_loss = val_loss_observable + val_loss_unobservable + val_corr_loss

        return {
            "val_loss_observable": val_loss_observable.item(),
            "val_loss_unobservable": val_loss_unobservable.item(),
            "val_loss": val_loss.item(),
            "val_corr_loss": val_corr_loss.item(),
            "val_loss": val_loss.item(),
        }
