import torch
from torch import nn, Tensor
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path

from clfm.utils.grid import RectangularGrid
from clfm.utils import reparameterize


def cov_func(x, cov_len, variance):
    """
    Evaluate true covariance function where x is a column vector - tensor of shape (n_x, 1)
    """
    dx = x - x.T
    return variance * torch.exp(-(dx**2) / (2 * cov_len**2))


class GPDataset(Dataset):

    def __init__(
        self,
        N: int = 1000,
        num_sensors: int = 100,
        x_min: float = 0.0,
        x_max: float = 1.0,
        discretization: int = 100,
        cov_len: float = 0.25,
        mean: float = 1.0,
        variance: float = 1.0,
        samples_dir: str = None,
    ):
        super().__init__()
        self.mean = mean
        self.variance = variance
        self.cov_len = cov_len

        # define grid on which to generate samples
        self.x_min = x_min
        self.x_max = x_max
        self.x = torch.linspace(x_min, x_max, discretization).reshape(-1, 1)
        self.sensor_indices = self._get_sensor_indices(num_sensors, discretization)

        if type(self.mean) == str and self.mean.upper() == "LINEAR":
            mean_func = self.x
        else:
            mean_func = self.mean * torch.ones(discretization, 1)

        if samples_dir is None:
            cov = cov_func(self.x, cov_len, self.variance)
            cov += 1e-4 * torch.eye(discretization)
            L = torch.tensor(np.linalg.cholesky(cov.numpy()))
            u = torch.randn(discretization, N)
            samples = mean_func + torch.matmul(L, u)
            self.samples = samples.T  # transpose to (N, n_x)
        else:
            self.samples = torch.load(Path(samples_dir) / "samples.pt")

        # required attributes
        self.grid = RectangularGrid(self.x_min, self.x_max)
        self.num_fields = 1
        self.num_sensors = num_sensors

    def __len__(self) -> int:
        return self.samples.shape[0]

    def __getitem__(self, idx: int):
        return (self.samples[idx, self.sensor_indices], self.x[self.sensor_indices, :])

    def _get_sensor_indices(self, num_sensors, discretization):
        if num_sensors == 1:
            sensor_indices = torch.tensor([round(discretization / 2)], dtype=torch.long)
        elif num_sensors == 2:
            sensor_indices = torch.tensor(
                [round(discretization / 3), round(2 * discretization / 3)],
                dtype=torch.long,
            )
        else:
            sensor_indices = torch.linspace(
                0, discretization - 1, num_sensors, dtype=torch.long
            )
        return sensor_indices

    def dense_eval(self, idx: int, n_x: int = None):
        return (self.samples[idx], self.x[:, :])

    def store_samples(self, output_dir):
        output_file = Path(output_dir) / "samples.pt"
        torch.save(self.samples, output_file)


class GPLoss(nn.Module):
    def __init__(
        self, train_data: GPDataset, num_colloc: int, use_covariance: bool = True
    ):
        super().__init__()
        self.cov_len = train_data.cov_len
        self.variance = train_data.variance
        self.mean = train_data.mean
        self.num_colloc = num_colloc
        self.use_covariance = use_covariance

    def reconstruction(self, vae, z: Tensor, x: Tensor, u_true: Tensor):
        u_pred = vae.decode(z, x)
        return torch.mean(torch.square(u_pred.squeeze() - u_true.squeeze()))

    def residual(self, vae, z: Tensor, x: Tensor = None):

        if x is None:
            x = vae.grid.sample(self.num_colloc).unsqueeze(0).repeat(z.shape[0], 1, 1)
        # assuming all x in the batch are on the same grid.
        sigma = cov_func(x[0, :, :], self.cov_len, self.variance)
        u_pred = vae.decode(z, x).squeeze()

        if self.use_covariance:
            # empirical covariance function of incoming u tensor
            sigma_hat = torch.cov(u_pred.T)
        else:
            # normalize the covariance function by variance to get correlation.
            sigma = sigma / self.variance
            # empirical correlation function of incoming u tensor
            sigma_hat = torch.corrcoef(u_pred.T)

        residual = torch.mean(torch.square(sigma - sigma_hat))
        return residual, {"residual": residual.item()}

    def validate(self, vae, u: Tensor, x_data: Tensor):
        NUM_VAL_PTS = 100
        z = reparameterize(*vae.encode(u))
        x = vae.grid.sample(NUM_VAL_PTS).unsqueeze(0).repeat(z.shape[0], 1, 1)
        y_pred = vae.decode(z, x).squeeze()
        # assuming all x in the batch are on the same grid.
        cov_true = cov_func(x[0, :, :], self.cov_len, self.variance)
        cov_gen = torch.cov(y_pred.T)

        if type(self.mean) == str and self.mean.upper() == "LINEAR":
            y_true_mean = x[0, :, :]
        else:
            y_true_mean = torch.ones(NUM_VAL_PTS, device=u.device) * self.mean

        y_gen_mean = torch.mean(y_pred, 0)
        y_true_var = torch.ones(NUM_VAL_PTS, device=u.device) * self.variance
        y_gen_var = torch.var(y_pred, 0)

        cov_mse = torch.mean(torch.square(cov_true - cov_gen))
        mean_mse = torch.mean(torch.square(y_true_mean - y_gen_mean))
        var_mse = torch.mean(torch.square(y_true_var - y_gen_var))
        val_loss = cov_mse + mean_mse + var_mse

        return {
            "val_cov_mse": cov_mse.item(),
            "val_mean_mse": mean_mse.item(),
            "val_var_mse": var_mse.item(),
            "val_loss": val_loss.item(),
        }
