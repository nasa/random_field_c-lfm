import torch
from torch import nn, Tensor
from torch.utils.data import Dataset
from torch.distributions import Normal
from scipy.integrate import solve_bvp
import numpy as np
from pathlib import Path

from clfm.utils.utils import grad
from clfm.utils.grid import RectangularGrid


def v_func(x, A):
    return 1 + A * x**2


def dv_dx_func(x, A):
    return 2 * A * x


def f_func(x):
    x = torch.tensor(x)
    return torch.sin(x)


def ode_system(x, y, A):
    _, v = y
    du_dx = v
    dv_dx = -f_func(x) / v_func(x, A) - (dv_dx_func(x, A) / v_func(x, A)) * v
    return np.vstack((du_dx, dv_dx))


def bc(ya, yb):
    return np.array([ya[0], yb[0]])  # u(0) = 0, u(pi) = 0


def compute_u(x_grid, A):
    u_sols = np.zeros((len(A), len(x_grid)), dtype=np.float32)
    u_init = np.zeros((2, len(x_grid)))
    for i, A_sample in enumerate(A):
        sol = solve_bvp(lambda x, u: ode_system(x, u, A_sample), bc, x_grid, u_init)
        u_sols[i, :] = sol.y[0]
    return u_sols


class Poisson1DDataset(Dataset):

    def __init__(
        self,
        N: int = None,
        num_sensors: int = None,
        x_min: float = 0.0,
        x_max: float = torch.pi,
        A_mean: float = 0.2,
        A_std: float = 0.05,
        samples_dir: str = None,
    ):

        super().__init__()
        self.x_min = x_min
        self.x_max = x_max
        self.grid = RectangularGrid(x_min, x_max)
        # randomized parameter of spatialy varying coefficient function
        dist = Normal(torch.tensor(A_mean), torch.tensor(A_std))

        # define grid of sensors over domain.
        if samples_dir is None:
            A = dist.sample((N,))
            self.x_sensor = self.grid.dense(num_sensors).reshape(-1, 1)
            self.v_samples = torch.tensor(v_func(self.x_sensor, A)).T
            self.u_samples = torch.tensor(compute_u(self.x_sensor[:, 0], A))
        else:
            self.v_samples = torch.load(Path(samples_dir) / "v_samples.pt")
            self.u_samples = torch.load(Path(samples_dir) / "u_samples.pt")
            N, num_sensors = self.v_samples.shape
            self.x_sensor = self.grid.dense(num_sensors).reshape(-1, 1)

        # required attributes
        self.num_fields = 2
        self.N = N
        self.num_sensors = num_sensors

    def __len__(self):
        return self.N

    def __getitem__(self, idx: int):
        sample = (self.u_samples[idx], self.x_sensor)
        return sample

    def dense_eval(self, idx: int, num_sensors: int = 100):
        """
        for this particular problem we can get a full function evaluation to test our model against.
        """
        x = torch.linspace(self.x_min, self.x_max, num_sensors).reshape(-1, 1)
        return self.u_samples[idx], x

    def store_samples(self, output_dir):
        torch.save(self.v_samples, Path(output_dir) / "v_samples.pt")
        torch.save(self.u_samples, Path(output_dir) / "u_samples.pt")


class Poisson1DLoss(nn.Module):
    def __init__(self, num_colloc: int):
        super().__init__()
        self.num_colloc = num_colloc

    def reconstruction(self, vae, z: Tensor, x: Tensor, u_true: Tensor):
        f = vae.decode(z, x)
        u, _ = f.chunk(2, dim=2)
        # squeeze off the singleton dim: (batch_size x num_points x 1)
        u = u.squeeze(2)
        return torch.mean(torch.square(u - u_true))

    def residual(self, vae, z: Tensor, x: Tensor = None):
        """
        if x is not none it is expected to be of size: (batch_size x num_points x ndim_point)
        """

        if x is None:
            x = vae.grid.sample(z.shape[0] * self.num_colloc)
            x = x.reshape(z.shape[0], self.num_colloc, -1)

        x.requires_grad_()
        f_target = f_func(x)
        u_v_gen = vae.decode(z, x)
        u, v = u_v_gen.chunk(2, dim=2)
        du_dx = grad(u, x)
        v_du_dx = v * du_dx
        d_v_du_dx_dx = grad(v_du_dx, x)

        residual = torch.mean(torch.square(d_v_du_dx_dx + f_target))
        return residual, {"residual": residual.item()}

    def validate(self, vae, *args):
        return {}
