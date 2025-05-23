from pathlib import Path
import h5py
import numpy as np
import torch
from torch import nn, Tensor
from torch.utils.data import Dataset

from clfm.utils.grid import RectangularGrid
from clfm.utils.utils import reparameterize, dense_grid_eval


# constants used for coherence evaluation:
# decay coefficients determine how fast coh. decays in each direction
C = [3.0, 3.0, 0.5]
# constants determining mean wind velocity
U_STAR = 1.90  # shear velocity
Z_0 = 0.015  # roughness length


def get_true_coherence(freq, x_coord_j, x_coord_k):
    """
    Evaluate prescribed coherence function for given frequency and spatial coordinates
    """
    mean_vel_j = 2.5 * U_STAR * np.log(x_coord_j[2] / Z_0)
    mean_vel_k = 2.5 * U_STAR * np.log(x_coord_k[2] / Z_0)
    numerator = freq * np.linalg.norm(np.multiply(C, x_coord_j - x_coord_k))
    denominator = np.abs(mean_vel_j) + np.abs(mean_vel_k)
    coherence = np.exp(-numerator / denominator)
    return coherence


def stft(signal: Tensor, nperseg: int, hop_length: int = None):
    """
    signal: Tensor of size (batch x time)
    """
    # parameters
    n_fft = nperseg
    window = torch.hann_window(nperseg).to(signal.device)
    if hop_length == None:
        noverlap = nperseg // 2
        hop_length = nperseg - noverlap

    # ## convert the signal into sliding windows
    segments = signal.unfold(1, nperseg, hop_length)
    # individually detrend each window
    segments = segments - segments.mean(dim=2, keepdim=True)
    # apply hann window
    segments = window[None, None, :] * segments
    # compute fft on each window
    w = torch.fft.fft(segments, n=n_fft)
    # the other half contains redundant information somehow
    return w[:, :, : nperseg // 2 + 1]
    # return w


def csd(signal1: Tensor, signal2: Tensor, nperseg: int, hop_length: int = None):
    """
    signal1: Tensor of size (batch x time)
    signal2: Tensor of size (batch x time)
    """

    stft1 = stft(signal1, nperseg, hop_length)
    stft2 = stft(signal2, nperseg, hop_length)
    # mean over number of samples
    result = torch.mean(stft1 * stft2.conj(), dim=0)
    # (windows x num_freq/2)
    return result


def coherence(
    signal1: Tensor, signal2: Tensor, nperseg: int = None, hop_length: int = None
):
    """
    signal1: Tensor of size (batch x time)
    signal2: Tensor of size (batch x time)
    """
    assert len(signal1.shape) == 2
    assert signal1.shape == signal2.shape

    if nperseg == None:
        nperseg = signal1.shape[-1]

    Pjj = csd(signal1, signal1, nperseg, hop_length)
    Pkk = csd(signal2, signal2, nperseg, hop_length)
    Pjk = csd(signal1, signal2, nperseg, hop_length)
    # Compute coherence
    coh = (torch.abs(Pjk) ** 2 / (Pjj * Pkk)).real
    # mean over number of windows
    return coh.mean(dim=0)


"""
Generate a vectorized verrsion of the coherence function to operate on batches inputs signals:
(batch x num_colloc x time) X (batch x num_colloc x time) --> (num_colloc x frequency)
"""
v_coherence = torch.vmap(coherence, in_dims=1)

# assumes a 10x10 spatial grid of data
DEFAULT_NUM_SENSORS = 100


class WindDataset(Dataset):
    def __init__(self, data_file, sparse_sensors=False, num_sensors=None):
        super().__init__()

        path = (Path(__file__).parent / data_file).resolve()
        self.f = h5py.File(path)
        """
        Hard coding lower and upper bound on v1 component for normalization
        """
        self.v1_min = -2.0
        self.v1_max = 47.0

        if num_sensors is None:
            num_sensors = DEFAULT_NUM_SENSORS
        # constructing grid
        # sparse sensors mimick sodar - vertical strips; skip every two columns of data (inds: 0-9, 30-39, 60-69, 90-99) - NOTE: this assumes data was generated on 10x10 spatial grid
        if sparse_sensors:
            self.sensor_idx = torch.cat(
                [torch.arange(start, start + 10) for start in range(0, 100, 30)]
            )
        else:
            self.sensor_idx = torch.arange(0, num_sensors)

        # extracting space and time grids from dataset
        x_grid = torch.tensor(np.array(self.f["x_grid"]))[self.sensor_idx]
        t_grid = torch.tensor(np.array(self.f["t_grid"]))

        low = torch.cat(
            [x_grid.min(dim=0).values, t_grid.min(dim=0, keepdim=True).values]
        )
        high = torch.cat(
            [x_grid.max(dim=0).values, t_grid.max(dim=0, keepdim=True).values]
        )

        # required member variables
        self.grid = RectangularGrid(low, high)
        self.num_fields = 1  # scalar velocity field
        self.num_sensors = len(self.sensor_idx)

    @property
    def dt(self):
        t = np.array(self.f["t_grid"])
        return t[1] - t[0]

    def __len__(self):
        return len(self.f["wind_samples"])

    def __getitem__(self, idx: int):
        u = torch.tensor(
            np.array(self.f["wind_samples"][str(idx)]["v1"]), dtype=torch.float
        )[self.sensor_idx, :]
        x = torch.tensor(np.array(self.f["x_grid"]), dtype=torch.float)[self.sensor_idx]
        t = torch.tensor(np.array(self.f["t_grid"]), dtype=torch.float)
        T = t.numel()
        # unsqueeze and repeat space grid along time dimention
        x = x[:, None, :].repeat(1, T, 1)
        # unsqueeze and repeat time grid along space dimention
        t = t[None, :, None].repeat(x.shape[0], 1, 1)
        # stack to form 4d points
        points = torch.cat([x, t], dim=2)
        # flatten space and time dim so the tensor is just: (points x point_dim)
        points = points.reshape(-1, 4)
        # normalize approximately between zero and one
        u_norm = (u - self.v1_min) / (self.v1_max - self.v1_min)
        return u_norm, points


class WindLoss(nn.Module):
    """
    Loss function for the wind problem.

    num_colloc: number of collocation points (space).
    T: number of evenly spaced time points to sample signals at.
    """

    def __init__(self, num_colloc: int, T: int, dataset: WindDataset):
        super().__init__()
        self.num_colloc = num_colloc
        self.T = T
        t = np.array(dataset.f["t_grid"])
        self.t_min = t.min()
        self.t_max = t.max()
        self.sensor_idx = dataset.sensor_idx

    def reconstruction(self, vae, z: Tensor, x: Tensor, u_true: Tensor):
        u_hat = vae.decode(z, x)
        u_hat = u_hat.reshape(*u_true.shape)
        return torch.mean(torch.square(u_hat - u_true))

    def residual(self, vae, z: Tensor):
        device = z.device
        batch_size = z.shape[0]
        num_colloc = self.num_colloc
        nperseg = self.T

        # randomly select coordinates in the spatial domain. Multiply: 2* because we want pairs.
        # strip off random dummy time component
        x_collocation = vae.grid.sample(2 * num_colloc)[:, :3]

        # x1 = (num_colloc x 3)
        # x2 = (num_colloc x 3)
        # get the pairs of collocation points
        x1, x2 = x_collocation.chunk(2, dim=0)

        # unfold into (batch x num_colloc x 1 x dim)
        x_collocation = x_collocation.reshape(1, 2 * num_colloc, 1, 3)
        """
        Repeat along batch dim because signals should be measured at the same collocation point acrosss many samples.
        Repeat along time dimention because we need to slap on the time label for each point.
        """
        x_collocation = x_collocation.repeat(
            batch_size, 1, self.T, 1
        )  # repeat points along batch and time series

        # a fixed dense time grid to use instead of random samples
        fixed_dense_t = torch.linspace(
            self.t_min, self.t_max, self.T, device=device
        ).reshape(1, 1, -1, 1)
        dt = (self.t_max - self.t_min) / self.T
        fixed_dense_t = fixed_dense_t.repeat(batch_size, 2 * num_colloc, 1, 1)
        collocation = torch.cat([x_collocation, fixed_dense_t], dim=3)

        # flattening collocation into (batch x points x dim)
        collocation = collocation.reshape(batch_size, -1, vae.grid.ndim)
        u_hat = vae.decode(z, collocation)
        u_hat = u_hat.reshape(batch_size, 2 * self.num_colloc, self.T)

        """
        We want two pairs of signals:
        (batch x num_colloc x T) and (batch x num_colloc x T)
        """
        u_hat_1, u_hat_2 = u_hat.chunk(2, dim=1)
        coh = v_coherence(u_hat_1, u_hat_2, nperseg=nperseg)
        freq_grid = torch.fft.rfftfreq(nperseg, d=dt)

        # compute the true coherence
        true_coh = []
        with torch.no_grad():
            for i in range(num_colloc):
                true_coh.append(
                    torch.tensor(
                        [
                            get_true_coherence(f, x1[i, :].cpu(), x2[i, :].cpu()) ** 2
                            for f in freq_grid
                        ]
                    )
                )

        # should result in a (num_colloc x frequency) tensor
        true_coh = torch.stack(true_coh).to(coh.device)
        residual = torch.mean(torch.square(true_coh - coh))
        return residual, {"residual": residual.item()}

    def validate(self, vae, u_data: Tensor, x_data: Tensor):

        # If sparse sensors are used for training, need to filter full u_data for encoding:
        u_data_sensors = u_data[:, self.sensor_idx, :]
        z = reparameterize(*vae.encode(u_data_sensors))

        val_disc = (1, 10, 10, 256)
        x = dense_grid_eval(vae.grid, val_disc)
        x = x[None, :, :].repeat(z.shape[0], 1, 1)

        u_hat = vae.decode(z, x)
        u_hat = u_hat.reshape(-1, val_disc[1] * val_disc[2], val_disc[3])

        u_gen_mean = torch.mean(u_hat, [0, 2])
        u_true_mean = torch.mean(u_data, [0, 2])
        u_gen_var = torch.var(u_hat, [0, 2])
        u_true_var = torch.var(u_data, [0, 2])

        mean_mse = torch.mean(torch.square(u_gen_mean - u_true_mean))
        var_mse = torch.mean(torch.square(u_gen_var - u_true_var))

        coh_mse, _ = self.residual(vae, z)

        return {
            "val_mean_mse": mean_mse.item(),
            "val_var_mse": var_mse.item(),
            "val_coherence_mse": coh_mse.item(),
        }
