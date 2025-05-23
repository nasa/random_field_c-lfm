import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchdyn.core import NeuralODE
from tqdm import tqdm

from clfm.nn.fully_connected_nets import FlowModel
from clfm.utils import reparameterize


class torch_wrapper(torch.nn.Module):
    """
    https://github.com/atong01/conditional-flow-matching/tree/main
    Wraps model to torchdyn compatible format.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, t, x, *args, **kwargs):
        t = t.repeat(x.shape[0]).unsqueeze(-1)
        return self.model(torch.cat([x, t], dim=-1))


def train_lfm(
    vae,
    train_data,
    latent_dim=64,
    hidden_layer_size=128,
    num_hidden_layers=3,
    num_epochs=1000,
    sigma_min=0.01,
    batch_size=256,
    lr=0.001,
    device="cuda",
    num_workers=0,
):
    """
    Trains a latent flow model (LFM) to learn the velocity field for generative modeling in the latent space of a pre-trained VAE.

    This function implements the training procedure where a neural network learns topredict the velocity field between random noise and encoded data points. The flow model is trained to predict the direction from noise to data samples in the latent space of the VAE.

    Parameters
    ----------
    vae : nn.Module
        Pre-trained variational autoencoder model used to encode training data into latent vectors
    train_data : Dataset
        PyTorch dataset containing training samples (pairs of data and coordinates)
    latent_dim : int, default=64
        Dimension of the latent space for both the VAE and flow model
    hidden_layer_size : int, default=128
        Size of hidden layers in the flow model
    num_hidden_layers : int, default=3
        Number of hidden layers in the flow model
    num_epochs : int, default=1000
        Number of training epochs
    sigma_min : float, default=0.01
        Minimum noise level added during the diffusion process
    batch_size : int, default=256
        Batch size for training
    lr : float, default=0.001
        Learning rate for the Adam optimizer
    device : str, default="gpu"
        Device to run the training on ("cpu" or "gpu")
    num_workers : int, default=0
        Number of workers for the data loader

    Returns
    -------
    flow : FlowModel
        The trained flow model
    loss_hist : list
        History of training losses (one value per epoch)

    Notes
    -----
    The training procedure follows a simplified diffusion model approach where:
    1. Data points are encoded into latent vectors z1 using the VAE encoder
    2. Random noise vectors z0 are generated from a standard normal distribution
    3. Interpolated points are created between z0 and z1 at random timesteps t
    4. The flow model learns to predict the velocity field (z1 - z0) at these points
    """

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    flow = FlowModel(
        latent_dim=latent_dim,
        hidden_size=hidden_layer_size,
        num_hidden_layers=num_hidden_layers,
    ).to(device)
    opt = torch.optim.Adam(flow.parameters(), lr=lr)

    loss_hist = []

    for _ in tqdm(range(num_epochs)):
        for batch in train_loader:
            opt.zero_grad()
            u, x = batch
            u, x = u.to(device), x.to(device)

            with torch.no_grad():
                z1 = reparameterize(*vae.encode(u))

            z0 = torch.randn_like(z1)
            t = torch.rand(z1.shape[0], device=device).unsqueeze(-1)
            mu_t = t * z1 + (1.0 - t) * z0
            zt = mu_t + sigma_min * torch.randn_like(mu_t)
            vt = flow(torch.cat([zt, t], dim=-1))
            loss = F.mse_loss(vt, z1 - z0)
            loss.backward()
            opt.step()
        loss_hist.append(loss.item())

    return flow, loss_hist


def sample_lfm(num_samples, x_grid, flow_model, vae, num_time_steps=100, device="cuda"):
    """
    Generates samples using latent flow matching by simulating the learned flow in latent space and decoding the results.

    This function performs generative sampling through a two-step process:
    1. Uses a Neural ODE to simulate the learned flow, transforming random noise into meaningful latent vectors
    2. Decodes these latent vectors into function outputs using the VAE decoder at specified coordinates

    Parameters
    ----------
    num_samples : int
        Number of samples to generate
    x_grid : Tensor
        Tensor of shape (num_points, point_dim) specifying the coordinates at which to evaluate
        the generated functions
    flow_model : nn.Module
        Trained flow model that predicts the velocity field in the latent space
    vae : nn.Module
        Trained variational autoencoder with a decode method that maps from latent space to function space
    num_time_steps : int, default=100
        Number of integration steps for the Neural ODE solver
    device : str, default="cuda"
        Device on which to perform the computation ("cpu", "cuda", "mps", ...)

    Returns
    -------
    y_gen : Tensor
        Generated function samples evaluated at the provided coordinates
        Shape: (num_samples, num_points, output_dim)

    Notes
    -----
    The sampling process implements continuous normalizing flows in the latent space:
    1. Starting from random Gaussian noise in the latent space
    2. Integrating the learned velocity field using a Neural ODE solver
    3. Obtaining samples from the data distribution in latent space
    4. Decoding these samples to obtain function values at specified coordinates
    """

    x_gen = x_grid[None, :, :].repeat(num_samples, 1, 1).to(device)

    node = NeuralODE(
        torch_wrapper(flow_model),
        solver="rk4",
        sensitivity="adjoint",
    ).to(device)

    with torch.no_grad():
        traj = node.trajectory(
            torch.randn(num_samples, flow_model.latent_dim, device=device),
            t_span=torch.linspace(0.0, 1.0, num_time_steps, device=device),
        )
    z = traj[-1, :, :]

    with torch.no_grad():
        y_gen = vae.decode(z, x_gen)

    return y_gen
