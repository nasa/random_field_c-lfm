import torch
from torch import Tensor
import glob

from clfm.utils.grid import RectangularGrid


def exists(x):
    return x is not None


def grad(y: Tensor, x: Tensor) -> Tensor:
    return torch.autograd.grad(
        y, x, grad_outputs=torch.ones_like(y), create_graph=True, allow_unused=True
    )[0]


def min_max_normalize(x, x_range: tuple):
    return (x - x_range[0]) / (x_range[1] - x_range[0])


def min_max_unnormalize(x, x_range: tuple):
    return x * (x_range[1] - x_range[0]) + x_range[0]


def kl_divergence(mu: Tensor, logvar: Tensor):
    """
    mu: (batch x dim)
    logvar: (batch x dim)
    """
    mu_flat, logvar_flat = mu.flatten(start_dim=1), logvar.flatten(start_dim=1)
    return torch.mean(
        -0.5 * torch.sum(1 + logvar_flat - mu_flat**2 - logvar_flat.exp(), dim=1), dim=0
    )


def reparameterize(mu: Tensor, logvar: Tensor):
    """
    applies reparameterization trick to sample from N(mu, sigma)
    """
    return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)


def get_epoch_and_step_for_checkpoint(result_dir):
    """
    get largest epoch/step to load latest model from training checkpoint
    """
    result_dir = result_dir / "checkpoints/*"
    check_points = glob.glob(str(result_dir))
    epoch = 0
    step = 0
    for chk_pt in check_points:
        epoch_new = int(chk_pt[chk_pt.rfind("epoch") + 6 : chk_pt.rfind("-")])
        step_new = int(chk_pt[chk_pt.rfind("step") + 5 : chk_pt.rfind(".ckpt")])
        if epoch_new > epoch:
            epoch = epoch_new
        if step_new > step:
            step = step_new
    return epoch, step


def dense_grid_eval(grid: RectangularGrid, discretization: tuple):
    """
    Returns a (N**ndim, ndim) discretized grid over the domain.
    """
    assert len(discretization) == grid.ndim

    # Create a tuple of ranges for each dimension
    ranges = [
        torch.linspace(0.0, 1.0, d, device=grid.low.device) for d in discretization
    ]
    # Generate the coordinate matrices
    coords = torch.meshgrid(*ranges)
    # Stack the coordinates and reshape to (N**ndim, ndim)
    x = torch.stack(coords, dim=-1).reshape(-1, grid.ndim)
    # Scale and shift to the grid's domain
    low = grid.low.unsqueeze(0)
    high = grid.high.unsqueeze(0)
    x = x * (high - low) + low
    return x
