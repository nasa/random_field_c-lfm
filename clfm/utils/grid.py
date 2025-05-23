import torch
from torch import nn, Tensor


class RectangularGrid(nn.Module):
    def __init__(self, low: Tensor, high: Tensor):
        super().__init__()

        if isinstance(low, float):
            low = torch.tensor([low])

        if isinstance(high, float):
            high = torch.tensor([high])

        self.register_buffer("low", low.float())
        self.register_buffer("high", high.float())

        assert self.low.shape == self.high.shape
        assert self.low.ndim == 1
        assert (self.low <= self.high).all()

    @property
    def ndim(self) -> int:
        return self.low.shape[0]

    def sample(self, N: int):
        """
        generates N uniformly sampled points between low and high.
        """
        epsilon = torch.rand(N, self.ndim, device=self.low.device)
        low = self.low.unsqueeze(0)
        high = self.high.unsqueeze(0)

        return epsilon * (high - low) + low

    def normalize(self, x: Tensor):
        """
        Scales collocation points to the unit cube in R^n.
        Good for normalizing input to a neural net.

        x: (batch_size x dim)
        """
        low = self.low.unsqueeze(0)
        high = self.high.unsqueeze(0)

        denom = high - low
        denom = (
            denom + (denom == 0).float()
        )  ## make sure that we divide by 1 and not zero.
        return (x - low) / denom

    def dense(self, N: int):
        """
        returns a (N**ndim) discretized grid over the domain.
        """
        x = torch.linspace(0.0, 1.0, N, device=self.low.device).unsqueeze(1)
        low = self.low.unsqueeze(0)
        high = self.high.unsqueeze(0)
        x = x * (high - low) + low
        return torch.cartesian_prod(
            *[_x.squeeze(1) for _x in x.chunk(self.ndim, dim=1)]
        )
