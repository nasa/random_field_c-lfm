import torch.nn as nn
from einops.layers.torch import Rearrange


class FCEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_hidden_layers):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_hidden_layers = num_hidden_layers

        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.GELU())

        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.GELU())

        layers.append(nn.Linear(hidden_size, output_size * 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """
        u is a tensor of size (batch_size, s1, ..., sn)
        num_sensors = s1 * ... * sn
        we flatten u so it is of size (batch_size, num_sensors)
        """
        x_flat = x.flatten(start_dim=1)
        return self.net(x_flat)


class FCTrunk(nn.Module):
    def __init__(
        self, input_size, hidden_size, output_size, num_outputs, num_hidden_layers
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_outputs = num_outputs
        self.num_hidden_layers = num_hidden_layers

        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.GELU())

        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.GELU())

        layers.append(nn.Linear(hidden_size, output_size * num_outputs))
        layers.append(nn.GELU())
        layers.append(Rearrange("b n (i f) -> b n i f", f=num_outputs))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class FCBranch(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_hidden_layers):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_hidden_layers = num_hidden_layers

        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.GELU())

        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.GELU())

        layers.append(nn.Linear(hidden_size, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    """
    Residual block for improved gradient flow in deeper networks.
    Can be used to enhance your VAE decoder's capacity.
    """

    def __init__(self, input_dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = input_dim

        # Main branch
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(input_dim),
            nn.Linear(hidden_dim, input_dim),
        )

        self.activation = nn.SiLU()
        # Optional projection if dimensions don't match
        self.proj = (
            nn.Identity()
            if input_dim == hidden_dim
            else nn.Linear(input_dim, hidden_dim)
        )

    def forward(self, x):
        # Residual connection: output = activation(x + F(x))
        return self.activation(x + self.net(x))


class EnhancedBranchNetwork(nn.Module):
    """
    Enhanced decoder branch network with residual connections
    for better gradient flow and increased expressivity.
    """

    def __init__(self, latent_dim, hidden_dim, output_dim, num_blocks=3):
        super().__init__()
        self.input_proj = nn.Linear(latent_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim) for _ in range(num_blocks)]
        )
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        # Project latent vector to working dimension
        h = self.input_proj(z)
        for block in self.blocks:
            h = block(h)
        return self.output_proj(h)


class FlowModel(nn.Module):
    def __init__(self, latent_dim, hidden_size, num_hidden_layers):
        super().__init__()

        self.latent_dim = latent_dim
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers

        layers = []
        layers.append(nn.Linear(latent_dim + 1, hidden_size))
        layers.append(nn.GELU())

        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.GELU())

        layers.append(nn.Linear(hidden_size, latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
