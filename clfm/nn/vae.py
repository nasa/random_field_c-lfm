import torch
from torch import nn, Tensor
import lightning as L
from einops import einsum

from clfm.utils import kl_divergence, reparameterize


class FunctionalVAE(L.LightningModule):
    """
    Variational Autoencoder with a function decoder implemented with a DeepONet and a residual loss term to incorporate physical or statistical constraints.

    This model combines a traditional VAE architecture with DeepONet (Deep Operator Network) capabilities for decoding continuous functions from latent representations.

    The model consists of:
    - An encoder that maps input data to a latent space representation
    - A DeepONet-based decoder (branch and trunk networks) that reconstructs functions from latent representations
    - A residual loss component that enforces physical or statistical constraints

    Parameters
    ----------
    encoder : nn.Module
        Neural network module for encoding input data into latent space
    branch : nn.Module
        Branch network of the DeepONet that processes the latent representation
    trunk : nn.Module
        Trunk network of the DeepONet that processes the spatial/temporal coordinates
    num_fields : int
        Number of output fields/channels in the function representation
    grid : object
        Object with a normalize method to preprocess spatial/temporal coordinates
    lr : float
        Learning rate for the optimizer
    res_weight : float
        Weight factor for the residual loss component
    kld_weight : float
        Weight factor for the KL divergence loss component
    loss : object
        Loss object with methods for computing reconstruction loss, residual loss, and validation metrics

    Attributes
    ----------
    latent_dim : int
        Dimension of the latent space (derived from branch network input size)
    """

    def __init__(
        self,
        encoder,
        branch,
        trunk,
        num_fields,
        grid,
        lr,
        res_weight,
        kld_weight,
        loss,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.grid = grid
        self.lr = lr
        self.res_weight = res_weight
        self.kld_weight = kld_weight
        self.loss = loss
        self._encoder = encoder
        self._branch = branch
        self._trunk = trunk
        self._bias = nn.Parameter(torch.zeros(1, 1, num_fields))

    @property
    def latent_dim(self):
        return self._branch.input_size

    def encode(self, u: Tensor):
        return self._encoder(u).chunk(2, dim=1)  # [mu, logvar]

    def decode(self, z: Tensor, x: Tensor):
        """
        evaluate latent function representation z at points x with deeponet
        z: (batch x interact_dim)
        x: (batch x num_points x point_dim)
        """
        x = self.grid.normalize(x)
        x = self._trunk(x)
        z = self._branch(z)
        return einsum(z, x, "b i, b n i f -> b n f") + self._bias

    def forward(self, u: Tensor, x: Tensor) -> Tensor:
        return self.decode(reparameterize(*self.encode(u)), x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx: int):

        u, x = batch  # (sensor data, space/time coords where data is obtained)
        mu, logvar = self.encode(u)
        z = reparameterize(mu, logvar)  # sample latent function representation

        # evaluate reconstruction loss on function representation at sensor
        rec_loss = self.loss.reconstruction(self, z, x, u)
        # evaluate constraint residual on the function representation
        res_loss, metrics = self.loss.residual(self, z)
        # compute Div_KL( N(mu, logvar) || N(0, I) )
        kld_loss = kl_divergence(mu, logvar)

        loss = rec_loss + self.kld_weight * kld_loss + self.res_weight * res_loss

        self.log_dict(
            {
                "reconstruction": rec_loss,
                "kld": kld_loss,
                "total_loss": loss,
                **metrics,
            },
            prog_bar=True,
            sync_dist=True,
            on_epoch=True,
            on_step=False,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        metrics = self.loss.validate(self, *batch)
        self.log_dict(metrics)
