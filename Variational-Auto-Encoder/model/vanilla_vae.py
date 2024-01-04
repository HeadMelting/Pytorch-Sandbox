import torch
from torch import nn
from torch.nn import functional as F

from .types_ import *
from .base import BaseVAE


class VanillaVAE(BaseVAE):

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 **kwargs) -> None:

        super().__init__()

        self.latent_dim = latent_dim

        # Encoder
        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=in_channels,
                              out_channels=h_dim,
                              kernel_size=3,
                              stride=2,
                              padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )

            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)

        # mean and var
        '''
        input img shape : [B, 3, 64, 64]
        after encode: [B, 512, 2, 2] -> flatten -> [B, 512 * 4]
        '''
        self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dim)

        # Decoder
        modules = []
        self.decoder_input = nn.Linear(
            latent_dim, hidden_dims[-1] * 4)  # [latent_dim, 512 * 4]

        hidden_dims.reverse()  # [512, 256, 128, 64, 32]

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_channels=hidden_dims[i],
                                       out_channels=hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels=hidden_dims[-1],
                               out_channels=hidden_dims[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1],
                      out_channels=3,
                      kernel_size=3,
                      padding=1),
            nn.Tanh())

    def encode(self, input: Tensor) -> List[Tensor]:
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, log_var: Tensor) -> Tensor:
        '''
        q(z|x_i) ~ N(mu_phi, var_phi)
        - using Monte-Carlo (M = 1);
        '''
        std = torch.exp(0.5 * log_var)
        # log_var이랑 동일한 shape으로 noise ~ N(0,1)에서 sampling -> 1번만 샘플링했음.
        eps = torch.randn_like(std)
        # 만약 여러번 샘플링 한다면
        '''
        (eps1 * std + mu) + (eps2 * std + mu) + (eps3 * std + mu) + (eps4 * std + mu) + ... + (epsM * std + mu) 
        = (eps1 + ... + epsM)/M * std + mu
        '''
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), input, mu, log_var]

    def loss_function(self, *args, **kwargs) -> dict:

        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs['M_N']

        recons_loss = F.mse_loss(recons, input)
        kld_loss = torch.mean(-0.5 * torch.sum(1 +
                              log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        loss = recons_loss + kld_weight * kld_loss

        return {'loss': loss, 'Reconstruction_Loss': recons_loss.detach(), 'KLD': -kld_loss.detach()}

    def sample(self, num_samples: int, current_device: torch.device, **kwargs) -> Tensor:
        z = torch.randn(num_samples, self.latent_dim)

        z = z.to(current_device)

        # since KL loss encouraged the encoder to map input x to latent distribution q(z|x) that closely approximates the Standard Normal Dist.
        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs):
        '''
        reconstruction.
        '''
        return self.forward(x)[0]
