import torch
import torch.nn as nn
from typing import List, Union


class EDMPrecond(nn.Module):
    """
    Improved preconditioning proposed in the paper "Elucidating the Design Space of
    Diffusion-Based Generative Models" (EDM)
    Adopted from https://github.com/NVIDIA/physicsnemo/blob/main/physicsnemo/models/diffusion/preconditioning.py
    Parameters
    ----------
    net: the main network
    """

    def __init__(
        self,
        net: nn.Module,
        sigma_min=0.0,
        sigma_max=float("inf"),
        sigma_data=0.5,
    ):
        super().__init__()
        self.net = net
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data

    def forward(
        self,
        x,
        sigma,
        condition,
    ):
        sigma = sigma.reshape(-1, 1, 1, 1)

        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2).sqrt()
        c_in = 1 / (self.sigma_data**2 + sigma**2).sqrt()
        c_noise = sigma.log() / 4

        arg = c_in * x

        if condition is not None:
            arg = torch.cat([arg, condition], dim=1)

        F_x = self.net(arg, c_noise.flatten())
        
        D_x = c_skip * x + c_out * F_x
        return D_x

    @staticmethod
    def round_sigma(sigma: Union[float, List, torch.Tensor]):
        """
        Convert a given sigma value(s) to a tensor representation.

        Parameters
        ----------
        sigma : Union[float list, torch.Tensor]
            The sigma value(s) to convert.

        Returns
        -------
        torch.Tensor
            The tensor representation of the provided sigma value(s).
        """
        return torch.as_tensor(sigma)