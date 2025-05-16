"""Loss functions used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models".
adopted from https://github.com/NVIDIA/physicsnemo/blob/main/physicsnemo/metrics/diffusion/loss.py"""

import torch

class EDMLoss:
    """
    Loss function proposed in the EDM paper.

    Parameters
    ----------
    P_mean: float, optional
        Mean value for `sigma` computation, by default -1.2.
    P_std: float, optional:
        Standard deviation for `sigma` computation, by default 1.2.
    sigma_data: float, optional
        Standard deviation for data, by default 0.5.

    Note
    ----
    Reference: Karras, T., Aittala, M., Aila, T. and Laine, S., 2022. Elucidating the
    design space of diffusion-based generative models. Advances in Neural Information
    Processing Systems, 35, pp.26565-26577.
    """

    def __init__(
        self, P_mean: float = -1.2, P_std: float = 1.2, sigma_data: float = 0.5
    ):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, x, condition):
        """
        Calculate and return the loss corresponding to the EDM formulation.

        The method adds random noise to the input images and calculates the loss as the
        square difference between the network's predictions and the input images.
        The noise level is determined by 'sigma', which is computed as a function of
        'P_mean' and 'P_std' random values. The calculated loss is weighted as a
        function of 'sigma' and 'sigma_data'.

        Parameters:
        ----------
        net: torch.nn.Module
            The neural network model that will make predictions.

        x: torch.Tensor
            Input to the neural network.

        Returns:
        -------
        torch.Tensor
            A tensor representing the loss calculated based on the network's
            predictions.
        """
        rnd_normal = torch.randn([x.shape[0], 1, 1, 1], device=x.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2
        eps = torch.randn_like(x) * sigma
        D_yn = net(
            x + eps,
            sigma,
            condition=condition,
        )
        loss = weight * ((D_yn - x) ** 2)
        return loss
