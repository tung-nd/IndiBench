from typing import Iterable, List
import torch
from torch import nn
from india_benchmark.models.networks.cnn_blocks import (
    DownBlock, Downsample, MiddleBlock, UpBlock, Upsample, PositionalEmbedding, get_activation
)

# Largely based on https://github.com/pdearena/pdearena/blob/main/pdearena/modules/twod_unet.py

class Unet(nn.Module):
    def __init__(
        self,
        variables: List[str],
        n_input_steps: int,
        hidden_channels=64,
        activation="gelu",
        norm: bool = True,
        dropout: float = 0.0,
        ch_mults: Iterable[int] = (1, 2, 2, 4),
        is_attn: Iterable[bool] = (False, False, False, False),
        mid_attn: bool = False,
        n_blocks: int = 2,
        diffusion_model: bool = False,
        noise_embedding_channels: int = 128,
    ) -> None:
        super().__init__()
        self.in_channels = n_input_steps * len(variables)
        self.out_channels = len(variables)
        self.hidden_channels = hidden_channels
        self.diffusion_model = diffusion_model
        
        if diffusion_model:
            self.map_noise = PositionalEmbedding(num_channels=hidden_channels, endpoint=True)
            self.map_layer = nn.Sequential(
                nn.Linear(hidden_channels, noise_embedding_channels),
                get_activation(activation),
                nn.Linear(noise_embedding_channels, noise_embedding_channels),
            )

        self.activation = get_activation(activation)
        self.image_proj = nn.Conv2d(self.in_channels, self.hidden_channels, kernel_size=3, padding=1)

        # #### First half of U-Net - decreasing resolution
        down = []
        # Number of channels
        out_channels = in_channels = self.hidden_channels
        # For each resolution
        n_resolutions = len(ch_mults)
        for i in range(n_resolutions):
            # Number of output channels at this resolution
            out_channels = in_channels * ch_mults[i]
            # Add `n_blocks`
            for _ in range(n_blocks):
                down.append(
                    DownBlock(
                        in_channels,
                        out_channels,
                        has_attn=is_attn[i],
                        activation=activation,
                        norm=norm,
                        dropout=dropout,
                        diffusion_model=diffusion_model,
                        noise_embedding_channels=noise_embedding_channels,
                    )
                )
                in_channels = out_channels
            # Down sample at all resolutions except the last
            if i < n_resolutions - 1:
                down.append(Downsample(in_channels))

        # Combine the set of modules
        self.down = nn.ModuleList(down)

        # Middle block
        self.middle = MiddleBlock(
            out_channels,
            has_attn=mid_attn,
            activation=activation,
            norm=norm,
            dropout=dropout,
            diffusion_model=diffusion_model,
            noise_embedding_channels=noise_embedding_channels,
        )

        # #### Second half of U-Net - increasing resolution
        up = []
        # Number of channels
        in_channels = out_channels
        # For each resolution
        for i in reversed(range(n_resolutions)):
            # `n_blocks` at the same resolution
            out_channels = in_channels
            for _ in range(n_blocks):
                up.append(
                    UpBlock(
                        in_channels,
                        out_channels,
                        has_attn=is_attn[i],
                        activation=activation,
                        norm=norm,
                        dropout=dropout,
                        diffusion_model=diffusion_model,
                        noise_embedding_channels=noise_embedding_channels,
                    )
                )
            # Final block to reduce the number of channels
            out_channels = in_channels // ch_mults[i]
            up.append(
                UpBlock(
                    in_channels,
                    out_channels,
                    has_attn=is_attn[i],
                    activation=activation,
                    norm=norm,
                    dropout=dropout,
                    diffusion_model=diffusion_model,
                    noise_embedding_channels=noise_embedding_channels,
                )
            )
            in_channels = out_channels
            # Up sample at all resolutions except last
            if i > 0:
                up.append(Upsample(in_channels))

        # Combine the set of modules
        self.up = nn.ModuleList(up)

        if norm:
            self.norm = nn.GroupNorm(8, hidden_channels)
        else:
            self.norm = nn.Identity()
        self.final = nn.Conv2d(in_channels, self.out_channels, kernel_size=3, padding=1)

    def forward(self, x, noise_labels=None):
        if len(x.shape) == 5:  # x.shape = [B,T,C,H,W]
            x = x.flatten(1, 2)
        
        if noise_labels is not None:
            noise_emb = self.map_noise(noise_labels)
            noise_emb = (
                noise_emb.reshape(noise_emb.shape[0], 2, -1).flip(1).reshape(*noise_emb.shape)
            )  # swap sin/cos
            noise_emb = self.map_layer(noise_emb)
        else:
            noise_emb = None
        
        # x.shape = [B,T*C,H,W]
        x = self.image_proj(x)
        h = [x]
        for m in self.down:
            x = m(x, noise_emb)
            h.append(x)
        x = self.middle(x, noise_emb)
        for m in self.up:
            if isinstance(m, Upsample):
                x = m(x, noise_emb)
            else:
                # Get the skip connection from first half of U-Net and concatenate
                s = h.pop()
                x = torch.cat((x, s), dim=1)
                x = m(x, noise_emb)
        yhat = self.final(self.activation(self.norm(x)))
        return yhat

# model = Unet(
#     3, 3, 1, 128, ch_mults=(1, 2, 2, 4)
# ).cuda()
# x = torch.randn(1, 3, 128, 256).cuda()
# y = model(x)
# print (y.shape)