from typing import Optional
import torch
import torch.nn as nn

# Largely based on https://github.com/pdearena/pdearena/blob/main/pdearena/modules/twod_unet.py

def get_activation(activation: str):
    if activation == "gelu":
        return nn.GELU()
    elif activation == "relu":
        return nn.ReLU()
    elif activation == "silu":
        return nn.SiLU()
    elif activation == "leaky":
        return nn.LeakyReLU(0.3)
    else:
        raise NotImplementedError(f"Activation {activation} not implemented")

class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: str = "gelu",
        norm: bool = False,
        dropout: float = 0.0,
        n_groups: int = 1,
        diffusion_model=False,
        noise_embedding_channels=None,
    ):
        super().__init__()
        self.activation = get_activation(activation)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        # If the number of input channels is not equal to the number of output channels we have to
        # project the shortcut connection
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        else:
            self.shortcut = nn.Identity()

        if norm:
            self.norm1 = nn.GroupNorm(n_groups, in_channels)
            self.norm2 = nn.GroupNorm(n_groups, out_channels)
            # self.norm1 = nn.BatchNorm2d(in_channels)
            # self.norm2 = nn.BatchNorm2d(out_channels)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

        self.drop = nn.Dropout(dropout)
        
        if diffusion_model:
            self.noise_affine = nn.Linear(
                in_features=noise_embedding_channels,
                out_features=out_channels * 2,
            )

    def forward(self, x: torch.Tensor, noise_emb: Optional[torch.Tensor] = None):
        h = self.drop(self.conv1(self.activation(self.norm1(x))))
                
        if noise_emb is not None:
            params = self.noise_affine(noise_emb).unsqueeze(2).unsqueeze(3)
            scale, shift = params.chunk(chunks=2, dim=1)
            h = torch.addcmul(shift, self.norm2(h), scale + 1)
        else:
            h = self.norm2(h)
        
        h = self.drop(self.conv2(self.activation(h)))
        
        # Add the shortcut connection and return
        return h + self.shortcut(x)


class AttentionBlock(nn.Module):
    """Attention block This is similar to [transformer multi-head
    attention]

    Args:
        n_channels (int): the number of channels in the input
        n_heads (int): the number of heads in multi-head attention
        d_k: the number of dimensions in each head
        n_groups (int): the number of groups for [group normalization][torch.nn.GroupNorm].

    """

    def __init__(self, n_channels: int, n_heads: int = 1, d_k: Optional[int] = None, n_groups: int = 1):
        super().__init__()

        # Default `d_k`
        if d_k is None:
            d_k = n_channels
        # Normalization layer
        self.norm = nn.GroupNorm(n_groups, n_channels)
        # Projections for query, key and values
        self.projection = nn.Linear(n_channels, n_heads * d_k * 3)
        # Linear layer for final transformation
        self.output = nn.Linear(n_heads * d_k, n_channels)
        # Scale for dot-product attention
        self.scale = d_k**-0.5
        #
        self.n_heads = n_heads
        self.d_k = d_k

    def forward(self, x: torch.Tensor):
        # Get shape
        batch_size, n_channels, height, width = x.shape
        # Change `x` to shape `[batch_size, seq, n_channels]`
        x = x.view(batch_size, n_channels, -1).permute(0, 2, 1)
        # Get query, key, and values (concatenated) and shape it to `[batch_size, seq, n_heads, 3 * d_k]`
        qkv = self.projection(x).view(batch_size, -1, self.n_heads, 3 * self.d_k)
        # Split query, key, and values. Each of them will have shape `[batch_size, seq, n_heads, d_k]`
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        # Calculate scaled dot-product $\frac{Q K^\top}{\sqrt{d_k}}$
        attn = torch.einsum("bihd,bjhd->bijh", q, k) * self.scale
        # Softmax along the sequence dimension $\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_k}}\Bigg)$
        attn = attn.softmax(dim=1)
        # Multiply by values
        res = torch.einsum("bijh,bjhd->bihd", attn, v)
        # Reshape to `[batch_size, seq, n_heads * d_k]`
        res = res.view(batch_size, -1, self.n_heads * self.d_k)
        # Transform to `[batch_size, seq, n_channels]`
        res = self.output(res)

        # Add skip connection
        res += x

        # Change to shape `[batch_size, in_channels, height, width]`
        res = res.permute(0, 2, 1).view(batch_size, n_channels, height, width)
        return res


class DownBlock(nn.Module):
    """Down block This combines [`ResidualBlock`][pdearena.modules.twod_unet.ResidualBlock] and [`AttentionBlock`][pdearena.modules.twod_unet.AttentionBlock].

    These are used in the first half of U-Net at each resolution.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        has_attn (bool): Whether to use attention block
        activation (nn.Module): Activation function
        norm (bool): Whether to use normalization
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        has_attn: bool = False,
        activation: str = "gelu",
        norm: bool = False,
        dropout: float = 0.0,
        diffusion_model=False,
        noise_embedding_channels=None,
    ):
        super().__init__()
        self.res = ResidualBlock(
            in_channels,
            out_channels,
            activation=activation,
            norm=norm, dropout=dropout,
            diffusion_model=diffusion_model,
            noise_embedding_channels=noise_embedding_channels,
        )
        if has_attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor, noise_emb: Optional[torch.Tensor] = None):
        x = self.res(x, noise_emb=noise_emb)
        x = self.attn(x)
        return x


class UpBlock(nn.Module):
    """Up block that combines [`ResidualBlock`][pdearena.modules.twod_unet.ResidualBlock] and [`AttentionBlock`][pdearena.modules.twod_unet.AttentionBlock].

    These are used in the second half of U-Net at each resolution.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        has_attn (bool): Whether to use attention block
        activation (str): Activation function
        norm (bool): Whether to use normalization
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        has_attn: bool = False,
        activation: str = "gelu",
        norm: bool = False,
        dropout: float = 0.0,
        diffusion_model=False,
        noise_embedding_channels=None,
    ):
        super().__init__()
        # The input has `in_channels + out_channels` because we concatenate the output of the same resolution
        # from the first half of the U-Net
        self.res = ResidualBlock(
            in_channels + out_channels,
            out_channels,
            activation=activation,
            norm=norm,
            dropout=dropout,
            diffusion_model=diffusion_model,
            noise_embedding_channels=noise_embedding_channels,
        )
        if has_attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor, noise_emb: Optional[torch.Tensor] = None):
        x = self.res(x, noise_emb=noise_emb)
        x = self.attn(x)
        return x


class MiddleBlock(nn.Module):
    """Middle block

    It combines a `ResidualBlock`, `AttentionBlock`, followed by another
    `ResidualBlock`.

    This block is applied at the lowest resolution of the U-Net.

    Args:
        n_channels (int): Number of channels in the input and output.
        has_attn (bool, optional): Whether to use attention block. Defaults to False.
        activation (str): Activation function to use. Defaults to "gelu".
        norm (bool, optional): Whether to use normalization. Defaults to False.
    """

    def __init__(
        self,
        n_channels: int,
        has_attn: bool = False,
        activation: str = "gelu",
        norm: bool = False,
        dropout: float = 0.0,
        diffusion_model=False,
        noise_embedding_channels=None,
    ):
        super().__init__()
        self.res1 = ResidualBlock(
            n_channels,
            n_channels,
            activation=activation,
            norm=norm,
            dropout=dropout,
            diffusion_model=diffusion_model,
            noise_embedding_channels=noise_embedding_channels,
        )
        self.attn = AttentionBlock(n_channels) if has_attn else nn.Identity()
        self.res2 = ResidualBlock(
            n_channels,
            n_channels,
            activation=activation,
            norm=norm,
            dropout=dropout,
            diffusion_model=diffusion_model,
            noise_embedding_channels=noise_embedding_channels,
        )

    def forward(self, x: torch.Tensor, noise_emb: Optional[torch.Tensor] = None):
        x = self.res1(x, noise_emb=noise_emb)
        x = self.attn(x)
        x = self.res2(x, noise_emb=noise_emb)
        return x


class Upsample(nn.Module):
    r"""Scale up the feature map by $2 \times$

    Args:
        n_channels (int): Number of channels in the input and output.
    """

    def __init__(self, n_channels: int):
        super().__init__()
        self.conv = nn.ConvTranspose2d(n_channels, n_channels, (4, 4), (2, 2), (1, 1))

    def forward(self, x: torch.Tensor, noise_emb: Optional[torch.Tensor] = None):
        return self.conv(x)


class Downsample(nn.Module):
    r"""Scale down the feature map by $\frac{1}{2} \times$

    Args:
        n_channels (int): Number of channels in the input and output.
    """

    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.Conv2d(n_channels, n_channels, (3, 3), (2, 2), (1, 1))

    def forward(self, x: torch.Tensor, noise_emb: Optional[torch.Tensor] = None):
        return self.conv(x)


class PositionalEmbedding(torch.nn.Module):
    """
    A module for generating positional embeddings based on timesteps.
    This embedding technique is employed in the DDPM++ and ADM architectures.

    Parameters:
    -----------
    num_channels : int
        Number of channels for the embedding.
    max_positions : int, optional
        Maximum number of positions for the embeddings, by default 10000.
    endpoint : bool, optional
        If True, the embedding considers the endpoint. By default False.

    """

    def __init__(
        self, num_channels: int, max_positions: int = 10000, endpoint: bool = False
    ):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(
            start=0, end=self.num_channels // 2, dtype=torch.float32, device=x.device
        )
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x