"""
Adopted from https://github.com/NVIDIA/physicsnemo/tree/main/examples/generative/stormcast
"""

import torch
import torch.nn as nn
import numpy as np
from torch.nn.functional import silu
from typing import List, Union
from india_benchmark.models.networks.stormcast_layers import (
    Conv2d,
    FourierEmbedding,
    GroupNorm,
    Linear,
    PositionalEmbedding,
    UNetBlock,
)


class SongUNet(nn.Module):
    """
    Reimplementation of the DDPM++ and NCSN++ architectures, U-Net variants with
    optional self-attention, embeddings, and encoder-decoder components.

    This model supports conditional and unconditional setups, as well as several
    options for various internal architectural choices such as encoder and decoder
    type, embedding type, etc., making it flexible and adaptable to different tasks
    and configurations.

    Parameters
    -----------
    img_resolution : Union[List[int], int]
        The resolution of the input/output image, 1 value represents a square image.
    in_channels : int
        Number of channels in the input image.
    out_channels : int
        Number of channels in the output image.
    label_dim : int, optional
        Number of class labels; 0 indicates an unconditional model. By default 0.
    augment_dim : int, optional
        Dimensionality of augmentation labels; 0 means no augmentation. By default 0.
    model_channels : int, optional
        Base multiplier for the number of channels across the network, by default 128.
    channel_mult : List[int], optional
        Per-resolution multipliers for the number of channels. By default [1,2,2,2].
    channel_mult_emb : int, optional
        Multiplier for the dimensionality of the embedding vector. By default 4.
    num_blocks : int, optional
        Number of residual blocks per resolution. By default 4.
    attn_resolutions : List[int], optional
        Resolutions at which self-attention layers are applied. By default [16].
    dropout : float, optional
        Dropout probability applied to intermediate activations. By default 0.10.
    label_dropout : float, optional
       Dropout probability of class labels for classifier-free guidance. By default 0.0.
    embedding_type : str, optional
        Timestep embedding type: 'positional' for DDPM++, 'fourier' for NCSN++, 'zero' for none
        By default 'positional'.
    channel_mult_noise : int, optional
        Timestep embedding size: 1 for DDPM++, 2 for NCSN++. By default 1.
    encoder_type : str, optional
        Encoder architecture: 'standard' for DDPM++, 'residual' for NCSN++. By default
        'standard'.
    decoder_type : str, optional
        Decoder architecture: 'standard' for both DDPM++ and NCSN++. By default
        'standard'.
    resample_filter : List[int], optional (default=[1,1])
        Resampling filter: [1,1] for DDPM++, [1,3,3,1] for NCSN++.
    additive_pos_embed: bool = False,
        Set to True to add a learned position embedding after the first conv (used in StormCast)


    Reference
    ----------
    Reference: Song, Y., Sohl-Dickstein, J., Kingma, D.P., Kumar, A., Ermon, S. and
    Poole, B., 2020. Score-based generative modeling through stochastic differential
    equations. arXiv preprint arXiv:2011.13456.

    Note
    -----
    Equivalent to the original implementation by Song et al., available at
    https://github.com/yang-song/score_sde_pytorch

    Example
    --------
    >>> model = SongUNet(img_resolution=16, in_channels=2, out_channels=2)
    >>> noise_labels = torch.randn([1])
    >>> class_labels = torch.randint(0, 1, (1, 1))
    >>> input_image = torch.ones([1, 2, 16, 16])
    >>> output_image = model(input_image, noise_labels, class_labels)
    >>> output_image.shape
    torch.Size([1, 2, 16, 16])
    """

    def __init__(
        self,
        img_resolution: Union[List[int], int],
        in_channels: int,
        out_channels: int,
        label_dim: int = 0,
        augment_dim: int = 0,
        model_channels: int = 128,
        channel_mult: List[int] = [1, 2, 2, 2],
        channel_mult_emb: int = 4,
        num_blocks: int = 4,
        attn_resolutions: List[int] = [16],
        dropout: float = 0.10,
        label_dropout: float = 0.0,
        embedding_type: str = "positional",
        channel_mult_noise: int = 1,
        encoder_type: str = "standard",
        decoder_type: str = "standard",
        resample_filter: List[int] = [1, 1],
        additive_pos_embed: bool = False,
    ):
        valid_embedding_types = ["fourier", "positional", "zero"]
        if embedding_type not in valid_embedding_types:
            raise ValueError(
                f"Invalid embedding_type: {embedding_type}. Must be one of {valid_embedding_types}."
            )

        valid_encoder_types = ["standard", "skip", "residual"]
        if encoder_type not in valid_encoder_types:
            raise ValueError(
                f"Invalid encoder_type: {encoder_type}. Must be one of {valid_encoder_types}."
            )

        valid_decoder_types = ["standard", "skip"]
        if decoder_type not in valid_decoder_types:
            raise ValueError(
                f"Invalid decoder_type: {decoder_type}. Must be one of {valid_decoder_types}."
            )

        super().__init__()
        self.label_dropout = label_dropout
        self.embedding_type = embedding_type
        emb_channels = model_channels * channel_mult_emb
        self.emb_channels = emb_channels
        noise_channels = model_channels * channel_mult_noise
        init = dict(init_mode="xavier_uniform")
        init_zero = dict(init_mode="xavier_uniform", init_weight=1e-5)
        init_attn = dict(init_mode="xavier_uniform", init_weight=np.sqrt(0.2))
        block_kwargs = dict(
            emb_channels=emb_channels,
            num_heads=1,
            dropout=dropout,
            skip_scale=np.sqrt(0.5),
            eps=1e-6,
            resample_filter=resample_filter,
            resample_proj=True,
            adaptive_scale=False,
            init=init,
            init_zero=init_zero,
            init_attn=init_attn,
        )

        # for compatibility with older versions that took only 1 dimension
        self.img_resolution = img_resolution
        if isinstance(img_resolution, int):
            self.img_shape_y = self.img_shape_x = img_resolution
        else:
            self.img_shape_y = img_resolution[0]
            self.img_shape_x = img_resolution[1]

        # Optional additive learned positition embed after the first conv
        self.additive_pos_embed = additive_pos_embed
        if self.additive_pos_embed:
            self.spatial_emb = torch.nn.Parameter(
                torch.randn(1, model_channels, self.img_shape_y, self.img_shape_x)
            )
            torch.nn.init.trunc_normal_(self.spatial_emb, std=0.02)

        # Mapping.
        if self.embedding_type != "zero":
            self.map_noise = (
                PositionalEmbedding(num_channels=noise_channels, endpoint=True)
                if embedding_type == "positional"
                else FourierEmbedding(num_channels=noise_channels)
            )
            self.map_label = (
                Linear(in_features=label_dim, out_features=noise_channels, **init)
                if label_dim
                else None
            )
            self.map_augment = (
                Linear(
                    in_features=augment_dim,
                    out_features=noise_channels,
                    bias=False,
                    **init,
                )
                if augment_dim
                else None
            )
            self.map_layer0 = Linear(
                in_features=noise_channels, out_features=emb_channels, **init
            )
            self.map_layer1 = Linear(
                in_features=emb_channels, out_features=emb_channels, **init
            )

        # Encoder.
        self.enc = torch.nn.ModuleDict()
        cout = in_channels
        caux = in_channels
        for level, mult in enumerate(channel_mult):
            res = self.img_shape_y >> level
            if level == 0:
                cin = cout
                cout = model_channels
                self.enc[f"{res}x{res}_conv"] = Conv2d(
                    in_channels=cin, out_channels=cout, kernel=3, **init
                )
            else:
                self.enc[f"{res}x{res}_down"] = UNetBlock(
                    in_channels=cout, out_channels=cout, down=True, **block_kwargs
                )
                if encoder_type == "skip":
                    self.enc[f"{res}x{res}_aux_down"] = Conv2d(
                        in_channels=caux,
                        out_channels=caux,
                        kernel=0,
                        down=True,
                        resample_filter=resample_filter,
                    )
                    self.enc[f"{res}x{res}_aux_skip"] = Conv2d(
                        in_channels=caux, out_channels=cout, kernel=1, **init
                    )
                if encoder_type == "residual":
                    self.enc[f"{res}x{res}_aux_residual"] = Conv2d(
                        in_channels=caux,
                        out_channels=cout,
                        kernel=3,
                        down=True,
                        resample_filter=resample_filter,
                        fused_resample=True,
                        **init,
                    )
                    caux = cout
            for idx in range(num_blocks):
                cin = cout
                cout = model_channels * mult
                attn = res in attn_resolutions
                self.enc[f"{res}x{res}_block{idx}"] = UNetBlock(
                    in_channels=cin, out_channels=cout, attention=attn, **block_kwargs
                )
        skips = [
            block.out_channels for name, block in self.enc.items() if "aux" not in name
        ]

        # Decoder.
        self.dec = torch.nn.ModuleDict()
        for level, mult in reversed(list(enumerate(channel_mult))):
            res = self.img_shape_y >> level
            if level == len(channel_mult) - 1:
                self.dec[f"{res}x{res}_in0"] = UNetBlock(
                    in_channels=cout, out_channels=cout, attention=True, **block_kwargs
                )
                self.dec[f"{res}x{res}_in1"] = UNetBlock(
                    in_channels=cout, out_channels=cout, **block_kwargs
                )
            else:
                self.dec[f"{res}x{res}_up"] = UNetBlock(
                    in_channels=cout, out_channels=cout, up=True, **block_kwargs
                )
            for idx in range(num_blocks + 1):
                cin = cout + skips.pop()
                cout = model_channels * mult
                attn = idx == num_blocks and res in attn_resolutions
                self.dec[f"{res}x{res}_block{idx}"] = UNetBlock(
                    in_channels=cin, out_channels=cout, attention=attn, **block_kwargs
                )
            if decoder_type == "skip" or level == 0:
                if decoder_type == "skip" and level < len(channel_mult) - 1:
                    self.dec[f"{res}x{res}_aux_up"] = Conv2d(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        kernel=0,
                        up=True,
                        resample_filter=resample_filter,
                    )
                self.dec[f"{res}x{res}_aux_norm"] = GroupNorm(
                    num_channels=cout, eps=1e-6
                )
                self.dec[f"{res}x{res}_aux_conv"] = Conv2d(
                    in_channels=cout, out_channels=out_channels, kernel=3, **init_zero
                )

    def forward(self, x, noise_labels, class_labels, augment_labels=None):
        if self.embedding_type != "zero":
            # Mapping.
            emb = self.map_noise(noise_labels)
            emb = (
                emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape)
            )  # swap sin/cos
            if self.map_label is not None:
                tmp = class_labels
                if self.training and self.label_dropout:
                    tmp = tmp * (
                        torch.rand([x.shape[0], 1], device=x.device)
                        >= self.label_dropout
                    ).to(tmp.dtype)
                emb = emb + self.map_label(tmp * np.sqrt(self.map_label.in_features))
            if self.map_augment is not None and augment_labels is not None:
                emb = emb + self.map_augment(augment_labels)
            emb = silu(self.map_layer0(emb))
            emb = silu(self.map_layer1(emb))
        else:
            emb = torch.zeros(
                (noise_labels.shape[0], self.emb_channels), device=x.device
            )

        # Encoder.
        skips = []
        aux = x
        for name, block in self.enc.items():
            if "aux_down" in name:
                aux = block(aux)
            elif "aux_skip" in name:
                x = skips[-1] = x + block(aux)
            elif "aux_residual" in name:
                x = skips[-1] = aux = (x + block(aux)) / np.sqrt(2)
            elif "_conv" in name:
                x = block(x)
                if self.additive_pos_embed:
                    x = x + self.spatial_emb.to(dtype=x.dtype)
                skips.append(x)
            else:
                if isinstance(block, UNetBlock):
                    x = block(x, emb)
                else:
                    x = block(x)
                skips.append(x)

        # Decoder.
        aux = None
        tmp = None
        for name, block in self.dec.items():
            if "aux_up" in name:
                aux = block(aux)
            elif "aux_norm" in name:
                tmp = block(x)
            elif "aux_conv" in name:
                tmp = block(silu(tmp))
                aux = tmp if aux is None else tmp + aux
            else:
                if x.shape[1] != block.in_channels:
                    x = torch.cat([x, skips.pop()], dim=1)
                x = block(x, emb)
        return aux


class StormCastUNet(nn.Module):
    """
    U-Net wrapper for StormCast; used so the same Song U-Net network can be re-used for this model.

    Parameters
    -----------
    img_resolution : int or List[int]
        The resolution of the input/output image.
    img_channels : int
         Number of color channels.
    img_in_channels : int
        Number of input color channels.
    img_out_channels : int
        Number of output color channels.
    use_fp16: bool, optional
        Execute the underlying model at FP16 precision?, by default False.
    sigma_min: float, optional
        Minimum supported noise level, by default 0.
    sigma_max: float, optional
        Maximum supported noise level, by default float('inf').
    sigma_data: float, optional
        Expected standard deviation of the training data, by default 0.5.
    model_type: str, optional
        Class name of the underlying model, by default 'DhariwalUNet'.
    **model_kwargs : dict
        Keyword arguments for the underlying model.

    """

    def __init__(
        self,
        img_resolution,
        img_in_channels,
        img_out_channels,
        hidden_channels,
        channel_mult,
        num_blocks,
        dropout,
        **model_kwargs,
    ):
        super().__init__()

        if isinstance(img_resolution, int):
            self.img_shape_x = self.img_shape_y = img_resolution
        else:
            self.img_shape_x = img_resolution[0]
            self.img_shape_y = img_resolution[1]

        self.img_in_channels = img_in_channels
        self.img_out_channels = img_out_channels

        self.model = SongUNet(
            img_resolution=img_resolution,
            in_channels=img_in_channels,
            out_channels=img_out_channels,
            model_channels=hidden_channels,
            channel_mult=channel_mult,
            num_blocks=num_blocks,
            dropout=dropout,
            **model_kwargs,
        )

    def forward(self, x, **model_kwargs):
        """Run a forward pass of the StormCast regression U-Net.

        Args:
            x (torch.Tensor): input to the U-Net
            force_fp32 (bool, optional): force casting to fp_32 if True. Defaults to False.

        Raises:
            ValueError: If input data type is a mismatch with provided options

        Returns:
            D_x (torch.Tensor): Output (prediction) of the U-Net
        """
        if len(x.shape) == 5:  # x.shape = [B,T,C,H,W]
            x = x.flatten(1, 2)
        return self.model(
            x,
            torch.zeros(x.shape[0], dtype=x.dtype, device=x.device),
            class_labels=None,
            **model_kwargs,
        )

# model = StormCastUNet(
#     img_resolution=(256, 256),
#     img_in_channels=16,
#     img_out_channels=16,
#     embedding_type="zero",
#     channel_mult=[1, 2, 2, 2, 2],
#     attn_resolutions=[],
#     additive_pos_embed=False,
# ).cuda()
# print(model)
# x = torch.randn(1, 16, 256, 256).cuda()
# y = model(x)
# print(y.shape)
