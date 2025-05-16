import numpy as np
import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed
from india_benchmark.utils.pos_embed import get_2d_sincos_pos_embed, get_1d_sincos_pos_embed_from_grid


class WeatherEmbedding(nn.Module):
    def __init__(
        self,
        variables,
        img_size,
        patch_size=2,
        embed_dim=1024,
        num_heads=16,
    ):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.variables = tuple(variables)

        # variable tokenization: separate embedding layer for each input variable
        self.token_embeds = nn.ModuleList(
            [PatchEmbed(None, patch_size, 1, embed_dim) for i in range(len(variables))]
        )
        self.num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)

        # variable aggregation: a learnable query and a single-layer cross attention
        self.channel_query = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
        self.channel_agg = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        # positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim), requires_grad=True)
        self.channel_embed = nn.Parameter(torch.zeros(1, len(variables), embed_dim), requires_grad=True)

        self.initialize_weights()

    def initialize_weights(self):
        # initialize pos_emb and var_emb
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.img_size[0] / self.patch_size),
            int(self.img_size[1] / self.patch_size),
            cls_token=False,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        channel_embed = get_1d_sincos_pos_embed_from_grid(self.channel_embed.shape[-1], np.arange(len(self.variables)))
        self.channel_embed.data.copy_(torch.from_numpy(channel_embed).float().unsqueeze(0))

        # # token embedding layer
        # for i in range(len(self.token_embeds)):
        #     w = self.token_embeds[i].proj.weight.data
        #     trunc_normal_(w.view([w.shape[0], -1]), std=0.02)

    def aggregate_variables(self, x: torch.Tensor):
        """
        x: B, V, L, D
        """
        b, _, l, _ = x.shape
        x = torch.einsum("bvld->blvd", x)
        x = x.flatten(0, 1)  # BxL, V, D

        var_query = self.channel_query.repeat_interleave(x.shape[0], dim=0)
        x, _ = self.channel_agg(var_query, x, x)  # BxL, D
        x = x.squeeze()

        x = x.unflatten(dim=0, sizes=(b, l))  # B, L, D
        return x

    def forward(self, x: torch.Tensor):
        # tokenize each variable separately
        embeds = []
        for i in range(x.shape[1]):
            embed_variable = self.token_embeds[i](x[:, i : i + 1]) # B, L, D
            embeds.append(embed_variable)
        x = torch.stack(embeds, dim=1)  # B, V, L, D

        # add variable embedding
        x = x + self.channel_embed.unsqueeze(2)
        x = x + self.pos_embed.unsqueeze(1)

        # variable aggregation
        x = self.aggregate_variables(x)  # B, L, D

        return x