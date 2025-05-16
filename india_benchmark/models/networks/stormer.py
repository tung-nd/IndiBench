import torch
import torch.nn as nn
from timm.models.vision_transformer import trunc_normal_, Mlp, DropPath, PatchEmbed
from xformers.ops import memory_efficient_attention, unbind
from india_benchmark.models.networks.weather_embedding import WeatherEmbedding


class MemEffAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_bias=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    # Block with memory efficient attention
    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.,
            qkv_bias=False,
            proj_drop=0.,
            attn_drop=0.,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            mlp_layer=Mlp,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = MemEffAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path1(self.attn(self.norm1(x)))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


class Stormer(nn.Module):
    def __init__(self, 
        in_img_size,
        variables,
        n_input_steps,
        patch_size=2,
        emb_type='linear',
        hidden_size=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        drop_rate=0.0,
    ):
        super().__init__()
        
        self.in_img_size = in_img_size
        variables_all_steps = []
        for i in range(n_input_steps):
            variables_step = [f'{var}_{i}' for var in variables]
            variables_all_steps.extend(variables_step)
        self.variables = variables_all_steps
        self.n_input_steps = n_input_steps
        self.patch_size = patch_size
        
        # embedding
        if emb_type == 'climax':
            self.embedding = WeatherEmbedding(
                variables=variables_all_steps,
                img_size=in_img_size,
                patch_size=patch_size,
                embed_dim=hidden_size,
                num_heads=num_heads,
            )
        elif emb_type == 'linear':
            self.embedding = PatchEmbed(
                img_size=in_img_size,
                patch_size=patch_size,
                in_chans=len(variables_all_steps),
                embed_dim=hidden_size,
                flatten=True,
            )
        else:
            raise ValueError(f'Unknown emb_type: {emb_type}')
        self.embed_norm_layer = nn.LayerNorm(hidden_size)
        
        # backbone
        self.blocks = nn.ModuleList([
            Block(
                hidden_size,
                num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                proj_drop=drop_rate,
                attn_drop=drop_rate,
                drop_path=drop_rate,
            ) for _ in range(depth)
        ])
        
        # prediction layer
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, len(variables) * patch_size**2)
        )

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
        self.apply(_basic_init)

    def unpatchify(self, x: torch.Tensor, h=None, w=None):
        """
        x: (B, L, V * patch_size**2)
        return imgs: (B, V, H, W)
        """
        p = self.patch_size
        v = len(self.variables) // self.n_input_steps
        h = self.in_img_size[0] // p if h is None else h // p
        w = self.in_img_size[1] // p if w is None else w // p
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, v))
        x = torch.einsum("nhwpqv->nvhpwq", x)
        imgs = x.reshape(shape=(x.shape[0], v, h * p, w * p))
        return imgs

    def forward(self, x):
        if len(x.shape) == 5:  # x.shape = [B,T,C,H,W]
            x = x.flatten(1, 2)

        x = self.embedding(x) # B, L, D
        x = self.embed_norm_layer(x)

        for block in self.blocks:
            x = block(x)
        
        x = self.head(x)
        x = self.unpatchify(x)
        
        return x
