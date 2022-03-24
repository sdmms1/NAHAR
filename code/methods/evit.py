import torch
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from linformer import Linformer

default_linformer = Linformer(dim=128,
                              seq_len=49 + 1,  # 7x7 patches + 1 cls-token
                              depth=12,
                              heads=8,
                              k=64
                              )


class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, dim, transformer=None, patch=False, pool='cls', channels=3):
        super().__init__()
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        if patch:
            print('Efficient Vit for image initialize!')
            assert image_size % patch_size == 0, 'image dimensions must be divisible by the patch size'
            num_patches = (image_size // patch_size) ** 2
            patch_dim = channels * patch_size ** 2

            self.to_patch_embedding = nn.Sequential(
                Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
                nn.Linear(patch_dim, dim),
            )
        else:
            print('Efficient Vit for sequence initialize!')
            num_patches = 200
            self.to_patch_embedding = nn.Sequential(
                Rearrange('b h w -> b w h', h=256),
                nn.Linear(256, dim),
            )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        if transformer is None:
            self.transformer = Linformer(dim=dim, seq_len=num_patches + 1, depth=12, heads=8, k=64)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.final_feat_dim = dim

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return x
