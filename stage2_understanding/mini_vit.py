"""
🔧 mini_vit.py

Tiny toy Vision Transformer
Minimal transformer encoder for images
"""

import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, emb_dim=128):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_channels=3,
            out_channels=emb_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        # x: [B, 3, H, W]
        x = self.proj(x)  # [B, emb_dim, H/ps, W/ps]
        x = x.flatten(2)  # [B, emb_dim, num_patches]
        x = x.transpose(1, 2)  # [B, num_patches, emb_dim]
        return x


class MiniViT(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        emb_dim=128,
        num_heads=4,
        num_layers=4,
        num_classes=10,
    ):
        super().__init__()

        self.patch_embed = PatchEmbed(img_size, patch_size, emb_dim)
        self.num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, emb_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=num_heads,
            dim_feedforward=emb_dim * 4,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.cls_head = nn.Linear(emb_dim, num_classes)

        # CLS Token (optional)
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))

    def forward(self, x):
        B = x.size(0)

        x = self.patch_embed(x)  # [B, num_patches, emb_dim]

        cls_token = self.cls_token.expand(B, 1, -1)
        x = torch.cat((cls_token, x), dim=1)

        x = x + self.pos_embed[:, : x.size(1), :]

        x = self.encoder(x)  # [B, num_tokens, emb_dim]
        cls_out = x[:, 0]  # classification token

        return self.cls_head(cls_out)


if __name__ == "__main__":
    # Demo with a random image
    img = torch.randn(1, 3, 224, 224)
    model = MiniViT()
    out = model(img)
    print("Output shape:", out.shape)  # Output shape: torch.Size([1, 10])
