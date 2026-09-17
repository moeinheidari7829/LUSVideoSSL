#!/usr/bin/env python3
import torch
import torch.nn as nn

class TubeletPatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, tubelet_size=2,
                 clip_len=16, in_chans=3, embed_dim=384):
        super().__init__()
        self.tubelet_size = tubelet_size
        self.patch_size = patch_size
        self.num_temporal = clip_len // tubelet_size
        self.num_spatial = (img_size // patch_size) ** 2
        self.num_patches = self.num_temporal * self.num_spatial

        self.proj = nn.Conv3d(
            in_chans,
            embed_dim,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size),
        )

    def forward(self, x):
        # x: (B, T, C, H, W) -> (B, C, T, H, W)
        x = x.permute(0, 2, 1, 3, 4)
        x = self.proj(x)                  # (B, D, T', H', W')
        x = x.flatten(2).transpose(1, 2) # (B, N, D)
        return x


def build_video_vit_from_model(model, clip_len=16, tubelet_size=2,
                               img_size=224, patch_size=16):
    """
    Takes an already-created ViT model and replaces its 2D patch embed
    with a 3D tubelet patch embed.
    """
    embed_dim = getattr(model, "embed_dim", model.patch_embed.proj.out_channels)

    new_patch = TubeletPatchEmbed(
        img_size=img_size,
        patch_size=patch_size,
        tubelet_size=tubelet_size,
        clip_len=clip_len,
        in_chans=3,
        embed_dim=embed_dim,
    )

    model.patch_embed = new_patch

    num_patches = new_patch.num_patches

    old_pos = model.pos_embed
    cls_pos = old_pos[:, :1, :]

    new_pos = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
    new_pos.data[:, :1, :] = cls_pos
    nn.init.trunc_normal_(new_pos.data[:, 1:, :], std=0.02)
    model.pos_embed = new_pos

    return model