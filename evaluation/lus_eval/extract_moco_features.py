from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

from common import extract, parser


# MoCo v3 (video-adapted) is an external dependency.
# Set MOCOV3_REPO to the path of the pretraining/MoCo-v3 directory before running.
mocov3_repo = os.environ.get("MOCOV3_REPO")

if not mocov3_repo:
    raise RuntimeError(
        "MOCOV3_REPO is not set. "
        "Set it to the path of the pretraining/MoCo-v3 directory."
    )

MOCOV3_REPO = Path(mocov3_repo).expanduser().resolve()

if not MOCOV3_REPO.is_dir():
    raise FileNotFoundError(
        f"MoCo-v3 repository not found: {MOCOV3_REPO}"
    )

sys.path.insert(0, str(MOCOV3_REPO))

import vits  # noqa: E402
from tubelet_patch_embed import build_video_vit_from_model  # noqa: E402
from moco.builder import MoCo_ViT  # noqa: E402


CLIP_LEN = 16
TUBELET_SIZE = 2


def build_encoder(checkpoint_path):
    """Rebuild the video-adapted ViT-S/16 MoCo v3 encoder and load a
    pretraining checkpoint (the full MoCo_ViT.state_dict() saved by
    main_moco_video.py)."""

    def base_encoder_fn(num_classes):
        model = vits.vit_small(num_classes=num_classes)
        model = build_video_vit_from_model(
            model, clip_len=CLIP_LEN, tubelet_size=TUBELET_SIZE
        )
        return model

    moco = MoCo_ViT(base_encoder_fn, dim=256, mlp_dim=4096, T=0.2)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    # strict=False matches the loading convention used throughout this
    # project's own pretraining/eval scripts (main_moco_video.py's checkpoint
    # is architecturally identical, but this stays permissive rather than
    # risking a spurious strict-mode mismatch on momentum-encoder buffers).
    message = moco.load_state_dict(checkpoint, strict=False)
    print("CHECKPOINT:", message)

    return moco.base_encoder


def forward_patch_tokens(encoder, clips):
    """Run the ViT backbone up to (and including) the final norm, and return
    the patch-token sequence *without* the class token: (B, 1568, 384).

    This bypasses the encoder's normal forward()/forward_features(), which
    (in the timm version this backbone is built on) pool to the class token
    before returning -- run_probes.py needs the full per-token sequence, not
    a pooled embedding.
    """
    # common.py's preprocess() (shared eval pipeline) hands clips over as
    # (B, C, T, H, W); TubeletPatchEmbed (shared with this project's own
    # pretraining pipeline, where video_dataset.py already produces
    # (B, T, C, H, W)) expects the latter and permutes back internally --
    # without this, channels and time get silently swapped and Conv3d fails
    # on channel count.
    clips = clips.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W) -> (B, T, C, H, W)
    x = encoder.patch_embed(clips)  # (B, 1568, 384)
    cls_token = encoder.cls_token.expand(x.shape[0], -1, -1)
    x = torch.cat((cls_token, x), dim=1)
    x = encoder.pos_drop(x + encoder.pos_embed)
    for block in encoder.blocks:
        x = block(x)
    x = encoder.norm(x)
    return x[:, 1:]  # drop the class token -> (B, 1568, 384)


def main():
    args = parser("MoCo-v3-Small").parse_args()

    encoder = build_encoder(args.checkpoint).cuda()

    def forward_tokens(clips):
        return forward_patch_tokens(encoder, clips)

    extract(encoder, forward_tokens, args)


if __name__ == "__main__":
    main()
