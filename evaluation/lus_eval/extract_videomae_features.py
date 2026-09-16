from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

from common import extract, parser


# VideoMAE is an external dependency.
# Set VIDEOMAE_REPO to the path of the VideoMAE repository before running.
videomae_repo = os.environ.get("VIDEOMAE_REPO")

if not videomae_repo:
    raise RuntimeError(
        "VIDEOMAE_REPO is not set. "
        "Set it to the path of the VideoMAE repository."
    )

VIDEOMAE_REPO = Path(videomae_repo).expanduser().resolve()

if not VIDEOMAE_REPO.is_dir():
    raise FileNotFoundError(
        f"VideoMAE repository not found: {VIDEOMAE_REPO}"
    )

sys.path.insert(0, str(VIDEOMAE_REPO))

import modeling_pretrain  # noqa: F401,E402
from timm.models import create_model  # noqa: E402


def main():
    args = parser("VideoMAE-Small").parse_args()

    model = create_model(
        "pretrain_videomae_small_patch16_224",
        pretrained=False,
        decoder_depth=4,
    )

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    message = model.load_state_dict(checkpoint["model"], strict=True)
    print("CHECKPOINT:", message)

    encoder = model.encoder.cuda()

    def forward_tokens(clips):
        mask = torch.zeros(
            clips.shape[0],
            1568,
            dtype=torch.bool,
            device=clips.device,
        )
        return encoder(clips, mask)

    extract(encoder, forward_tokens, args)


if __name__ == "__main__":
    main()