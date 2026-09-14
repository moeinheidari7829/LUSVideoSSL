from __future__ import annotations

import sys
from pathlib import Path

import torch

from common import extract, parser

VIDEOMAE_REPO = Path("/arc/project/st-ilker-1/junbo2/src/VideoMAE")
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
            clips.shape[0], 1568, dtype=torch.bool, device=clips.device
        )
        return encoder(clips, mask)

    extract(encoder, forward_tokens, args)


if __name__ == "__main__":
    main()

