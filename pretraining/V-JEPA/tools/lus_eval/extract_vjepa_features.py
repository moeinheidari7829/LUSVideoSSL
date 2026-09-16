from __future__ import annotations

import sys
from pathlib import Path

import torch

from common import extract, parser

VJEPA_REPO = Path("/arc/project/st-ilker-1/junbo2/code/jepa")
sys.path.insert(0, str(VJEPA_REPO))

import src.models.vision_transformer as vit  # noqa: E402


def main():
    args = parser("V-JEPA-Small").parse_args()
    encoder = vit.vit_small(
        img_size=224,
        patch_size=16,
        num_frames=16,
        tubelet_size=2,
        uniform_power=False,
        use_sdpa=False,
        use_SiLU=False,
        tight_SiLU=True,
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state = checkpoint.get("target_encoder", checkpoint.get("encoder"))
    if state is None:
        raise KeyError("checkpoint has neither target_encoder nor encoder")
    state = {
        key.replace("module.", "").replace("backbone.", ""): value
        for key, value in state.items()
    }
    message = encoder.load_state_dict(state, strict=False)
    if message.missing_keys or message.unexpected_keys:
        raise RuntimeError(f"checkpoint mismatch: {message}")
    print("CHECKPOINT:", message)
    encoder = encoder.cuda()

    def forward_tokens(clips):
        tokens = encoder(clips)
        if isinstance(tokens, list):
            if len(tokens) != 1:
                raise RuntimeError(f"unexpected encoder list length: {len(tokens)}")
            tokens = tokens[0]
        return tokens

    extract(encoder, forward_tokens, args)


if __name__ == "__main__":
    main()

