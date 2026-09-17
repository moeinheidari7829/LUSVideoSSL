#!/usr/bin/env python3
"""
main_moco_video.py — VideoMoCo training using moco-v3's MoCo_ViT
builder, with a ViT-S/16 backbone adapted to 16-frame clips.
Run from inside the moco-v3 repo.
"""
import argparse
import os
import torch
from torch.utils.data import DataLoader

import vits
import moco.builder
from tubelet_patch_embed import build_video_vit_from_model
from video_dataset import VideoClipMoCoDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--clip-len", type=int, default=16)
    ap.add_argument("--tubelet-size", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1.5e-4)
    ap.add_argument("--moco-dim", type=int, default=256)
    ap.add_argument("--moco-mlp-dim", type=int, default=4096)
    ap.add_argument("--moco-t", type=float, default=0.2)
    ap.add_argument("--checkpoint-dir", default="./outputs/checkpoints")
    ap.add_argument("--save-every", type=int, default=5)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def base_encoder_fn(num_classes):
        model = vits.vit_small(num_classes=num_classes)
        model = build_video_vit_from_model(
            model,
            clip_len=args.clip_len,
            tubelet_size=args.tubelet_size,
        )
        return model

    model = moco.builder.MoCo_ViT(
        base_encoder_fn,
        dim=args.moco_dim,
        mlp_dim=args.moco_mlp_dim,
        T=args.moco_t,
    ).to(device)

    dataset = VideoClipMoCoDataset(args.manifest, clip_len=args.clip_len)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.1,
    )

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0

        for i, (q, k) in enumerate(loader):
            q = q.to(device, non_blocking=True)
            k = k.to(device, non_blocking=True)

            m = 0.99
            loss = model(q, k, m)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if i % 20 == 0:
                print(f"epoch {epoch} step {i}/{len(loader)} loss {loss.item():.4f}")

        avg = total_loss / len(loader)
        print(f"== epoch {epoch} avg loss: {avg:.4f} ==")

        if (epoch + 1) % args.save_every == 0:
            ckpt_path = os.path.join(
                args.checkpoint_dir,
                f"videomoco_epoch{epoch+1}.pt"
            )
            torch.save(model.state_dict(), ckpt_path)
            print(f"saved checkpoint: {ckpt_path}")

            ckpts = sorted(
                f for f in os.listdir(args.checkpoint_dir)
                if f.startswith("videomoco_epoch")
            )
            for old in ckpts[:-2]:
                os.remove(os.path.join(args.checkpoint_dir, old))


if __name__ == "__main__":
    main()