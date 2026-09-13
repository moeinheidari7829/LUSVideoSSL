from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from decord import VideoReader, cpu
from PIL import Image, ImageSequence
from torch.utils.data import DataLoader, Dataset

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
MEAN = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
STD = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)


def _sample_indices(length: int, frames: int = 16, sampling_rate: int = 2) -> np.ndarray:
    if length < 1:
        raise ValueError("media contains no frames")
    span = (frames - 1) * sampling_rate + 1
    start = max(0, (length - span) // 2)
    return np.minimum(start + np.arange(frames) * sampling_rate, length - 1)


def _read_gif(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        frames = [np.asarray(frame.convert("RGB")) for frame in ImageSequence.Iterator(image)]
    return np.stack(frames)


def read_frames(path: str, modality: str, frames: int = 16, sampling_rate: int = 2) -> np.ndarray:
    media_path = Path(path)
    if modality == "image" or media_path.suffix.lower() in IMAGE_SUFFIXES:
        with Image.open(media_path) as image:
            frame = np.asarray(image.convert("RGB"))
        return np.repeat(frame[None], frames, axis=0)
    if media_path.suffix.lower() == ".gif":
        video = _read_gif(media_path)
        return video[_sample_indices(len(video), frames, sampling_rate)]
    reader = VideoReader(str(media_path), ctx=cpu(0), num_threads=1)
    indices = _sample_indices(len(reader), frames, sampling_rate)
    return reader.get_batch(indices).asnumpy()


def preprocess(frames: np.ndarray, crop_size: int = 224) -> torch.Tensor:
    x = torch.from_numpy(frames).permute(0, 3, 1, 2).float().div_(255.0)
    height, width = x.shape[-2:]
    scale = 256.0 / min(height, width)
    new_height, new_width = round(height * scale), round(width * scale)
    x = F.interpolate(x, size=(new_height, new_width), mode="bilinear", align_corners=False)
    top = (new_height - crop_size) // 2
    left = (new_width - crop_size) // 2
    x = x[:, :, top:top + crop_size, left:left + crop_size]
    x = (x - MEAN) / STD
    return x.permute(1, 0, 2, 3).contiguous()


class ManifestDataset(Dataset):
    def __init__(self, manifest: Path, limit: int | None = None):
        self.frame = pd.read_csv(manifest)
        if limit is not None:
            self.frame = self.frame.iloc[:limit].copy()
        required = {"sample_id", "patient_id", "path", "label_id", "modality"}
        missing = required - set(self.frame.columns)
        if missing:
            raise ValueError(f"manifest missing columns: {sorted(missing)}")
        missing_paths = [p for p in self.frame.path if not Path(p).is_file()]
        if missing_paths:
            raise FileNotFoundError(f"missing media: {missing_paths[:3]}")

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        clip = preprocess(read_frames(row.path, row.modality))
        return clip, index


def parser(model_name: str) -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=f"Extract frozen {model_name} LUS features")
    result.add_argument("--manifest", type=Path, required=True)
    result.add_argument("--checkpoint", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--batch-size", type=int, default=4)
    result.add_argument("--workers", type=int, default=4)
    result.add_argument("--limit", type=int, help="smoke-test first N manifest rows")
    return result


def extract(encoder, forward_tokens, args) -> None:
    dataset = ManifestDataset(args.manifest, args.limit)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    pooled, temporal, indices = [], [], []
    encoder.eval().requires_grad_(False)
    with torch.inference_mode():
        for clips, batch_indices in loader:
            clips = clips.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                tokens = forward_tokens(clips)
            if tokens.ndim != 3 or tokens.shape[1:] != (1568, 384):
                raise RuntimeError(f"unexpected token shape: {tuple(tokens.shape)}")
            tokens = tokens.float()
            pooled.append(tokens.mean(1).cpu())
            temporal.append(tokens.reshape(-1, 8, 196, 384).mean(2).cpu())
            indices.append(batch_indices)

    order = torch.cat(indices).numpy()
    if not np.array_equal(order, np.arange(len(dataset))):
        raise RuntimeError("feature order differs from manifest order")
    frame = dataset.frame
    payload = {
        "features": torch.cat(pooled).numpy(),
        "temporal_features": torch.cat(temporal).numpy(),
        "labels": frame.label_id.to_numpy(np.int64),
        "patient_ids": frame.patient_id.astype(str).to_numpy(),
        "sample_ids": frame.sample_id.astype(str).to_numpy(),
        "paths": frame.path.astype(str).to_numpy(),
        "modalities": frame.modality.astype(str).to_numpy(),
    }
    if "fold" in frame:
        payload["folds"] = frame.fold.to_numpy(np.int64)
    if "split" in frame:
        payload["splits"] = frame.split.astype(str).to_numpy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)
    print("OUTPUT:", args.output)
    print("POOLED:", payload["features"].shape)
    print("TEMPORAL:", payload["temporal_features"].shape)

