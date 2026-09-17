#!/usr/bin/env python3
"""
video_dataset.py — two-view clip sampler for VideoMoCo (ViT-S/16).
Reads covid_blues_videos_train.csv (or val) and returns two augmented
16-frame clips per video.
"""
import os
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

CLIP_LEN = 16

def load_all_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return frames

def sample_clip(frames, clip_len=CLIP_LEN):
    n = len(frames)
    if n == 0:
        return []
    if n < clip_len:
        idxs = [i % n for i in range(clip_len)]
    else:
        start = random.randint(0, n - clip_len)
        idxs = list(range(start, start + clip_len))
    return [frames[i] for i in idxs]

class VideoClipMoCoDataset(Dataset):
    """
    manifest_csv: video_path,patient_id,video_id
    Returns (q, k) for MoCo, each of shape (T, C, H, W).
    """
    def __init__(self, manifest_csv, clip_len=CLIP_LEN, image_size=224):
        import csv
        with open(manifest_csv) as f:
            self.rows = list(csv.DictReader(f))
        self.clip_len = clip_len
        self.image_size = image_size
        self.aug = T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.5, 1.0)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.4, 0.4, 0.4, 0.1),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.rows)

    def _clip_to_tensor(self, clip_frames):
        # Apply SAME augmentation to all frames in a clip for temporal consistency
        seed = random.randint(0, 2**31)
        tensors = []
        for frame in clip_frames:
            random.seed(seed)
            torch.manual_seed(seed)
            from PIL import Image
            img = Image.fromarray(frame)
            tensors.append(self.aug(img))
        return torch.stack(tensors, dim=0)  # (T, C, H, W)

    def __getitem__(self, idx):
        video_path = self.rows[idx]["video_path"]
        frames = load_all_frames(video_path)
        if len(frames) == 0:
            # fallback: zeros if decode fails
            dummy = torch.zeros(self.clip_len, 3, self.image_size, self.image_size)
            return dummy, dummy

        clip_q = sample_clip(frames, self.clip_len)
        clip_k = sample_clip(frames, self.clip_len)

        q = self._clip_to_tensor(clip_q)
        k = self._clip_to_tensor(clip_k)
        return q, k