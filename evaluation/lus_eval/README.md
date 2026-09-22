# Sockeye frozen LUS feature extraction

Copy `common.py`, all extractor files, and all SLURM files into:

```text
evaluation/lus_eval/
```

Before full jobs, run one-sample smoke tests from a GPU allocation:

```bash
python extract_videomae_features.py ... --limit 1
python extract_vjepa_features.py ... --limit 1
python extract_moco_features.py ... --limit 1
```

Inputs are deterministic center clips of 16 frames with stride 2. Short videos
repeat the final frame. Static images are repeated for 16 frames. Frames are
resized with short side 256, center-cropped to 224, and ImageNet-normalized.

Outputs contain pooled `[N,384]` features and spatially pooled temporal
`[N,8,384]` features, plus manifest metadata.
