# Sockeye frozen LUS feature extraction

Copy `common.py`, both extractor files, and both SLURM files into:

```text
/arc/project/st-ilker-1/junbo2/code/jepa/tools/lus_eval/
```

Before full jobs, run one-sample smoke tests from a GPU allocation:

```bash
python extract_videomae_features.py ... --limit 1
python extract_vjepa_features.py ... --limit 1
```

Inputs are deterministic center clips of 16 frames with stride 2. Short videos
repeat the final frame. Static images are repeated for 16 frames. Frames are
resized with short side 256, center-cropped to 224, and ImageNet-normalized.

Outputs contain pooled `[N,384]` features and spatially pooled temporal
`[N,8,384]` features, plus manifest metadata.
