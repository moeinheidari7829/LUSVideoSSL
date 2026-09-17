# MoCo v3 lung ultrasound pretraining

This folder contains the MoCo v3 code used for lung ultrasound self-supervised
pretraining, vendored from `facebookresearch/moco-v3` (An Empirical Study of
Training Self-Supervised Vision Transformers, Chen, Xie & He, ICCV 2021),
plus a video adaptation layered on top so it pretrains on 16-frame LUS clips
instead of single-image crops.

## What's added on top of stock MoCo v3

MoCo v3 is originally a two-crop, single-image contrastive method. To turn it
into a video pretext task for LUS clips, three things change here (everything
else -- the `MoCo_ViT` projector/predictor MLPs, InfoNCE loss, momentum
update -- is unmodified upstream MoCo v3):

- **`tubelet_patch_embed.py`** replaces the ViT's 2D patch embedding with a
  3D `Conv3d` tubelet embedding (tubelet size 2, patch size 16x16, over
  16-frame clips) and inflates the learned positional embedding to match the
  new temporal x spatial patch grid (8 x 196 = 1568 patches).
- **`video_dataset.py`** samples two 16-frame clips (`q`, `k`) per video
  instead of two image crops, applying the same random augmentation to every
  frame within a clip (shared RNG seed) so the two views stay temporally
  coherent.
- **`moco/builder.py`** -- `concat_all_gather` and the rank lookup in
  `contrastive_loss` now fall back to a no-op / rank 0 when
  `torch.distributed` is never initialized, so this runs on a single GPU
  instead of requiring `DistributedDataParallel`.

`main_moco_video.py` is the pretraining entrypoint; `make_video_manifest.py`
builds its input manifest from a directory of videos.

The pretraining split is the same **362 COVID-BLUeS videos** used by the
VideoMAE and V-JEPA branches in this repo.

## Usage

Expected dataset layout (`DATA_ROOT` is the parent of the video directory):

```text
COVID-BLUeS/
└── lus_videos/
    ├── patient_10_L1.mp4
    ├── patient_10_L2.mp4
    └── ...
```

```bash
cd pretraining/MoCo-v3
pip install -r requirements.txt

# Build the pretraining manifest (video_path,patient_id,video_id)
python make_video_manifest.py \
  --video_dir "$DATA_ROOT/lus_videos" \
  --manifest  ./manifests/covid_blues_mocov3_train.csv

# Pretrain (randomly initialized, no ImageNet weights, AdamW wd 0.1,
# fixed MoCo momentum m=0.99, 100 epochs -- matching the shared protocol
# used by the VideoMAE and V-JEPA branches)
python main_moco_video.py \
  --manifest ./manifests/covid_blues_mocov3_train.csv \
  --clip-len 16 --tubelet-size 2 \
  --batch-size 4 --epochs 100 --lr 1.5e-4 \
  --checkpoint-dir ./outputs/checkpoints
```

Or on the Slurm cluster, with an existing `mocov3` Conda environment:

```bash
export MANIFEST=/absolute/path/to/covid_blues_mocov3_train.csv
export MOCOV3_ENV=mocov3   # Conda environment name or absolute path
export OUTPUT_DIR=/absolute/path/to/outputs/mocov3_100e_seed0
sbatch train_mocov3_100e.slurm
```

Checkpoints land in `$OUTPUT_DIR/checkpoints/videomoco_epoch<N>.pt` (only the
last two are kept). Feed the final one to
`evaluation/lus_eval/extract_moco_features.py` for frozen feature extraction
and probing -- see the repo-root README and `evaluation/lus_eval/README.md`.

## Attribution and license

Everything except `tubelet_patch_embed.py`, `video_dataset.py`,
`main_moco_video.py`, and `make_video_manifest.py` is vendored, with minimal
modification, from
[facebookresearch/moco-v3](https://github.com/facebookresearch/moco-v3),
licensed CC-BY-NC 4.0 -- see `LICENSE`.
