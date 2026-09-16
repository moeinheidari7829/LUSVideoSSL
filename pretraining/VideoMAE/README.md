# VideoMAE lung ultrasound pretraining

This folder contains the VideoMAE code used for lung ultrasound self-supervised
pretraining, based on MCG-NJU/VideoMAE commit
`14ef8d856287c94ef1f985fe30f958eb4ec2c55d`.

The pretraining split contains **362 COVID-BLUeS videos**. `all_videomae.txt`
contains relative paths and a label field, for example:

```text
lus_videos/patient_10_L1.mp4 0
```

Expected dataset layout (`DATA_ROOT` is the parent of `lus_videos`):

```text
COVID-BLUeS/
└── lus_videos/
    ├── patient_10_L1.mp4
    ├── patient_10_L2.mp4
    └── ...
```

From the repository root, on the Slurm cluster with the required modules and an
existing VideoMAE Conda environment:

```bash
cd pretraining/VideoMAE
export DATA_ROOT=/absolute/path/to/COVID-BLUeS
export VIDEOMAE_ENV=videomae  # Conda environment name or absolute path
export OUTPUT_DIR=/absolute/path/to/outputs/videomae_100e_seed0
sbatch train_videomae_100e.slurm
```

The job prefixes split paths with `DATA_ROOT` in a temporary runtime manifest
and writes training outputs to `OUTPUT_DIR`. Training settings are defined in
`train_videomae_100e.slurm`.

The current launcher resolves its code directory through `BASH_SOURCE[0]`;
if Slurm uses a spool copy, this path must be resolved before the job can run.

Compatibility changes included in this copy:

- `kinetics.py` removes the unused NumPy `disp` import.
- `utils.py` replaces `torch._six.inf` with `math.inf` (`from math import inf`).

See `INSTALL.md` for upstream dependency guidance and `LICENSE` / `NOTICE.md`
for licensing and attribution.
