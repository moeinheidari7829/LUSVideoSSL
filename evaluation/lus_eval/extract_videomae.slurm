#!/bin/bash

#SBATCH --job-name=lus_vmae_feat
#SBATCH --account=st-ilker-1-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=02:00:00

set -euo pipefail

# Locate this repository automatically.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

module load gcc/9.4.0
module load miniconda3/4.9.2

export PS1=""
source "$(conda info --base)/etc/profile.d/conda.sh"

# Can be either a Conda environment name or an absolute environment path.
VIDEOMAE_ENV="${VIDEOMAE_ENV:-videomae}"
conda activate "$VIDEOMAE_ENV"

# Default locations relative to the repository.
# Each one can be overridden with an environment variable.
export VIDEOMAE_REPO="${VIDEOMAE_REPO:-$REPO_ROOT/third_party/VideoMAE}"

MANIFESTS="${MANIFESTS:-$REPO_ROOT/manifests}"
FEATURES="${FEATURES:-$REPO_ROOT/outputs/features/videomae_small}"
CHECKPOINT="${VIDEOMAE_CHECKPOINT:-$REPO_ROOT/checkpoints/videomae/checkpoint-99.pth}"

POCUS_MANIFEST="$MANIFESTS/pocus_diagnostic_3class.csv"
MENDELEY_MANIFEST="$MANIFESTS/mendeley_uganda_diagnostic_3class.csv"

# Validate required inputs before starting the GPU job.
if [[ ! -d "$VIDEOMAE_REPO" ]]; then
    echo "ERROR: VideoMAE repository not found: $VIDEOMAE_REPO"
    exit 1
fi

if [[ ! -f "$CHECKPOINT" ]]; then
    echo "ERROR: VideoMAE checkpoint not found: $CHECKPOINT"
    exit 1
fi

if [[ ! -f "$POCUS_MANIFEST" ]]; then
    echo "ERROR: POCUS manifest not found: $POCUS_MANIFEST"
    exit 1
fi

if [[ ! -f "$MENDELEY_MANIFEST" ]]; then
    echo "ERROR: Mendeley-Uganda manifest not found: $MENDELEY_MANIFEST"
    exit 1
fi

mkdir -p "$FEATURES"

cd "$SCRIPT_DIR"

python extract_videomae_features.py \
    --manifest "$POCUS_MANIFEST" \
    --checkpoint "$CHECKPOINT" \
    --output "$FEATURES/pocus_diagnostic.npz"

python extract_videomae_features.py \
    --manifest "$MENDELEY_MANIFEST" \
    --checkpoint "$CHECKPOINT" \
    --output "$FEATURES/mendeley_uganda_diagnostic.npz"