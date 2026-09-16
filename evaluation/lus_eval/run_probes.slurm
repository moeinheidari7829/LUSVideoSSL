#!/bin/bash

#SBATCH --job-name=lus_probes
#SBATCH --account=st-ilker-1-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

set -euo pipefail

# Locate this repository automatically.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

module load gcc/9.4.0
module load miniconda3/4.9.2

export PS1=""
source "$(conda info --base)/etc/profile.d/conda.sh"

# Can be either a Conda environment name or an absolute environment path.
EVAL_ENV="${EVAL_ENV:-jepa}"
conda activate "$EVAL_ENV"

# Default locations relative to the repository.
# These can be overridden with environment variables.
FEATURES="${FEATURES:-$REPO_ROOT/outputs/features}"
RESULTS="${RESULTS:-$REPO_ROOT/outputs/evaluation}"

# The feature root must contain:
#   videomae_small/
#       pocus_diagnostic.npz
#       mendeley_uganda_diagnostic.npz
#   vjepa_small/
#       pocus_diagnostic.npz
#       mendeley_uganda_diagnostic.npz
if [[ ! -d "$FEATURES/videomae_small" ]]; then
    echo "ERROR: VideoMAE features not found: $FEATURES/videomae_small"
    exit 1
fi

if [[ ! -d "$FEATURES/vjepa_small" ]]; then
    echo "ERROR: V-JEPA features not found: $FEATURES/vjepa_small"
    exit 1
fi

mkdir -p "$RESULTS"

cd "$SCRIPT_DIR"

python run_probes.py \
    --feature-root "$FEATURES" \
    --output-dir "$RESULTS"