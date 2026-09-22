#!/bin/bash

#SBATCH --job-name=mocov3_100e
#SBATCH --account=st-ilker-1-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00

set -euo pipefail

CODE_DIR="${CODE_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"

if [[ ! -f "$CODE_DIR/main_moco_video.py" ]]; then
    echo "ERROR: main_moco_video.py not found in $CODE_DIR"
    exit 1
fi

module load gcc/9.4.0
module load miniconda3/4.9.2
source "$(conda info --base)/etc/profile.d/conda.sh"

MOCOV3_ENV="${MOCOV3_ENV:-mocov3}"
conda activate "$MOCOV3_ENV"

# CSV with columns: video_path,patient_id,video_id -- see make_video_manifest.py.
# Build it once with:
#   python make_video_manifest.py --video_dir "$DATA_ROOT/lus_videos" --manifest "$MANIFEST"
MANIFEST="${MANIFEST:?Please set MANIFEST to the COVID-BLUeS video manifest CSV}"
OUTPUT_DIR="${OUTPUT_DIR:-$CODE_DIR/outputs/covid_blues_mocov3_100e_seed0}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$OUTPUT_DIR/checkpoints}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

mkdir -p "$CHECKPOINT_DIR"

cd "$CODE_DIR"

python main_moco_video.py \
  --manifest "$MANIFEST" \
  --clip-len 16 \
  --tubelet-size 2 \
  --batch-size 4 \
  --epochs 100 \
  --lr 1.5e-4 \
  --checkpoint-dir "$CHECKPOINT_DIR"
