#!/bin/bash

#SBATCH --job-name=videomae100
#SBATCH --account=st-ilker-1-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00

set -euo pipefail

CODE_DIR="${CODE_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"

if [[ ! -f "$CODE_DIR/run_mae_pretraining.py" ]]; then
    echo "ERROR: run_mae_pretraining.py not found in $CODE_DIR"
    exit 1
fi

module load gcc/9.4.0
module load miniconda3/4.9.2
source "$(conda info --base)/etc/profile.d/conda.sh"

VIDEOMAE_ENV="${VIDEOMAE_ENV:-videomae}"
conda activate "$VIDEOMAE_ENV"

DATA_ROOT="${DATA_ROOT:?Please set DATA_ROOT to the COVID-BLUeS dataset root}"
SPLIT_FILE="${SPLIT_FILE:-$CODE_DIR/all_videomae.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-$CODE_DIR/outputs/covid_blues_all362_videomae_100e_seed0}"

RUNTIME_SPLIT="${SLURM_TMPDIR:-/tmp}/all_videomae_${SLURM_JOB_ID}.txt"

export MASTER_ADDR="$(hostname)"
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

mkdir -p "$OUTPUT_DIR"

awk -v root="$DATA_ROOT" '{
    print root "/" $1, $2
}' "$SPLIT_FILE" > "$RUNTIME_SPLIT"

cd "$CODE_DIR"

python run_mae_pretraining.py \
  --data_path "$RUNTIME_SPLIT" \
  --model pretrain_videomae_small_patch16_224 \
  --decoder_depth 4 \
  --mask_type tube \
  --mask_ratio 0.9 \
  --num_frames 16 \
  --sampling_rate 2 \
  --batch_size 32 \
  --epochs 100 \
  --warmup_epochs 10 \
  --lr 0.00015 \
  --weight_decay 0.05 \
  --seed 0 \
  --save_ckpt_freq 20 \
  --num_workers 8 \
  --opt adamw \
  --opt_betas 0.9 0.95 \
  --output_dir "$OUTPUT_DIR"
