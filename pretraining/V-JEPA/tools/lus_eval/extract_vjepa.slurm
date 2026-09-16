#!/bin/bash
#SBATCH --job-name=lus_vjepa_feat
#SBATCH --account=st-ilker-1-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/arc/project/st-ilker-1/junbo2/logs/lus_vjepa_feat_%j.out
#SBATCH --error=/arc/project/st-ilker-1/junbo2/logs/lus_vjepa_feat_%j.err

set -euo pipefail
module load gcc/9.4.0
module load miniconda3/4.9.2
export PS1=""
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /arc/project/st-ilker-1/junbo2/envs/jepa

TOOLS=/arc/project/st-ilker-1/junbo2/code/jepa/tools/lus_eval
MANIFESTS=/arc/project/st-ilker-1/junbo2/data/lus_diagnostic/manifests
FEATURES=/scratch/st-ilker-1/junbo2/lus_diagnostic/features/vjepa_small
CHECKPOINT=/scratch/st-ilker-1/moein/junbo2_backup/vjepa_small_100e_seed0/vjepa_small_100e_seed0-latest.pth.tar

mkdir -p "$FEATURES"
cd "$TOOLS"
python extract_vjepa_features.py --manifest "$MANIFESTS/pocus_diagnostic_3class.csv" --checkpoint "$CHECKPOINT" --output "$FEATURES/pocus_diagnostic.npz"
python extract_vjepa_features.py --manifest "$MANIFESTS/mendeley_uganda_diagnostic_3class.csv" --checkpoint "$CHECKPOINT" --output "$FEATURES/mendeley_uganda_diagnostic.npz"

