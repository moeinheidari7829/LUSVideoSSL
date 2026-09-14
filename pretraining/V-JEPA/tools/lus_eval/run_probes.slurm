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
module load gcc/9.4.0
module load miniconda3/4.9.2
export PS1=""
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /arc/project/st-ilker-1/junbo2/envs/jepa

TOOLS=/arc/project/st-ilker-1/junbo2/code/jepa/tools/lus_eval
FEATURES=/arc/project/st-ilker-1/junbo2/results/lus_diagnostic/features
RESULTS=/scratch/st-ilker-1/junbo2/lus_diagnostic/results

mkdir -p "$RESULTS"
cd "$TOOLS"
python run_probes.py --feature-root "$FEATURES" --output-dir "$RESULTS"
