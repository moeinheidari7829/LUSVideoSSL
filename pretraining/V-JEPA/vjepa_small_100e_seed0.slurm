#!/bin/bash
#SBATCH --job-name=vjepa100e_s0
#SBATCH --account=st-ilker-1-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=vjepa100e_s0-%j.out
#SBATCH --error=vjepa100e_s0-%j.err

set -euo pipefail

# Submit from pretraining/V-JEPA, or set CODE_DIR to that directory.
CODE_DIR="${CODE_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$CODE_DIR"
export CODE_DIR="$(pwd)"
export DATA_ROOT="${DATA_ROOT:?Set DATA_ROOT to the COVID-BLUeS dataset root}"
export OUTPUT_DIR="${OUTPUT_DIR:-$CODE_DIR/outputs/vjepa_small_100e_seed0}"

module load gcc/9.4.0
module load miniconda3/4.9.2
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${JEPA_ENV:-jepa}"

# Stage only runtime inputs; leave the source config and manifest untouched.
export RUNTIME_DIR="$(mktemp -d "${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/vjepa.XXXXXX")"
trap 'rm -rf -- "$RUNTIME_DIR"' EXIT
python - <<'PY'
import csv
import os
from pathlib import Path
import yaml

code = Path(os.environ['CODE_DIR'])
runtime = Path(os.environ['RUNTIME_DIR'])
relative_config = Path('configs/custom/vits16_covid_100e_seed0.yaml')
config = yaml.safe_load((code / relative_config).read_text())
root = Path(os.environ['DATA_ROOT']).expanduser().resolve(strict=True)
with (code / config['data']['datasets'][0]).open(newline='') as source:
    rows = list(csv.reader(source, delimiter=' '))
if len(rows) != 362:
    raise ValueError(f'Expected 362 rows, found {len(rows)}')
manifest = runtime / 'covid_blues_all.csv'
with manifest.open('w', newline='') as target:
    writer = csv.writer(target, delimiter=' ', lineterminator='\n')
    for path, label in rows:
        video = root / path
        if not video.is_file():
            raise FileNotFoundError(video)
        writer.writerow([str(video), label])
output = Path(os.environ['OUTPUT_DIR']).expanduser().resolve()
output.mkdir(parents=True, exist_ok=True)
config['data']['datasets'] = [str(manifest)]
config['logging']['folder'] = str(output)
destination = runtime / relative_config
destination.parent.mkdir(parents=True, exist_ok=True)
with destination.open('w') as target:
    yaml.safe_dump(config, target)
PY

# Keep the original module, config filename, and device arguments.
export PYTHONPATH="$CODE_DIR${PYTHONPATH:+:$PYTHONPATH}"
cd "$RUNTIME_DIR"

echo "Node: $(hostname)"
nvidia-smi
python -c "import torch, decord; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0)); print('Decord:', decord.__version__)"

python -m app.main \
  --fname configs/custom/vits16_covid_100e_seed0.yaml \
  --devices cuda:0

