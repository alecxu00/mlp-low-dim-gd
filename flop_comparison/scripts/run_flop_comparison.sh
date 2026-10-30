#!/bin/bash

#SBATCH --job-name=flop_comparison
#SBATCH --account=girasole1
#SBATCH --partition=spgpu
#SBATCH --gpus=1
#SBATCH --ntasks-per-gpu=1
#SBATCH --mem-per-gpu=8G
#SBATCH --time=00-01:00:00
#SBATCH --output=../logs/flop_comparison.log

# Run from inside flop_comparison/scripts/ (paths are relative, like mlp/scripts/).
# Works both via `sbatch run_flop_comparison.sh` and plain `bash run_flop_comparison.sh`.

set -e

mkdir -p ../logs

python3 -m pytest ../tests -q
python3 ../compare.py --save_fig --out_dir ../results/

# Use the GPU when available; otherwise fall back to the --small config, which is
# equally exact for the FLOP comparisons.
if python3 -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
    python3 ../torch_check.py --device cuda
else
    echo "No GPU detected: running torch_check at the --small config."
    python3 ../torch_check.py --small --device cpu
fi
