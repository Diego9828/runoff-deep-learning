#!/usr/bin/env bash
#SBATCH --job-name=dmel-train
#SBATCH --partition=gpu
#SBATCH --qos=a-postgrado
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
# Entrena DMEL y las dos ablaciones en secuencia en la A100 (1 GPU por usuario) y luego evalúa.
# Uso: sbatch --dependency=afterok:<id_ceemdan> hpc/train.slurm
#      MODELS="dmel" SEED=1 sbatch hpc/train.slurm
# Las corridas terminadas (runs/<modelo>_s<seed>/done) se omiten, así que se puede reenviar sin perder nada.
set -uo pipefail

echo "commit $(cat REVISION 2>/dev/null) | $(hostname)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMBA_NUM_THREADS=${SLURM_CPUS_PER_TASK}

for model in ${MODELS:-dmel dmel_swap hybrid_raw}; do
    uv run --no-sync dmel-train --model "$model" --seed "${SEED:-0}" || echo "ERROR: falló $model"
done
uv run --no-sync dmel-evaluate
