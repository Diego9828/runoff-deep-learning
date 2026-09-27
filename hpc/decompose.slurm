#!/usr/bin/env bash
#SBATCH --job-name=dmel-ceemdan
#SBATCH --partition=standard
#SBATCH --qos=a-postgrado
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
# CEEMDAN de todas las ventanas de train/val y test (solo CPU) y agrupación MRS.
# Uso: sbatch hpc/decompose.slurm
set -euo pipefail

echo "commit $(cat REVISION 2>/dev/null) | $(hostname) | ${SLURM_CPUS_PER_TASK} CPUs"
export NUMBA_NUM_THREADS=${SLURM_CPUS_PER_TASK}
uv run --no-sync dmel-decompose --split train
uv run --no-sync dmel-decompose --split test
uv run --no-sync dmel-mrs
