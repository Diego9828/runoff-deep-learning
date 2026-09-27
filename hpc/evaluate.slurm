#!/usr/bin/env bash
#SBATCH --job-name=dmel-eval
#SBATCH --partition=standard
#SBATCH --qos=a-postgrado
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
# Recalcula métricas y figuras con las corridas terminadas en runs/. Uso: sbatch hpc/evaluate.slurm
set -euo pipefail

export NUMBA_NUM_THREADS=${SLURM_CPUS_PER_TASK}
uv run --no-sync dmel-evaluate
