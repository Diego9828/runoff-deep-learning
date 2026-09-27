#!/usr/bin/env bash
# Trae de KHIPU los resultados livianos: tablas y figuras, predicciones de test, historias de
# entrenamiento, configuración de cada corrida, salidas del MRS y logs de Slurm.
# Uso, desde la raíz del repo:  bash hpc/pull.sh
set -euo pipefail

REMOTE=${REMOTE:-khipu}
DEST=${DEST:-runoff_deep_learning}

ssh "$REMOTE" "cd $DEST && tar --ignore-failed-read -cf - results predictions logs REVISION \
    artifacts/mrs.json artifacts/se_table.csv artifacts/mrs_entropy.png \
    runs/*/history.csv runs/*/config.json 2>/dev/null" | tar -xf -
echo "Resultados copiados desde $REMOTE:~/$DEST (commit $(cat REVISION 2>/dev/null))"
