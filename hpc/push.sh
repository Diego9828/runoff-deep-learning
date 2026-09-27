#!/usr/bin/env bash
# Sube el commit actual (git archive) a KHIPU e instala el entorno con uv.
# Uso, desde la raíz del repo:  bash hpc/push.sh
# Los datos se suben una sola vez:  scp data/{train.h5,test.h5,test_targets.csv} khipu:runoff_deep_learning/data/
set -euo pipefail

REMOTE=${REMOTE:-khipu}
DEST=${DEST:-runoff_deep_learning}
REV=$(git rev-parse --short HEAD)
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
    echo "Aviso: hay cambios sin commitear; se sube solo el commit $REV."
fi

git archive --format=tar HEAD | ssh "$REMOTE" "mkdir -p $DEST && tar -xf - -C $DEST && cd $DEST \
    && echo $REV > REVISION && mkdir -p logs && ~/.local/bin/uv sync --frozen"
echo "KHIPU:~/$DEST actualizado al commit $REV"
