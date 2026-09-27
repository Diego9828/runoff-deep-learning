"""CLI: sample entropy de los componentes CEEMDAN de train y agrupación MRS en alta/baja frecuencia."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dmel.data import load_discharge, load_h5
from dmel.mrs import build_mrs
from dmel.plots import entropy_bars


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data")
    p.add_argument("--artifacts", default="artifacts")
    p.add_argument("--windows", type=int, default=8000, help="ventanas de train usadas para las entropías")
    p.add_argument("--tau", type=float, default=0.1, help="umbral de la banda de SE 'casi tendencia' (paper: 0-0.1)")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    art = Path(a.artifacts)
    comps = np.load(art / "components_train.npy", mmap_mode="r")
    h5 = Path(a.data) / "train.h5"
    split = load_h5(h5, len(comps), fields=("split",))["split"]
    q = load_discharge(h5, len(comps))

    rng = np.random.default_rng(a.seed)
    train_rows = np.flatnonzero(split == 0)  # solo train: la validación no participa en el MRS
    rows = np.sort(rng.choice(train_rows, min(a.windows, len(train_rows)), replace=False))
    mrs = build_mrs(q[rows], np.asarray(comps[rows]), a.tau)

    (art / "mrs.json").write_text(json.dumps(mrs, indent=2))
    table = pd.DataFrame({
        "componente": mrs["components"],
        "presente": mrs["component_present"],
        "SE_mediana": mrs["component_se_median"],
        "voto_SE>SE_orig": mrs["component_votes"],
    })
    table.to_csv(art / "se_table.csv", index=False)
    entropy_bars(mrs, art / "mrs_entropy.png")

    print(table.round(3).to_string(index=False))
    print(f"SE mediana de la serie original: {mrs['se_orig_median']:.3f}")
    for g, label, vote, se in zip(mrs["groups"], mrs["labels"], mrs["group_votes"], mrs["group_se_median"]):
        names = " + ".join(mrs["components"][k] for k in g)
        print(f"RIMF {names:<28} SE={se:.3f} voto={vote:.2f} -> {'alta' if label == 'high' else 'baja'} frecuencia")


if __name__ == "__main__":
    main()
