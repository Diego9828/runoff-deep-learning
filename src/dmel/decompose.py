"""CLI: CEEMDAN de cada ventana usando solo sus 336 h de historia (sin información del futuro)."""

import argparse
import time
from pathlib import Path

import numpy as np

from dmel.ceemdan import decompose_all, noise_bank
from dmel.data import HIST, load_discharge


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", choices=["train", "test"], required=True,
                   help="train = todas las filas de train.h5 (train y validación)")
    p.add_argument("--data", default="data")
    p.add_argument("--out", default="artifacts")
    p.add_argument("--imfs", type=int, default=6, help="número máximo de IMFs (más el residuo)")
    p.add_argument("--trials", type=int, default=100, help="realizaciones de ruido (paper: 100)")
    p.add_argument("--eps", type=float, default=0.2, help="desviación relativa del ruido (paper: 0.2)")
    p.add_argument("--sifts", type=int, default=10, help="iteraciones de sifting por IMF")
    p.add_argument("--chunk", type=int, default=10_000)
    p.add_argument("--limit", type=int, help="solo las primeras N ventanas (benchmark / pruebas)")
    a = p.parse_args()

    q = load_discharge(Path(a.data) / f"{a.split}.h5", a.limit)
    bank = noise_bank(a.trials, HIST, a.imfs, a.sifts, seed=0)  # el mismo banco para train y test
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = np.lib.format.open_memmap(out_dir / f"components_{a.split}.npy", mode="w+", dtype=np.float32,
                                    shape=(len(q), a.imfs + 1, HIST))

    decompose_all(q[:2], bank, a.eps, a.sifts)  # compila con numba antes de medir el tiempo
    start = time.time()
    for s in range(0, len(q), a.chunk):
        e = min(s + a.chunk, len(q))
        out[s:e] = decompose_all(q[s:e], bank, a.eps, a.sifts)
        per_window = (time.time() - start) / e
        print(f"{e}/{len(q)} ventanas | {1000 * per_window:.2f} ms/ventana | "
              f"faltan {per_window * (len(q) - e) / 60:.1f} min", flush=True)
    out.flush()

    n = min(len(q), 1000)
    error = np.abs(out[:n].sum(1) - q[:n]).max()
    print(f"Listo: {out_dir / f'components_{a.split}.npy'} {out.shape}. "
          f"Error de reconstrucción (primeras {n} ventanas): {error:.2e} mm/h")


if __name__ == "__main__":
    main()
