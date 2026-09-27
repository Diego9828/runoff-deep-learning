"""Carga de datos, escalado por cuenca y construcción de las entradas de cada rama."""

import json
from pathlib import Path

import h5py
import numpy as np

TARGET = 11  # canal del caudal (metadata.json: target_channel); los canales 0-10 son meteorológicos
HIST, HORIZON = 336, 48
EPS = 0.01  # mm/h: piso de la escala por cuenca (mismo rol que el epsilon de la pérdida NSE*)


def load_h5(path, limit=None, fields=("X", "y", "basin_id", "split")):
    """Lee arreglos completos del .h5 con lecturas contiguas (indexar muestra por muestra es ~50x más lento)."""
    with h5py.File(path, "r") as f:
        n = len(f["X"]) if limit is None else min(limit, len(f["X"]))
        return {name: f[name][:n] for name in fields if name in f}


def load_discharge(path, limit=None, chunk=20_000):
    """Solo la historia de caudal [N, 336] (para la descomposición)."""
    with h5py.File(path, "r") as f:
        n = len(f["X"]) if limit is None else min(limit, len(f["X"]))
        return np.concatenate([f["X"][s : min(s + chunk, n), :, TARGET] for s in range(0, n, chunk)])


def take(d, mask):
    """Subconjunto de filas de todos los arreglos de un diccionario."""
    return {k: v[mask] for k, v in d.items()}


def basin_scales(q, basin, n_basins):
    """Escala s_b = σ_b + EPS y media μ_b del caudal histórico de cada cuenca (solo ventanas de train).

    Las cuencas sin ventanas de train (solo pasa con --limit) reciben la mediana de las demás.
    """
    q = q.astype(np.float64)
    count = np.bincount(basin, minlength=n_basins) * q.shape[1]
    s1 = np.bincount(basin, weights=q.sum(1), minlength=n_basins)
    s2 = np.bincount(basin, weights=(q**2).sum(1), minlength=n_basins)
    seen = count > 0
    mu = np.where(seen, s1 / np.maximum(count, 1), 0.0)
    sigma = np.sqrt(np.maximum(s2 / np.maximum(count, 1) - mu**2, 0.0))
    mu[~seen], sigma[~seen] = np.median(mu[seen]), np.median(sigma[seen])
    return (sigma + EPS).astype(np.float32), mu.astype(np.float32)


def load_groups(path):
    """Grupos MRS (índices de componentes CEEMDAN por RIMF) separados en alta y baja frecuencia."""
    mrs = json.loads(Path(path).read_text())
    high = [g for g, label in zip(mrs["groups"], mrs["labels"]) if label == "high"]
    low = [g for g, label in zip(mrs["groups"], mrs["labels"]) if label == "low"]
    return high, low


def rimfs(comps, groups):
    """Suma los componentes de cada grupo: [N, K+1, T] -> [N, T, n_grupos]."""
    return np.stack([comps[:, g].sum(1) for g in groups], axis=-1)


def make_inputs(keys, X, comps, basin, scale, mu, groups=((), ())):
    """Entradas (aún sin normalizar) de cada rama: dict clave -> [N, 336, C] float32.

    Canales: [caudal o RIMFs] / s_b  +  11 covariables  +  2 estáticos (log s_b, log μ_b).
      raw: caudal observado (sin descomposición)
      hf / lf: RIMFs de alta / baja frecuencia del MRS (groups = (high, low))
    """
    s = scale[basin][:, None, None]
    met = X[:, :, :TARGET]
    static = np.stack([np.log(scale[basin]), np.log(mu[basin] + EPS)], axis=-1).astype(np.float32)
    static = np.broadcast_to(static[:, None, :], (len(X), X.shape[1], 2))
    out = {}
    for key in keys:
        if key == "raw":
            q = X[:, :, TARGET : TARGET + 1] / s
        else:
            high, low = groups
            q = rimfs(comps, high if key == "hf" else low) / s
        out[key] = np.concatenate([q, met, static], axis=-1, dtype=np.float32)
    return out


def fit_norm(arr, max_rows=20_000, seed=0):
    """Media y desviación por canal, estimadas en una muestra de ventanas de train."""
    rng = np.random.default_rng(seed)
    rows = np.sort(rng.choice(len(arr), min(max_rows, len(arr)), replace=False))
    sample = arr[rows].reshape(-1, arr.shape[-1]).astype(np.float64)
    mean, std = sample.mean(0), sample.std(0)
    std[std == 0] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def normalize(arr, mean, std):
    """z-score por canal, en el mismo arreglo para no duplicar memoria."""
    arr -= mean
    arr /= std
    return arr


def batches(n, batch_size, rng):
    """Índices aleatorios de cada batch de una época."""
    order = rng.permutation(n)
    for start in range(0, n, batch_size):
        yield order[start : start + batch_size]
