"""CLI: entrena una variante, elige la mejor época con validación y predice validación y test."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from dmel.data import (HORIZON, SOURCES, TARGET, basin_scales, batches, fit_norm, load_groups, load_h5, make_inputs,
                       normalize)
from dmel.metrics import nse, per_basin
from dmel.models import VARIANTS, build_model, input_keys


def load_sets(a, keys):
    """Entradas normalizadas, objetivos escalados (y / s_b) y cuencas de train, val y test."""
    data, art = Path(a.data), Path(a.artifacts)
    full = load_h5(data / "train.h5", a.limit)
    test = load_h5(data / "test.h5", a.limit, fields=("X", "basin_id"))
    uses_mrs = any(src in ("high", "low") for k in keys for src in SOURCES[k])
    groups = load_groups(art / "mrs.json") if uses_mrs else ((), ())

    def components(name, n):
        return np.load(art / f"components_{name}.npy", mmap_mode="r")[:n] if uses_mrs else None

    is_train = full["split"] == 0
    n_basins = int(max(full["basin_id"].max(), test["basin_id"].max())) + 1
    scale, mu = basin_scales(full["X"][is_train, :, TARGET], full["basin_id"][is_train], n_basins)

    comps, comps_test = components("train", len(full["X"])), components("test", len(test["X"]))
    parts = {
        "train": (full, comps, is_train),
        "val": (full, comps, ~is_train),
        "test": (test, comps_test, np.ones(len(test["X"]), bool)),
    }
    sets = {}
    for name, (src, c, rows) in parts.items():
        basin = src["basin_id"][rows]
        c_rows = None if c is None else np.asarray(c[rows])
        sets[name] = {"x": make_inputs(keys, src["X"][rows], c_rows, basin, scale, mu, groups), "basin": basin}
        if "y" in src:
            sets[name]["y"] = (src["y"][rows] / scale[basin][:, None]).astype(np.float32)
    del full, test

    norm = {k: fit_norm(sets["train"]["x"][k]) for k in keys}  # estadísticas solo de train
    for s in sets.values():
        for k in keys:
            normalize(s["x"][k], *norm[k])
    return sets, scale, norm


def to_device(inputs, rows, device):
    return {k: torch.from_numpy(v[rows]).to(device, non_blocking=True) for k, v in inputs.items()}


@torch.no_grad()
def predict(model, inputs, device, batch=2048):
    """Predicción escalada: aporte de cada rama [N, n_ramas, 48] (la predicción es su suma)."""
    model.eval()
    n = len(next(iter(inputs.values())))
    parts = []
    with torch.random.fork_rng():  # ProbSparse muestrea claves: predicción reproducible sin tocar el RNG
        torch.manual_seed(0)
        for s in range(0, n, batch):
            xb = to_device(inputs, slice(s, s + batch), device)
            parts.append(torch.stack(model.branch_outputs(xb), dim=1).float().cpu().numpy())
    return np.concatenate(parts)


def write_submission(pred, path):
    """CSV con el mismo formato que test_targets.csv: Id, q_01..q_48 (mm/h)."""
    df = pd.DataFrame({f"q_{h:02d}": pred[:, h - 1] for h in range(1, HORIZON + 1)})
    df.insert(0, "Id", np.arange(len(pred)))
    df.to_csv(path, index=False, float_format="%.8g")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=list(VARIANTS), required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data", default="data")
    p.add_argument("--artifacts", default="artifacts")
    p.add_argument("--runs", default="runs")
    p.add_argument("--predictions", default="predictions")
    p.add_argument("--epochs", type=int, default=100, help="épocas de entrenamiento (paper: 100)")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=500, help="pasos de warmup lineal del learning rate")
    p.add_argument("--patience", type=int, default=None,
                   help="early stopping: épocas sin mejorar en validación antes de parar (por defecto no se para)")
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--limit", type=int, help="solo las primeras N filas de cada .h5 (pruebas rápidas)")
    a = p.parse_args()

    run = Path(a.runs) / f"{a.model}_s{a.seed}"
    if (run / "done").exists():
        print(f"{run} ya está completo; se omite.")
        return
    run.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True  # TF32 en la A100: matmuls más rápidos, precisión suficiente
    torch.backends.cudnn.allow_tf32 = True

    keys = input_keys(a.model)
    t_load = time.time()
    sets, scale, norm = load_sets(a, keys)
    train_x, train_y = sets["train"]["x"], sets["train"]["y"]
    n_train = len(train_y)
    channels = {k: train_x[k].shape[-1] for k in keys}
    model = build_model(a.model, channels).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{a.model}: {n_params / 1e6:.2f} M parámetros | canales {channels} | {n_train} ventanas de train | "
          f"datos listos en {time.time() - t_load:.0f} s | {device} "
          f"{torch.cuda.get_device_name() if device.type == 'cuda' else ''}", flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    best, bad_epochs, step, history = np.inf, 0, 0, []
    for epoch in range(1, a.epochs + 1):
        model.train()
        t0, total = time.time(), torch.zeros((), device=device)
        for rows in batches(n_train, a.batch, rng):
            if step < a.warmup:
                for group in opt.param_groups:
                    group["lr"] = a.lr * (step + 1) / a.warmup
            loss = F.mse_loss(model(to_device(train_x, rows, device)), torch.from_numpy(train_y[rows]).to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
            opt.step()
            total += loss.detach() * len(rows)
            step += 1
            if step == 50:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                print(f"  {(time.time() - t0) / 50 * 1000:.0f} ms/paso en los primeros 50 pasos", flush=True)

        val_scaled = predict(model, sets["val"]["x"], device).sum(1)
        val_loss = float(np.mean((val_scaled - sets["val"]["y"]) ** 2))  # NSE* en validación
        val_nse = float(np.nanmedian(list(per_basin(sets["val"]["y"], val_scaled, sets["val"]["basin"], nse).values())))
        plateau.step(val_loss)
        history.append({"epoch": epoch, "train_loss": total.item() / n_train, "val_loss": val_loss,
                        "val_nse_mediana": val_nse, "lr": opt.param_groups[0]["lr"], "segundos": time.time() - t0})
        print(f"época {epoch:2d} | train {history[-1]['train_loss']:.4f} | val {val_loss:.4f} | "
              f"NSE val (mediana) {val_nse:.3f} | {history[-1]['segundos']:.0f} s", flush=True)
        pd.DataFrame(history).to_csv(run / "history.csv", index=False)
        if val_loss < best:
            best, bad_epochs = val_loss, 0
            torch.save(model.state_dict(), run / "best.pt")
        else:
            bad_epochs += 1
            if a.patience is not None and bad_epochs >= a.patience:
                print(f"Early stopping: {a.patience} épocas sin mejorar en validación.")
                break

    # predicción con la mejor época: se desescala a mm/h y se recorta en 0 (no hay caudal negativo)
    model.load_state_dict(torch.load(run / "best.pt", map_location=device))
    for name in ("val", "test"):
        parts = predict(model, sets[name]["x"], device) * scale[sets[name]["basin"]][:, None, None]
        np.save(run / f"{name}_branches.npy", parts.astype(np.float32))
        np.save(run / f"{name}_pred.npy", np.maximum(parts.sum(1), 0).astype(np.float32))
    Path(a.predictions).mkdir(parents=True, exist_ok=True)
    write_submission(np.load(run / "test_pred.npy"), Path(a.predictions) / f"{run.name}_test.csv")

    best_epoch = int(np.argmin([h["val_loss"] for h in history])) + 1
    info = {**vars(a), "parametros": n_params, "canales": channels, "mejor_epoca": best_epoch,
            "val_loss": best, "norm": {k: [m.tolist(), s.tolist()] for k, (m, s) in norm.items()}}
    (run / "config.json").write_text(json.dumps(info, indent=2))
    (run / "done").touch()
    print(f"Listo: {run} (mejor época {best_epoch})")


if __name__ == "__main__":
    main()
