"""CLI: métricas de validación y test, persistencia, Wilcoxon por cuenca y figuras para el informe.

Es el único script que lee test_targets.csv (solo para el reporte final, nunca para elegir modelos).
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from dmel import metrics
from dmel.ceemdan import decompose_all, noise_bank
from dmel.data import HIST, HORIZON, load_discharge, load_h5

LABELS = {
    "persistence": "Persistencia",
    "dmel_s0": "DMEL (paper)",
    "dmel_swap_s0": "Ablación: asignación invertida",
    "hybrid_raw_s0": "Ablación: sin MRS",
}


def label(name):
    return LABELS.get(name, name)


def load_truth(data, limit):
    """Observado, cuenca e historia de caudal de validación y test."""
    full = load_h5(data / "train.h5", limit, fields=("y", "basin_id", "split"))
    hist = load_discharge(data / "train.h5", limit)
    val = full["split"] == 1
    test_basin = load_h5(data / "test.h5", limit, fields=("basin_id",))["basin_id"]
    test_y = pd.read_csv(data / "test_targets.csv").sort_values("Id").iloc[: len(test_basin), 1:].to_numpy(np.float32)
    return {
        "val": {"obs": full["y"][val], "basin": full["basin_id"][val], "hist": hist[val]},
        "test": {"obs": test_y, "basin": test_basin, "hist": load_discharge(data / "test.h5", limit)},
    }


def load_predictions(runs, split, hist):
    """Persistencia (último caudal observado repetido 48 h) + las corridas terminadas."""
    preds = {"persistence": np.repeat(hist[:, -1:], HORIZON, axis=1)}
    for run in sorted(Path(runs).glob("*/done")):
        preds[run.parent.name] = np.load(run.parent / f"{split}_pred.npy")
    return preds


def wilcoxon_table(obs, basin, preds, ref):
    """Wilcoxon pareado sobre el NSE de cada cuenca: `ref` contra cada modelo."""
    nse_b = {n: pd.Series(metrics.per_basin(obs, p, basin, metrics.nse)) for n, p in preds.items()}
    rows = []
    for name in preds:
        if name == ref:
            continue
        both = pd.concat([nse_b[ref], nse_b[name]], axis=1).dropna()
        diff = both.iloc[:, 0] - both.iloc[:, 1]
        rows.append({"comparación": f"{label(ref)} vs {label(name)}", "mediana ΔNSE": diff.median(),
                     "cuencas donde gana": f"{int((diff > 0).sum())}/{len(diff)}",
                     "p-valor": wilcoxon(both.iloc[:, 0], both.iloc[:, 1]).pvalue})
    return pd.DataFrame(rows).set_index("comparación"), nse_b


def markdown(df, digits=3):
    """Tabla markdown simple (evita depender de tabulate)."""
    def fmt(v):
        return f"{v:.{digits}g}" if isinstance(v, float) and abs(v) < 1e-2 else (f"{v:.{digits}f}" if isinstance(v, float) else str(v))
    lines = ["| " + " | ".join([""] + [str(c) for c in df.columns]) + " |", "|" + "---|" * (len(df.columns) + 1)]
    lines += ["| " + " | ".join([str(i)] + [fmt(v) for v in row]) + " |" for i, row in zip(df.index, df.to_numpy())]
    return "\n".join(lines)


def plot_leads(lead_nse, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, values in lead_nse.items():
        ax.plot(np.arange(1, HORIZON + 1), values, label=label(name), ls="--" if name == "persistence" else "-")
    ax.set(xlabel="Horizonte (h)", ylabel="NSE (mediana entre cuencas)", title="Test: NSE por horizonte")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_cdf(nse_b, path):
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, values in nse_b.items():
        v = np.sort(np.clip(values.dropna().to_numpy(), -1, 1))
        ax.plot(v, np.linspace(0, 1, len(v)), label=label(name), ls="--" if name == "persistence" else "-")
    ax.set(xlabel="NSE por cuenca (recortado en -1)", ylabel="Fracción de cuencas", title="Test: distribución del NSE")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def pick_events(split, n=4):
    """Ventanas de test con las mayores subidas de caudal, en cuencas distintas."""
    rise = (split["obs"].max(1) - split["hist"][:, -1]) / (split["hist"].std(1) + 1e-3)
    events, seen = [], set()
    for i in np.argsort(rise)[::-1]:
        if split["basin"][i] not in seen:
            events.append(i)
            seen.add(split["basin"][i])
        if len(events) == n:
            break
    return events


def plot_events(split, preds, branches, events, path):
    """Hidrogramas de eventos de crecida y aporte de cada rama de DMEL (fila inferior)."""
    fig, axes = plt.subplots(2, len(events), figsize=(4 * len(events), 6.5), squeeze=False)
    t_hist, t_future = np.arange(-72, 0), np.arange(0, HORIZON)
    for col, i in enumerate(events):
        top, bottom = axes[0, col], axes[1, col]
        for ax in (top, bottom):
            ax.plot(t_hist, split["hist"][i, -72:], color="k", lw=1)
            ax.plot(t_future, split["obs"][i], color="k", lw=2, label="observado")
        for name, p in preds.items():
            top.plot(t_future, p[i], ls="--" if name == "persistence" else "-", label=label(name))
        if branches is not None:
            bottom.plot(t_future, branches[i].sum(0), color="tab:purple", label="DMEL = suma")
            bottom.plot(t_future, branches[i, 0], color="tab:red", label="rama HF (Informer)")
            bottom.plot(t_future, branches[i, 1], color="tab:blue", label="rama LF (LSTM)")
        top.set_title(f"cuenca {split['basin'][i]}, ventana {i}")
        bottom.set_xlabel("horas desde el pronóstico")
    axes[0, 0].set_ylabel("caudal (mm/h)")
    axes[1, 0].set_ylabel("caudal (mm/h)")
    axes[0, 0].legend(fontsize=7)
    axes[1, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_regimes(split, preds, path):
    """Error absoluto relativo (Σ|s - o| / Σo) por régimen de caudal (terciles de cada cuenca)."""
    obs, basin = split["obs"], split["basin"]
    regime = np.zeros(obs.shape, int)
    for b in np.unique(basin):
        rows = basin == b
        regime[rows] = np.digitize(obs[rows], np.quantile(obs[rows], [1 / 3, 2 / 3]))
    names = ["bajo", "medio", "alto"]
    table = pd.DataFrame({label(n): [100 * np.abs(p - obs)[regime == r].sum() / obs[regime == r].sum() for r in range(3)]
                          for n, p in preds.items()}, index=names)
    ax = table.plot.bar(figsize=(7, 4), rot=0)
    ax.set(ylabel="error absoluto relativo (%)", xlabel="régimen de caudal", title="Test: error por régimen de caudal")
    ax.figure.tight_layout()
    ax.figure.savefig(path, dpi=150)
    plt.close(ax.figure)
    return table


def leakage_study(split, path, n_windows=40):
    """¿Cuánto cambian los componentes si se descompone también el futuro (como en el paper)?

    Compara la descomposición de la historia (336 h) con la de historia + futuro (384 h) en las últimas 48 h de
    historia. Control: dos descomposiciones de la misma historia con bancos de ruido distintos.
    """
    rows = pick_events(split, n_windows)
    hist = split["hist"][rows].astype(np.float64)
    both = np.concatenate([hist, split["obs"][rows]], axis=1)
    base = decompose_all(hist, noise_bank(100, HIST, 6, seed=0), 0.2, 10)
    control = decompose_all(hist, noise_bank(100, HIST, 6, seed=1), 0.2, 10)
    future = decompose_all(both, noise_bank(100, HIST + HORIZON, 6, seed=0), 0.2, 10)[:, :, :HIST]

    def change(other):  # RMS del cambio en las últimas 48 h / std del componente en toda la ventana
        rms = np.sqrt(np.mean((other[:, :, -HORIZON:] - base[:, :, -HORIZON:]) ** 2, axis=2))
        return np.median(rms / (base.std(2) + 1e-9), axis=0)

    names = [f"IMF{k + 1}" for k in range(6)] + ["residuo"]
    table = pd.DataFrame({"descomponiendo también el futuro": change(future), "control (otro ruido)": change(control)},
                         index=names)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
    table.plot.bar(ax=ax1, rot=0)
    ax1.set(ylabel="cambio RMS / std del componente", title="Últimas 48 h de historia")
    t = np.arange(-HIST, 0)
    ax2.plot(t[-120:], base[0, -1, -120:], label="residuo, solo historia (este trabajo)")
    ax2.plot(t[-120:], future[0, -1, -120:], label="residuo, historia + futuro (paper)")
    ax2.plot(t[-120:], hist[0, -120:], color="k", lw=0.8, label="caudal")
    ax2.set(xlabel="horas antes del pronóstico", ylabel="mm/h", title="Ejemplo: fuga de información del futuro")
    ax2.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return table


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data")
    p.add_argument("--runs", default="runs")
    p.add_argument("--out", default="results")
    p.add_argument("--ref", default="dmel_s0", help="corrida de referencia para el test de Wilcoxon")
    p.add_argument("--limit", type=int, help="mismo --limit usado en train (pruebas rápidas)")
    a = p.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    truth = load_truth(Path(a.data), a.limit)
    report = ["# Resultados\n"]
    for split in ("val", "test"):
        s = truth[split]
        preds = load_predictions(a.runs, split, s["hist"])
        table = pd.DataFrame({label(n): metrics.summary(s["obs"], pr, s["basin"]) for n, pr in preds.items()}).T
        table.to_csv(out / f"metrics_{split}.csv")
        report += [f"## Métricas de {'validación' if split == 'val' else 'test'}\n", markdown(table.T), ""]
        print(f"\n== {split} ==\n{table.T.round(3).to_string()}")

    s = truth["test"]
    preds = load_predictions(a.runs, "test", s["hist"])
    lead_nse = {n: metrics.per_lead(s["obs"], pr, s["basin"]) for n, pr in preds.items()}
    pd.DataFrame({label(n): v for n, v in lead_nse.items()}, index=np.arange(1, HORIZON + 1)).to_csv(out / "nse_by_lead.csv")
    plot_leads(lead_nse, out / "nse_by_lead.png")
    if a.ref in preds:
        wtable, nse_b = wilcoxon_table(s["obs"], s["basin"], preds, a.ref)
        wtable.to_csv(out / "wilcoxon_test.csv")
        plot_cdf(nse_b, out / "nse_cdf.png")
        report += ["## Wilcoxon pareado por cuenca (test)\n", markdown(wtable), ""]
        print(f"\n{wtable.to_string()}")

    events = pick_events(s)
    ref_branches = Path(a.runs) / a.ref / "test_branches.npy"
    plot_events(s, preds, np.load(ref_branches) if ref_branches.exists() else None, events, out / "events.png")
    regimes = plot_regimes(s, preds, out / "regimes.png")
    leak = leakage_study(truth["val"], out / "leakage.png")
    report += ["## Error por régimen de caudal (test, %)\n", markdown(regimes), "",
               "## Fuga de información al descomponer también el futuro (validación)\n", markdown(leak), ""]
    (out / "summary.md").write_text("\n".join(report), encoding="utf-8")
    print(f"\n{leak.round(3).to_string()}\nFiguras y tablas en {out}/")


if __name__ == "__main__":
    main()
