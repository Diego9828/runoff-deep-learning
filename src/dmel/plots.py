"""Figuras del reporte y la presentación (un color fijo por modelo en todas las figuras)."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dmel import metrics
from dmel.data import HIST, HORIZON

plt.rcParams.update({
    "savefig.dpi": 200, "font.size": 11, "axes.grid": True, "grid.alpha": 0.3,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})

LABELS = {
    "persistence": "Persistencia",
    "dmel_s0": "DMEL (paper)",
    "dmel_swap_s0": "Ablación: asignación invertida",
    "hybrid_raw_s0": "Ablación: sin MRS",
    "dmel_q_s0": "Variante: DMEL + caudal",
}
COLORS = {"persistence": "0.55", "dmel_s0": "tab:purple", "dmel_swap_s0": "tab:orange", "hybrid_raw_s0": "tab:green",
          "dmel_q_s0": "tab:brown"}
HF_COLOR, LF_COLOR = "tab:red", "tab:blue"


def label(name):
    return LABELS.get(name, name)


def color(name):
    return COLORS.get(name)


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def entropy_bars(mrs, path):
    """Mediana de SE por componente vs SE de la serie original (equivalente a la Fig. 9 del paper)."""
    label_of = {k: lab for g, lab in zip(mrs["groups"], mrs["labels"]) for k in g}
    colors = [HF_COLOR if label_of[k] == "high" else LF_COLOR for k in range(len(mrs["components"]))]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.bar(mrs["components"], mrs["component_se_median"], color=colors)
    ax.axhline(mrs["se_orig_median"], color="k", ls="--", label=f"serie original ({mrs['se_orig_median']:.3f})")
    ax.set(ylabel="Sample entropy (mediana)", title="MRS: rojo = alta frecuencia (Informer), azul = baja (LSTM)")
    ax.legend()
    save(fig, path)


def decomposition(hist, comps, mrs, basin, path):
    """Una ventana: caudal observado y sus componentes CEEMDAN coloreados según el grupo MRS."""
    label_of = {k: lab for g, lab in zip(mrs["groups"], mrs["labels"]) for k in g}
    names, t = mrs["components"], np.arange(-HIST, 0)
    fig, axes = plt.subplots(len(names) + 1, 1, figsize=(10, 1.15 * (len(names) + 1)), sharex=True)
    axes[0].plot(t, hist, color="k")
    axes[0].set_ylabel("caudal\n(mm/h)", rotation=0, ha="right", va="center")
    for k, ax in enumerate(axes[1:]):
        ax.plot(t, comps[k], color=HF_COLOR if label_of[k] == "high" else LF_COLOR, lw=1)
        ax.set_ylabel(f"{names[k]}\nSE {mrs['component_se_median'][k]:.2f}", rotation=0, ha="right", va="center")
    axes[-1].set_xlabel("horas antes del pronóstico")
    axes[0].set_title(f"CEEMDAN de una ventana de test (cuenca {basin}): rojo → Informer, azul → LSTM")
    save(fig, path)


def leakage(base, future, hist, table, path):
    """Cambio de los componentes al descomponer también el futuro (protocolo del paper) vs control."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3.9))
    table.plot.bar(ax=ax1, rot=0)
    ax1.set(ylabel="cambio RMS / std del caudal", title="Últimas 48 h de historia (mediana de 200 ventanas)")
    t = np.arange(-HIST, 0)[-120:]
    ax2.plot(t, base[0, -1, -120:], label="residuo: solo historia (este trabajo)")
    ax2.plot(t, future[0, -1, -120:], label="residuo: historia + futuro (paper)")
    ax2.plot(t, hist[0, -120:], color="k", lw=0.8, label="caudal observado")
    ax2.set(xlabel="horas antes del pronóstico", ylabel="mm/h", title="Ejemplo: el residuo 've' la crecida futura")
    ax2.legend(fontsize=8)
    save(fig, path)


def training_curves(histories, path):
    """Pérdida NSE* de train/val, NSE mediano de validación y learning rate por época."""
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(14, 3.9))
    for name, h in histories.items():
        a1.plot(h["epoch"], h["train_loss"], color=color(name), ls="--")
        a1.plot(h["epoch"], h["val_loss"], color=color(name), label=label(name))
        a2.plot(h["epoch"], h["val_nse_mediana"], color=color(name))
        a3.plot(h["epoch"], h["lr"], color=color(name))
    a1.set(title="Pérdida NSE* (-- train, — validación)", xlabel="época")
    a2.set(title="NSE mediano en validación", xlabel="época")
    a3.set(title="Learning rate", xlabel="época", yscale="log")
    a1.legend(fontsize=8)
    save(fig, path)


def nse_by_lead(lead_nse, path):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for name, values in lead_nse.items():
        ax.plot(np.arange(1, HORIZON + 1), values, color=color(name), label=label(name),
                ls="--" if name == "persistence" else "-")
    ax.set(xlabel="Horizonte (h)", ylabel="NSE (mediana entre cuencas)", title="Test: NSE por horizonte")
    ax.legend()
    save(fig, path)


def nse_boxes(split, preds, path, leads=(1, 6, 12, 24, 48)):
    """Distribución entre cuencas del NSE en horizontes clave (análogo a los violines del paper)."""
    fig, ax = plt.subplots(figsize=(11, 4.2))
    width = 0.8 / len(preds)
    for j, (name, p) in enumerate(preds.items()):
        data = []
        for h in leads:
            v = np.array(list(metrics.per_basin(split["obs"][:, h - 1], p[:, h - 1], split["basin"], metrics.nse).values()))
            data.append(np.clip(v[np.isfinite(v)], -1, 1))
        pos = np.arange(len(leads)) + (j - (len(preds) - 1) / 2) * width
        box = ax.boxplot(data, positions=pos, widths=0.9 * width, patch_artist=True, showfliers=False,
                         medianprops={"color": "k"})
        for patch in box["boxes"]:
            patch.set(facecolor=color(name), alpha=0.75)
        ax.plot([], [], color=color(name), lw=8, label=label(name))
    ax.set_xticks(range(len(leads)), [f"{h} h" for h in leads])
    ax.set(ylabel="NSE por cuenca (recortado en -1)", title="Test: NSE de las 508 cuencas por horizonte")
    ax.legend(ncol=len(preds), fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    save(fig, path)


def nse_cdf(nse_b, path):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for name, values in nse_b.items():
        v = np.sort(np.clip(values.dropna().to_numpy(), -1, 1))
        ax.plot(v, np.linspace(0, 1, len(v)), color=color(name), label=label(name),
                ls="--" if name == "persistence" else "-")
    ax.set(xlabel="NSE por cuenca (recortado en -1)", ylabel="fracción de cuencas", title="Test: distribución del NSE")
    ax.legend(fontsize=8)
    save(fig, path)


def scatter(split, preds, path, lead=24):
    """Observado vs pronosticado a un horizonte (hexbin log-log), como las Figs. 11-12 del paper."""
    obs = split["obs"][:, lead - 1]
    fig, axes = plt.subplots(1, len(preds), figsize=(3.7 * len(preds), 3.9), sharex=True, sharey=True, squeeze=False)
    lims = [max(obs[obs > 0].min(), 1e-4), obs.max()]
    for ax, (name, p) in zip(axes[0], preds.items()):
        sim = p[:, lead - 1]
        ok = (obs > 0) & (sim > 0)
        ax.hexbin(obs[ok], sim[ok], xscale="log", yscale="log", gridsize=45, mincnt=1, bins="log", cmap="viridis")
        ax.plot(lims, lims, "k--", lw=0.8)
        ax.set(title=f"{label(name)}\nR² = {np.corrcoef(obs, sim)[0, 1] ** 2:.3f}", xlabel="observado (mm/h)")
    axes[0, 0].set_ylabel(f"pronóstico a {lead} h (mm/h)")
    save(fig, path)


def events(split, preds, rows, path):
    """Hidrogramas de las mayores crecidas de test: últimas 72 h observadas + 48 h pronosticadas."""
    fig, axes = plt.subplots(1, len(rows), figsize=(4.2 * len(rows), 3.8), squeeze=False)
    for ax, i in zip(axes[0], rows):
        ax.plot(np.arange(-72, 0), split["hist"][i, -72:], color="k", lw=1)
        ax.plot(np.arange(HORIZON), split["obs"][i], color="k", lw=2.2, label="observado")
        for name, p in preds.items():
            ax.plot(np.arange(HORIZON), p[i], color=color(name), label=label(name), ls="--" if name == "persistence" else "-")
        ax.axvline(0, color="0.7", lw=0.8)
        ax.set(title=f"cuenca {split['basin'][i]}", xlabel="horas desde el pronóstico")
    axes[0, 0].set_ylabel("caudal (mm/h)")
    axes[0, 0].legend(fontsize=7)
    save(fig, path)


def branches(split, parts, rows, path):
    """Aporte de cada rama de DMEL: Informer (RIMFs de alta frecuencia) y LSTM (baja frecuencia)."""
    fig, axes = plt.subplots(1, len(rows), figsize=(4.2 * len(rows), 3.8), squeeze=False)
    for ax, i in zip(axes[0], rows):
        t = np.arange(HORIZON)
        ax.plot(np.arange(-72, 0), split["hist"][i, -72:], color="k", lw=1)
        ax.plot(t, split["obs"][i], color="k", lw=2.2, label="observado")
        ax.plot(t, parts[i].sum(0), color=COLORS["dmel_s0"], label="DMEL = suma")
        ax.plot(t, parts[i, 0], color=HF_COLOR, ls=":", label="rama HF (Informer)")
        ax.plot(t, parts[i, 1], color=LF_COLOR, ls=":", label="rama LF (LSTM)")
        ax.axvline(0, color="0.7", lw=0.8)
        ax.set(title=f"cuenca {split['basin'][i]}", xlabel="horas desde el pronóstico")
    axes[0, 0].set_ylabel("caudal (mm/h)")
    axes[0, 0].legend(fontsize=7)
    save(fig, path)


def regimes(table, names, path):
    fig, ax = plt.subplots(figsize=(8, 4))
    table.plot.bar(ax=ax, rot=0, color=[color(n) for n in names])
    ax.set(ylabel="error absoluto relativo (%)", xlabel="régimen de caudal (terciles de cada cuenca)",
           title="Test: error por régimen de caudal")
    ax.legend(fontsize=8)
    save(fig, path)


def peaks(peak_df, names, path):
    """Error de magnitud (EP) y de tiempo del pico en las ventanas de crecida (paper, Sec. 5.5)."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    groups = [peak_df[peak_df["modelo"] == label(n)] for n in names]
    for ax, column, title in ((a1, "EP_%", "Error del pico EP (%)"), (a2, "desfase_h", "Desfase del pico (h)")):
        box = ax.boxplot([g[column] for g in groups], tick_labels=[label(n) for n in names], patch_artist=True,
                         showfliers=False, medianprops={"color": "k"})
        for patch, n in zip(box["boxes"], names):
            patch.set(facecolor=color(n), alpha=0.75)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_title(title)
        ax.tick_params(axis="x", labelrotation=15, labelsize=8)
    save(fig, path)


def nse_vs_flow(split, nse_b, names, path):
    """NSE de cada cuenca vs su caudal medio: ¿en qué cuencas falla el modelo?"""
    basins = np.unique(split["basin"])
    mean_flow = pd.Series({b: split["obs"][split["basin"] == b].mean() for b in basins})
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for name in names:
        ax.scatter(mean_flow, nse_b[name].reindex(mean_flow.index).clip(-1, 1), s=9, alpha=0.6,
                   color=color(name), label=label(name))
    ax.set(xscale="log", xlabel="caudal medio de la cuenca en test (mm/h)", ylabel="NSE (recortado en -1)",
           title="Test: desempeño por cuenca vs magnitud del caudal")
    ax.legend(fontsize=8)
    save(fig, path)
