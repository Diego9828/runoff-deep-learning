"""CLI: métricas de validación y test, persistencia, Wilcoxon por cuenca, tablas y figuras del informe.

Es el único script que lee test_targets.csv (solo para el reporte final, nunca para elegir modelos).
Salidas en --out: *.csv (números), tables/*.md (tablas para incluir en Quarto) y figures/*.png.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from dmel import metrics, plots
from dmel.ceemdan import decompose_all, noise_bank
from dmel.data import HIST, HORIZON, load_discharge, load_h5
from dmel.plots import label


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


def pick_events(split, n=4):
    """Ventanas con las mayores subidas de caudal, en cuencas distintas."""
    rise = (split["obs"].max(1) - split["hist"][:, -1]) / (split["hist"].std(1) + 1e-3)
    events, seen = [], set()
    for i in np.argsort(rise)[::-1]:
        if split["basin"][i] not in seen:
            events.append(i)
            seen.add(split["basin"][i])
        if len(events) == n:
            break
    return events


def per_basin_table(split, preds):
    columns = {f"{m} {label(n)}": pd.Series(metrics.per_basin(split["obs"], p, split["basin"], fn))
               for n, p in preds.items() for m, fn in (("NSE", metrics.nse), ("KGE", metrics.kge))}
    return pd.DataFrame(columns).rename_axis("cuenca")


def wilcoxon_table(split, preds, ref):
    """Wilcoxon pareado sobre el NSE de cada cuenca: `ref` contra cada modelo."""
    nse_b = {n: pd.Series(metrics.per_basin(split["obs"], p, split["basin"], metrics.nse)) for n, p in preds.items()}
    rows = {}
    for name in preds:
        if name == ref:
            continue
        both = pd.concat([nse_b[ref], nse_b[name]], axis=1).dropna()
        diff = both.iloc[:, 0] - both.iloc[:, 1]
        pvalue = wilcoxon(both.iloc[:, 0], both.iloc[:, 1]).pvalue
        rows[f"{label(ref)} vs {label(name)}"] = {
            "mediana ΔNSE": diff.median(),
            "cuencas donde gana": f"{int((diff > 0).sum())}/{len(diff)}",
            "p-valor": f"{pvalue:.1e}" if pvalue > 0 else "< 1e-300",
        }
    return pd.DataFrame(rows).T.rename_axis("comparación"), nse_b


def wilcoxon_rows(wtable, preds, ref):
    """Filas del Wilcoxon contra `ref` con una columna por modelo, para sumarlas a la tabla de métricas."""
    rows = {}
    for n in preds:
        key = f"{label(ref)} vs {label(n)}"
        wins, pvalue = ("—", "—") if n == ref else (wtable.loc[key, "cuencas donde gana"], wtable.loc[key, "p-valor"])
        rows[label(n)] = {f"Wilcoxon vs {label(ref)}: p-valor": pvalue, f"Cuencas donde gana {label(ref)}": wins}
    return pd.DataFrame(rows)


def regimes_table(split, preds):
    """Error absoluto relativo (Σ|s - o| / Σo, %) por régimen de caudal (terciles de cada cuenca)."""
    obs, basin = split["obs"], split["basin"]
    regime = np.zeros(obs.shape, int)
    for b in np.unique(basin):
        rows = basin == b
        regime[rows] = np.digitize(obs[rows], np.quantile(obs[rows], [1 / 3, 2 / 3]))
    return pd.DataFrame(
        {label(n): [100 * np.abs(p - obs)[regime == r].sum() / obs[regime == r].sum() for r in range(3)]
         for n, p in preds.items()},
        index=pd.Index(["bajo", "medio", "alto"], name="régimen"),
    )


def peaks_table(split, preds):
    frames = []
    for name, p in preds.items():
        ep, lag = metrics.peak_errors(split["obs"], p, split["basin"])
        frames.append(pd.DataFrame({"modelo": label(name), "EP_%": ep, "desfase_h": lag}))
    return pd.concat(frames, ignore_index=True)


def training_table(runs, histories):
    rows = {}
    for name, h in histories.items():
        config = json.loads((Path(runs) / name / "config.json").read_text())
        best = h.loc[h["val_loss"].idxmin()]
        rows[label(name)] = {
            "parámetros (M)": config["parametros"] / 1e6,
            "épocas entrenadas": len(h),
            "mejor época": int(best["epoch"]),
            "NSE* val (mejor)": best["val_loss"],
            "NSE val mediana (mejor)": best["val_nse_mediana"],
            "min por época": h["segundos"].mean() / 60,
            "horas totales": h["segundos"].sum() / 3600,
        }
    return pd.DataFrame(rows).T.rename_axis("corrida")


def mrs_tables(mrs):
    components = pd.DataFrame({
        "presente (%)": 100 * np.array(mrs["component_present"]),
        "SE mediana": mrs["component_se_median"],
        "voto SE > SE original": mrs["component_votes"],
    }, index=pd.Index(mrs["components"], name="componente"))
    groups = pd.DataFrame({
        "componentes": [" + ".join(mrs["components"][k] for k in g) for g in mrs["groups"]],
        "SE mediana": mrs["group_se_median"],
        "voto": mrs["group_votes"],
        "rama": ["alta frecuencia → Informer" if lab == "high" else "baja frecuencia → LSTM" for lab in mrs["labels"]],
    }, index=pd.Index([f"RIMF{j + 1}" for j in range(len(mrs["groups"]))], name="RIMF"))
    return components, groups


def leakage_study(split, path, n_windows=200, seed=0):
    """¿Cuánto cambian los componentes si se descompone también el futuro (como en el paper)?

    Compara la descomposición de la historia (336 h) con la de historia + futuro (384 h) en las últimas 48 h de
    historia, en ventanas de validación al azar (la primera es la mayor crecida, para el ejemplo de la figura).
    Control: dos descomposiciones de la misma historia con bancos de ruido distintos.
    """
    rng = np.random.default_rng(seed)
    moving = np.flatnonzero(split["hist"].std(1) > 0)  # sin ventanas constantes (cuencas secas)
    rows = np.concatenate([pick_events(split, 1), rng.choice(moving, min(n_windows, len(moving)), replace=False)])
    hist = split["hist"][rows].astype(np.float64)
    both = np.concatenate([hist, split["obs"][rows]], axis=1)
    base = decompose_all(hist, noise_bank(100, HIST, 6, seed=0), 0.2, 10)
    control = decompose_all(hist, noise_bank(100, HIST, 6, seed=1), 0.2, 10)
    future = decompose_all(both, noise_bank(100, HIST + HORIZON, 6, seed=0), 0.2, 10)[:, :, :HIST]

    def change(other):  # RMS del cambio en las últimas 48 h / std del caudal de la ventana
        rms = np.sqrt(np.mean((other[:, :, -HORIZON:] - base[:, :, -HORIZON:]) ** 2, axis=2))
        return np.median(rms[1:] / hist[1:].std(1, keepdims=True), axis=0)

    names = [f"IMF{k + 1}" for k in range(6)] + ["residuo"]
    table = pd.DataFrame({"descomponiendo también el futuro": change(future), "control (otro ruido)": change(control)},
                         index=pd.Index(names, name="componente"))
    plots.leakage(base, future, hist, table, path)
    return table


def markdown(df, digits=3):
    """Tabla markdown simple (evita depender de tabulate)."""
    def fmt(v):
        if not isinstance(v, (float, np.floating)):
            return str(v)
        v = float(v)
        if v.is_integer() and abs(v) >= 1:  # conteos (épocas, cuencas)
            return str(int(v))
        return f"{v:.{digits}g}" if abs(v) < 1e-2 else f"{v:.{digits}f}"
    lines = ["| " + " | ".join([df.index.name or ""] + [str(c) for c in df.columns]) + " |",
             "|" + "---|" * (len(df.columns) + 1)]
    lines += ["| " + " | ".join([str(i)] + [fmt(v) for v in row]) + " |" for i, row in zip(df.index, df.to_numpy())]
    return "\n".join(lines)


def write_table(df, out, name, digits=3):
    """Guarda la tabla como CSV y como markdown (results/tables/<name>.md, para incluir en Quarto)."""
    df.to_csv(out / f"{name}.csv")
    text = markdown(df, digits)
    (out / "tables" / f"{name}.md").write_text(text + "\n", encoding="utf-8")
    return text


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data")
    p.add_argument("--runs", default="runs")
    p.add_argument("--artifacts", default="artifacts")
    p.add_argument("--out", default="results")
    p.add_argument("--ref", default="dmel_s0", help="corrida de referencia (Wilcoxon, aporte de las ramas)")
    p.add_argument("--limit", type=int, help="mismo --limit usado en train (pruebas rápidas)")
    a = p.parse_args()

    out, figs = Path(a.out), Path(a.out) / "figures"
    (out / "tables").mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    truth = load_truth(Path(a.data), a.limit)
    report = ["# Resultados\n"]

    preds, wtable, nse_b = {}, None, None
    for split in ("val", "test"):
        s = truth[split]
        preds[split] = load_predictions(a.runs, split, s["hist"])
        table = pd.DataFrame({label(n): metrics.summary(s["obs"], pr, s["basin"]) for n, pr in preds[split].items()})
        if split == "test" and a.ref in preds[split]:
            wtable, nse_b = wilcoxon_table(s, preds[split], a.ref)
            table = pd.concat([table, wilcoxon_rows(wtable, preds[split], a.ref)])
        name = "validación" if split == "val" else "test"
        report += [f"## Métricas de {name}\n", write_table(table.rename_axis("métrica"), out, f"metrics_{split}"), ""]
        print(f"\n== {split} ==\n{table.round(3).to_string()}")
        lead = pd.DataFrame({label(n): metrics.per_lead(s["obs"], pr, s["basin"]) for n, pr in preds[split].items()},
                            index=pd.Index(np.arange(1, HORIZON + 1), name="horizonte_h"))
        lead.to_csv(out / f"nse_by_lead_{split}.csv")
        per_basin_table(s, preds[split]).to_csv(out / f"per_basin_{split}.csv")

    s, p_test = truth["test"], preds["test"]
    names = list(p_test)
    plots.nse_by_lead({n: metrics.per_lead(s["obs"], pr, s["basin"]) for n, pr in p_test.items()}, figs / "nse_by_lead.png")
    plots.nse_boxes(s, p_test, figs / "nse_boxes.png")
    plots.scatter(s, p_test, figs / "scatter_24h.png")
    rows = pick_events(s)
    plots.events(s, p_test, rows, figs / "events.png")
    ref_parts = Path(a.runs) / a.ref / "test_branches.npy"
    if ref_parts.exists():
        plots.branches(s, np.load(ref_parts), rows, figs / "branches.png")

    regimes = regimes_table(s, p_test)
    plots.regimes(regimes, names, figs / "regimes.png")
    report += ["## Error por régimen de caudal (test, %)\n", write_table(regimes, out, "regimes_test"), ""]
    peak_df = peaks_table(s, p_test)
    peak_df.to_csv(out / "peaks_test.csv", index=False)
    plots.peaks(peak_df, names, figs / "peaks.png")

    if wtable is not None:
        report += ["## Wilcoxon pareado por cuenca (test)\n", write_table(wtable, out, "wilcoxon_test"), ""]
        plots.nse_cdf(nse_b, figs / "nse_cdf.png")
        plots.nse_vs_flow(s, nse_b, [a.ref, "persistence"], figs / "nse_vs_flow.png")
        print(f"\n{wtable.to_string()}")

    histories = {run.parent.name: pd.read_csv(run.parent / "history.csv") for run in sorted(Path(a.runs).glob("*/done"))}
    if histories:
        plots.training_curves(histories, figs / "training_curves.png")
        pd.concat([h.assign(corrida=n) for n, h in histories.items()]).to_csv(out / "training_history.csv", index=False)
        report += ["## Entrenamiento\n", write_table(training_table(a.runs, histories), out, "training"), ""]

    art = Path(a.artifacts)
    if (art / "mrs.json").exists():
        mrs = json.loads((art / "mrs.json").read_text())
        plots.entropy_bars(mrs, figs / "mrs_entropy.png")
        comp_table, group_table = mrs_tables(mrs)
        report += ["## MRS: entropía por componente\n", write_table(comp_table, out, "mrs_components"), "",
                   "## MRS: grupos\n", write_table(group_table, out, "mrs_groups"), ""]
        comps = np.load(art / "components_test.npy", mmap_mode="r")
        plots.decomposition(s["hist"][rows[0]], np.asarray(comps[rows[0]]), mrs, s["basin"][rows[0]],
                            figs / "decomposition.png")

    leak = leakage_study(truth["val"], figs / "leakage.png")
    report += ["## Fuga de información al descomponer también el futuro (validación)\n",
               write_table(leak, out, "leakage_val"), ""]
    (out / "summary.md").write_text("\n".join(report), encoding="utf-8")
    print(f"\n{leak.round(3).to_string()}\nTablas y figuras en {out}/")


if __name__ == "__main__":
    main()
