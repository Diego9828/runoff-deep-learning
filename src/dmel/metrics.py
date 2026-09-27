"""Métricas hidrológicas del paper (RMSE, NSE, KGE) más MAE, por cuenca, por horizonte y en picos."""

import numpy as np


def nse(obs, sim):
    """Nash-Sutcliffe: 1 - Σ(o - s)² / Σ(o - ō)². nan si la observación es constante."""
    denom = np.sum((obs - obs.mean()) ** 2)
    return 1 - np.sum((obs - sim) ** 2) / denom if denom > 0 else np.nan


def kge(obs, sim):
    """Kling-Gupta: 1 - sqrt((r - 1)² + (α - 1)² + (β - 1)²), α = σ_s/σ_o, β = μ_s/μ_o."""
    if obs.std() == 0 or sim.std() == 0 or obs.mean() == 0:
        return np.nan
    r = np.corrcoef(obs, sim)[0, 1]
    alpha, beta = sim.std() / obs.std(), sim.mean() / obs.mean()
    return 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)


def rmse(obs, sim):
    return float(np.sqrt(np.mean((obs - sim) ** 2)))


def mae(obs, sim):
    return float(np.mean(np.abs(obs - sim)))


def per_basin(obs, sim, basin, metric):
    """Métrica de cada cuenca sobre todas sus ventanas y horizontes: dict cuenca -> valor."""
    return {b: metric(obs[basin == b].ravel(), sim[basin == b].ravel()) for b in np.unique(basin)}


def per_lead(obs, sim, basin, metric=nse):
    """Mediana entre cuencas de la métrica en cada horizonte h = 1..48 -> [48]."""
    return np.array([np.nanmedian(list(per_basin(obs[:, h], sim[:, h], basin, metric).values()))
                     for h in range(obs.shape[1])])


def peak_errors(obs, sim, basin, quantile=0.95):
    """Error de pico en las ventanas de crecida (pico observado >= cuantil de su cuenca).

    Devuelve EP (%) = (pico simulado - pico observado) / pico observado · 100 y el desfase del pico en horas.
    """
    peak_obs, peak_sim = obs.max(1), sim.max(1)
    threshold = {b: np.quantile(peak_obs[basin == b], quantile) for b in np.unique(basin)}
    flood = (peak_obs >= np.array([threshold[b] for b in basin])) & (peak_obs > 0)
    ep = 100 * (peak_sim[flood] - peak_obs[flood]) / peak_obs[flood]
    lag = sim[flood].argmax(1) - obs[flood].argmax(1)
    return ep, lag


def summary(obs, sim, basin):
    """Resumen de un modelo: medianas por cuenca, errores agregados y NSE en horizontes clave."""
    basin_nse = np.array(list(per_basin(obs, sim, basin, nse).values()))
    basin_kge = np.array(list(per_basin(obs, sim, basin, kge).values()))
    lead_nse = per_lead(obs, sim, basin)
    ep, lag = peak_errors(obs, sim, basin)
    row = {
        "NSE_mediana": np.nanmedian(basin_nse),
        "KGE_mediana": np.nanmedian(basin_kge),
        "RMSE": rmse(obs, sim),
        "MAE": mae(obs, sim),
        "NSE_global": nse(obs.ravel(), sim.ravel()),
        "cuencas_NSE<0": int(np.sum(basin_nse < 0)),
        "EP_pico_mediana_%": float(np.median(ep)),
        "EP_pico_abs_mediana_%": float(np.median(np.abs(ep))),
        "desfase_pico_abs_mediana_h": float(np.median(np.abs(lag))),
    }
    for h in (1, 6, 12, 24, 48):
        row[f"NSE_{h}h"] = lead_nse[h - 1]
    return row
