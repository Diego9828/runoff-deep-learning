"""Modal Recognition Strategy (MRS): sample entropy de cada componente CEEMDAN y reagrupación.

Regla (reproduce la Tabla 3 del paper para sus dos estaciones):
  1. Voto de cada componente = fracción de ventanas de train donde existe y SE_k > SE(serie original).
     Voto > 0.5 -> RIMF de alta frecuencia individual.
  2. El resto se suma por bandas de entropía: SE <= tau ("casi tendencia") y tau < SE ("intermedia").
  3. Cada suma se re-clasifica con el voto de su propia SE: > 0.5 alta frecuencia, si no baja.
"""

import numpy as np


def sample_entropy(x, m=2, r=0.2):
    """SampEn(m, r·std) de Richman y Moorman (2000). nan si x es constante, inf si no hay coincidencias."""
    x = np.asarray(x, dtype=np.float64)
    tol = r * x.std()
    if tol == 0:
        return np.nan
    n_templates = len(x) - m  # mismo número de plantillas para m y m+1

    def matches(length):
        emb = np.lib.stride_tricks.sliding_window_view(x, length)[:n_templates]
        dist = np.abs(emb[:, None, :] - emb[None, :, :]).max(-1)  # distancia de Chebyshev
        return (np.count_nonzero(dist <= tol) - n_templates) / 2  # sin auto-coincidencias

    B, A = matches(m), matches(m + 1)
    if B == 0:
        return np.nan
    return np.inf if A == 0 else -np.log(A / B)


def entropy_table(q, comps):
    """SE de la serie original [N] y de cada componente [N, K+1] (nan = componente ausente/constante)."""
    se_orig = np.array([sample_entropy(w) for w in q])
    se_comp = np.array([[sample_entropy(c) for c in window] for window in comps])
    return se_orig, se_comp


def votes(se, se_orig):
    """Fracción de ventanas (con SE original definida) donde la señal existe y SE > SE original.

    se: [N] (una señal) o [N, C] (varios componentes).
    """
    valid = np.isfinite(se_orig)
    se = np.where(np.isnan(se), -np.inf, se)[valid]  # componente ausente = no es de alta frecuencia
    ref = se_orig[valid] if se.ndim == 1 else se_orig[valid, None]
    return np.mean(se > ref, axis=0)


def propose_groups(comp_votes, comp_se, tau=0.1):
    """Pasos 1-2 de la regla: HF individuales + una suma por banda de entropía (listas de índices)."""
    groups = [[k] for k, v in enumerate(comp_votes) if v > 0.5]
    rest = [k for k, v in enumerate(comp_votes) if v <= 0.5]
    mid = [k for k in rest if comp_se[k] > tau]
    low = [k for k in rest if comp_se[k] <= tau]
    return groups + [g for g in (mid, low) if g]


def build_mrs(q, comps, tau=0.1):
    """MRS completo sobre una muestra de ventanas de train: devuelve un diccionario serializable."""
    se_orig, se_comp = entropy_table(q, comps)
    comp_votes = votes(se_comp, se_orig)
    comp_se = np.nanmedian(np.where(np.isinf(se_comp), np.nan, se_comp), axis=0)
    groups = propose_groups(comp_votes, comp_se, tau)

    group_votes, group_se = [], []
    for g in groups:
        se_g = se_comp[:, g[0]] if len(g) == 1 else np.array([sample_entropy(w) for w in comps[:, g].sum(1)])
        group_votes.append(float(votes(se_g, se_orig)))
        group_se.append(float(np.nanmedian(np.where(np.isinf(se_g), np.nan, se_g))))
    labels = ["high" if v > 0.5 else "low" for v in group_votes]
    if "low" not in labels:  # siempre al menos un grupo por rama
        labels[-1] = "low"
    if "high" not in labels:
        labels[0] = "high"

    n_imfs = comps.shape[1] - 1
    return {
        "n_windows": int(len(q)),
        "tau": tau,
        "se_orig_median": float(np.nanmedian(se_orig)),
        "components": [f"IMF{k + 1}" for k in range(n_imfs)] + ["residuo"],
        "component_votes": [float(v) for v in comp_votes],
        "component_se_median": [float(s) for s in comp_se],
        "component_present": [float(p) for p in np.mean(np.isfinite(se_comp), axis=0)],
        "groups": groups,
        "group_votes": group_votes,
        "group_se_median": group_se,
        "labels": labels,
    }
