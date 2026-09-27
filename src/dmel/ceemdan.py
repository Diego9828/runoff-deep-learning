"""CEEMDAN mejorado (Colominas, Schlotthauer y Torres, 2014), implementado con numba.

Notación:
    E_k(w)  k-ésimo modo EMD de la realización de ruido blanco w
    M(s)    media local de s = s - (primer IMF de s obtenido por sifting)

    r_0 = x
    r_k = < M(r_{k-1} + β_{k-1} E_k(w_i)) >_i      (promedio sobre las I realizaciones)
    IMF_k = r_{k-1} - r_k
    β_0 = ε, β_k = ε·std(r_k)   (x se normaliza a std 1; cada realización de ruido se divide por std(E_1))

El residuo final es x - Σ IMF_k, así la reconstrucción es exacta.
"""

import numpy as np
from numba import njit, prange


@njit(cache=True)
def extrema(x):
    """Índices de los máximos y mínimos locales (sin contar los bordes)."""
    n = x.size
    imax = np.empty(n, np.int64)
    imin = np.empty(n, np.int64)
    n_max = n_min = 0
    for i in range(1, n - 1):
        if x[i] > x[i - 1] and x[i] >= x[i + 1]:
            imax[n_max] = i
            n_max += 1
        elif x[i] < x[i - 1] and x[i] <= x[i + 1]:
            imin[n_min] = i
            n_min += 1
    return imax[:n_max], imin[:n_min]


@njit(cache=True)
def natural_spline(kx, ky, n):
    """Spline cúbica natural por los nodos (kx, ky), evaluada en t = 0..n-1 (kx creciente)."""
    m = kx.size
    h = kx[1:] - kx[:-1]
    M = np.zeros(m)  # segundas derivadas; natural: M[0] = M[m-1] = 0
    k = m - 2
    if k > 0:
        # sistema tridiagonal para M[1..m-2], resuelto con el algoritmo de Thomas
        diag = np.empty(k)
        upper = np.empty(k)
        rhs = np.empty(k)
        for i in range(k):
            diag[i] = 2.0 * (h[i] + h[i + 1])
            upper[i] = h[i + 1]
            rhs[i] = 6.0 * ((ky[i + 2] - ky[i + 1]) / h[i + 1] - (ky[i + 1] - ky[i]) / h[i])
        for i in range(1, k):
            w = h[i] / diag[i - 1]
            diag[i] -= w * upper[i - 1]
            rhs[i] -= w * rhs[i - 1]
        M[k] = rhs[k - 1] / diag[k - 1]
        for i in range(k - 2, -1, -1):
            M[i + 1] = (rhs[i] - upper[i] * M[i + 2]) / diag[i]
    out = np.empty(n)
    j = 0
    for t in range(n):
        while j < m - 2 and t > kx[j + 1]:
            j += 1
        a = kx[j + 1] - t
        b = t - kx[j]
        hj = h[j]
        out[t] = (M[j] * a**3 + M[j + 1] * b**3) / (6.0 * hj) + (ky[j] / hj - M[j] * hj / 6.0) * a + (
            ky[j + 1] / hj - M[j + 1] * hj / 6.0
        ) * b
    return out


@njit(cache=True)
def envelope(x, idx, upper):
    """Envolvente superior/inferior: spline por los extremos idx (al menos 2) y por los dos bordes.

    El valor en cada borde sale de la recta por los dos extremos más cercanos, sin quedar por
    debajo (envolvente superior) o por encima (inferior) del propio dato. No se refleja la señal:
    en una rama ascendente el espejo inventaría un pico en t = 336, justo la hora más importante.
    """
    n = x.size
    m = idx.size
    kx = np.empty(m + 2)
    ky = np.empty(m + 2)
    kx[0], kx[m + 1] = 0.0, n - 1.0
    for i in range(m):
        kx[i + 1] = idx[i]
        ky[i + 1] = x[idx[i]]
    left = ky[1] + (ky[2] - ky[1]) * (kx[0] - kx[1]) / (kx[2] - kx[1])
    right = ky[m] + (ky[m] - ky[m - 1]) * (kx[m + 1] - kx[m]) / (kx[m] - kx[m - 1])
    if upper:
        ky[0], ky[m + 1] = np.maximum(left, x[0]), np.maximum(right, x[n - 1])
    else:
        ky[0], ky[m + 1] = np.minimum(left, x[0]), np.minimum(right, x[n - 1])
    return natural_spline(kx, ky, n)


@njit(cache=True)
def sift(x, n_sift):
    """Primer IMF de x con un número fijo de iteraciones de sifting (Wu y Huang, 2009)."""
    h = x.copy()
    for it in range(n_sift):
        imax, imin = extrema(h)
        if imax.size < 2 or imin.size < 2:
            if it == 0:
                return np.zeros_like(x)  # x no oscila: no tiene IMF
            break
        h = h - 0.5 * (envelope(h, imax, True) + envelope(h, imin, False))
    return h


@njit(cache=True)
def is_residue(x):
    """True si x ya es una tendencia (menos de 3 extremos): no se pueden extraer más IMFs."""
    imax, imin = extrema(x)
    return imax.size + imin.size < 3


@njit(cache=True)
def emd(x, n_imfs, n_sift):
    """EMD clásica: [n_imfs + 1, T] con los IMFs (ceros si faltan) y el residuo en la última fila."""
    out = np.zeros((n_imfs + 1, x.size))
    r = x.copy()
    for k in range(n_imfs):
        imf = sift(r, n_sift)
        if not np.any(imf):
            break
        out[k] = imf
        r = r - imf
        if is_residue(r):
            break
    out[n_imfs] = r
    return out


def noise_bank(n_trials, length, n_imfs, n_sift=10, seed=0):
    """Modos EMD de n_trials realizaciones de ruido blanco: [I, K, T], divididos por std(E_1).

    Se calcula una sola vez y se reutiliza en todas las ventanas (descomposición reproducible).
    """
    rng = np.random.default_rng(seed)
    bank = np.zeros((n_trials, n_imfs, length))
    for i in range(n_trials):
        modes = emd(rng.standard_normal(length), n_imfs, n_sift)[:n_imfs]
        bank[i] = modes / modes[0].std()
    return bank


@njit(cache=True)
def iceemdan(x, bank, eps, n_sift):
    """CEEMDAN mejorado de una ventana: [K + 1, T] (K IMFs + residuo, en las unidades de x)."""
    n_trials, n_imfs, n = bank.shape
    out = np.zeros((n_imfs + 1, n))
    scale = x.std()
    if scale == 0.0:  # ventana constante (p. ej. cuenca seca): todo es residuo
        out[n_imfs] = x
        return out
    xn = x / scale
    r = xn.copy()
    for k in range(n_imfs):
        beta = eps if k == 0 else eps * r.std()
        r_next = np.zeros(n)
        for i in range(n_trials):
            noisy = r + beta * bank[i, k]  # si la realización no tiene modo k, bank[i, k] = 0
            r_next += noisy - sift(noisy, n_sift)  # M(.) = media local
        r_next /= n_trials
        out[k] = r - r_next
        r = r_next
        if is_residue(r):
            break
    out[n_imfs] = xn - out[:n_imfs].sum(axis=0)
    return out * scale


@njit(parallel=True, cache=True)
def decompose_all(Q, bank, eps, n_sift):
    """CEEMDAN de cada fila de Q [N, T] en paralelo -> [N, K + 1, T] float32."""
    n_windows, n = Q.shape
    out = np.zeros((n_windows, bank.shape[1] + 1, n), np.float32)
    for j in prange(n_windows):
        out[j] = iceemdan(Q[j].astype(np.float64), bank, eps, n_sift)
    return out
