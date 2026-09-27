import numpy as np
import pytest
from scipy.interpolate import CubicSpline

from dmel.ceemdan import decompose_all, emd, iceemdan, natural_spline, noise_bank

T = 336


@pytest.fixture(scope="module")
def bank():
    return noise_bank(n_trials=20, length=T, n_imfs=6)


def test_natural_spline_matches_scipy():
    rng = np.random.default_rng(0)
    kx = np.concatenate([[0.0], np.sort(rng.choice(np.arange(1, T - 1), 30, replace=False)).astype(float), [T - 1.0]])
    ky = rng.standard_normal(kx.size)
    expected = CubicSpline(kx, ky, bc_type="natural")(np.arange(T))
    np.testing.assert_allclose(natural_spline(kx, ky, T), expected, atol=1e-10)


def test_emd_is_complete():
    x = np.random.default_rng(1).standard_normal(T)
    np.testing.assert_allclose(emd(x, 6, 10).sum(0), x, atol=1e-10)


def test_iceemdan_is_complete(bank):
    t = np.arange(T)
    x = 1 + np.sin(2 * np.pi * t / 12) + 0.3 * np.random.default_rng(2).standard_normal(T)
    comps = iceemdan(x, bank, 0.2, 10)
    assert comps.shape == (7, T)
    np.testing.assert_allclose(comps.sum(0), x, atol=1e-9)


def test_iceemdan_affine_equivariance(bank):
    """Escalar la ventana escala los componentes; sumar una constante solo cambia el residuo."""
    x = np.random.default_rng(3).gamma(2.0, size=T)
    a, c = 3.7, 5.0
    base, moved = iceemdan(x, bank, 0.2, 10), iceemdan(a * x + c, bank, 0.2, 10)
    np.testing.assert_allclose(moved[:-1], a * base[:-1], atol=1e-8)
    np.testing.assert_allclose(moved[-1], a * base[-1] + c, atol=1e-8)


def test_iceemdan_separates_scales(bank):
    t = np.arange(T)
    fast, slow = np.sin(2 * np.pi * t / 6), 2 * np.sin(2 * np.pi * t / 96)
    comps = iceemdan(fast + slow + 0.01 * t, bank, 0.2, 10)
    assert np.corrcoef(comps[0], fast)[0, 1] > 0.9
    rest = comps[1:].sum(0)
    assert np.corrcoef(rest, slow + 0.01 * t)[0, 1] > 0.9


def test_constant_window_is_all_residue(bank):
    comps = iceemdan(np.full(T, 0.5), bank, 0.2, 10)
    assert not comps[:-1].any()
    np.testing.assert_allclose(comps[-1], 0.5)


def test_decompose_all_matches_single_window(bank):
    Q = np.random.default_rng(4).gamma(2.0, size=(3, T)).astype(np.float32)
    out = decompose_all(Q, bank, 0.2, 10)
    assert out.shape == (3, 7, T) and out.dtype == np.float32
    np.testing.assert_allclose(out[1], iceemdan(Q[1].astype(np.float64), bank, 0.2, 10), rtol=1e-5, atol=1e-6)
