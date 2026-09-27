import numpy as np

from dmel.mrs import propose_groups, sample_entropy, votes

# Tabla 2 del paper: SE de la serie original y de IMF1..IMF9
PAPER = {
    "Shuangpai": (0.1769, [0.1885, 0.2226, 0.1911, 0.2593, 0.2167, 0.0509, 0.0305, 0.0162, 0.0014]),
    "Fenghuang": (0.4523, [0.5190, 0.5895, 0.3247, 0.2056, 0.1522, 0.0601, 0.0339, 0.0127, 0.0007]),
}


def test_sample_entropy_white_noise():
    x = np.random.default_rng(0).standard_normal(1000)
    assert abs(sample_entropy(x) - 2.2) < 0.2


def test_sample_entropy_regular_signals():
    t = np.arange(336)
    assert sample_entropy(np.sin(2 * np.pi * t / 24)) < 0.5
    assert np.isnan(sample_entropy(np.ones(336)))


def test_votes_ignore_absent_components():
    se_orig = np.array([0.1, 0.1, np.nan])
    se = np.array([[0.5, np.nan], [0.5, 0.05], [0.9, 0.9]])  # la 3ra ventana no cuenta (SE original nan)
    np.testing.assert_allclose(votes(se, se_orig), [1.0, 0.0])


def test_rule_reproduces_paper_table3():
    """Con una sola serie el voto es 1/0; la regla debe dar exactamente los grupos de la Tabla 3."""
    expected = {
        "Shuangpai": [[0], [1], [2], [3], [4], [5, 6, 7, 8]],
        "Fenghuang": [[0], [1], [2, 3, 4], [5, 6, 7, 8]],
    }
    for station, (se_orig, se) in PAPER.items():
        comp_votes = [1.0 if s > se_orig else 0.0 for s in se]
        assert propose_groups(comp_votes, se, tau=0.1) == expected[station]


def test_paper_group_labels():
    """Con las SE de los grupos reagrupados del paper, RIMF3 de Fenghuang (0.6825) es de alta frecuencia."""
    se_orig = PAPER["Fenghuang"][0]
    group_se = [0.5190, 0.5895, 0.6825, 0.1074]
    assert ["high" if s > se_orig else "low" for s in group_se] == ["high", "high", "high", "low"]
