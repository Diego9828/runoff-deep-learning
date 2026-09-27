import pytest
import torch

from dmel.informer import Informer, full_attention, prob_attention
from dmel.models import VARIANTS, build_model, input_keys


@pytest.mark.parametrize("causal", [False, True])
def test_prob_attention_equals_full_when_all_queries_are_active(causal):
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 4, 40, 8) for _ in range(3))
    expected = full_attention(q, k, v, causal)
    torch.testing.assert_close(prob_attention(q, k, v, factor=100, causal=causal), expected, atol=1e-5, rtol=1e-5)


def test_full_causal_attention_ignores_the_future():
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 2, 10, 4) for _ in range(3))
    k2, v2 = k.clone(), v.clone()
    k2[:, :, 6:], v2[:, :, 6:] = torch.randn(1, 2, 4, 4), torch.randn(1, 2, 4, 4)
    torch.testing.assert_close(full_attention(q, k, v, True)[:, :, :6], full_attention(q, k2, v2, True)[:, :, :6])


def test_informer_encoder_distils_to_half():
    model = Informer(c_in=5)
    assert model.encode(torch.randn(2, 336, 5)).shape == (2, 168, 256)
    assert model(torch.randn(2, 336, 5)).shape == (2, 48)


@pytest.mark.parametrize("name", list(VARIANTS))
def test_variants_output_48_hours(name):
    channels = {"raw": 14, "hf": 18, "lf": 14, "lf_q": 15}
    model = build_model(name, {k: channels[k] for k in input_keys(name)})
    inputs = {k: torch.randn(3, 336, channels[k]) for k in input_keys(name)}
    assert model(inputs).shape == (3, 48)
    assert len(model.branch_outputs(inputs)) == 2
