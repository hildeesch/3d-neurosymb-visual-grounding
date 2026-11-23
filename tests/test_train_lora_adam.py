"""train_lora.py's optimiser reproduces mlx-lm's Adam (no bias correction)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from experiments.parser_adaptation.train_lora import DEFAULTS, mlx_adam_class


def _run_torch(grads, lr, **kw):
    p = torch.nn.Parameter(torch.zeros(grads.shape[1], dtype=torch.float64))
    opt = mlx_adam_class()([p], lr=lr, **kw)
    for g in grads:
        p.grad = torch.from_numpy(g)
        opt.step()
    return p.detach().numpy()


def _grads(seed=0, steps=200, dim=64):
    return np.random.default_rng(seed).normal(size=(steps, dim)) * 1e-2


def test_default_matches_thesis_mlx_setting():
    assert DEFAULTS["optimizer_config"]["adam"]["bias_correction"] is False


@pytest.mark.parametrize("bias_correction", [False, True])
def test_matches_mlx_adam(bias_correction):
    mx = pytest.importorskip("mlx.core")
    optim = pytest.importorskip("mlx.optimizers")
    grads, lr = _grads(), 1e-3
    opt = optim.Adam(learning_rate=lr, bias_correction=bias_correction)
    params = {"w": mx.zeros((grads.shape[1],), dtype=mx.float32)}
    for g in grads:
        opt.update(params, {"w": mx.array(g.astype(np.float32))})
    mx.eval(params)
    ours = _run_torch(grads, lr, bias_correction=bias_correction)
    np.testing.assert_allclose(ours, np.array(params["w"]), rtol=1e-4, atol=1e-6)


def test_bias_correction_true_is_torch_adam():
    grads, lr = _grads(1), 1e-3
    p = torch.nn.Parameter(torch.zeros(grads.shape[1], dtype=torch.float64))
    ref = torch.optim.Adam([p], lr=lr)
    for g in grads:
        p.grad = torch.from_numpy(g)
        ref.step()
    np.testing.assert_allclose(_run_torch(grads, lr, bias_correction=True),
                               p.detach().numpy(), rtol=1e-10, atol=1e-12)


def test_uncorrected_steps_are_larger_early():
    grads = np.ones((100, 4)) * 1e-2
    uncorrected = _run_torch(grads, 1e-5, bias_correction=False)
    corrected = _run_torch(grads, 1e-5, bias_correction=True)
    assert np.all(np.abs(uncorrected) > 2 * np.abs(corrected))
