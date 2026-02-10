import numpy as np

from src.accelerated import (
    get_array_backend,
    normalize_scores,
    normalize_scores_cuda,
    cuda_available,
    log_returns_numba,
)


def test_backend_falls_back_or_uses_valid_backend():
    backend = get_array_backend(prefer_gpu=True)
    assert backend.name in {"numpy", "cupy"}


def test_normalize_scores_mean_and_std():
    scores = np.array([1.0, 2.0, 3.0, 4.0])
    out = normalize_scores(scores, prefer_gpu=False, prefer_numba=True)
    assert abs(out.mean()) < 1e-7
    assert abs(out.std() - 1.0) < 1e-6


def test_cuda_normalization_fallback_or_cuda_path():
    scores = np.array([1.0, 2.0, 3.0, 4.0])
    out, used_cuda = normalize_scores_cuda(scores)
    assert out.shape == (4,)
    if cuda_available():
        assert used_cuda is True
    else:
        assert used_cuda is False


def test_log_returns_numba_output():
    prices = np.array([100.0, 101.0, 100.5, 102.0], dtype=np.float64)
    out = log_returns_numba(prices)
    assert np.isnan(out[0])
    assert out.shape == prices.shape
