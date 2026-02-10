import numpy as np

from src.accelerated import get_array_backend, normalize_scores


def test_backend_falls_back_or_uses_valid_backend():
    backend = get_array_backend(prefer_gpu=True)
    assert backend.name in {"numpy", "cupy"}


def test_normalize_scores_mean_and_std():
    scores = np.array([1.0, 2.0, 3.0, 4.0])
    out = normalize_scores(scores, prefer_gpu=False)
    assert abs(out.mean()) < 1e-7
    assert abs(out.std() - 1.0) < 1e-6
