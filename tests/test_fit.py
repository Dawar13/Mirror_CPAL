"""Test 1: fit_involution recovers a planted involution."""
import numpy as np

from mirror.fit import agreement, fit_involution, fit_layers, random_involution


def test_fit_involution_recovers_planted():
    rng = np.random.default_rng(0)
    d, n, d_plus = 24, 5000, 14
    A_true = random_involution(d, d_plus, seed=3)
    X = rng.standard_normal((n, d))
    Xt = X @ A_true + 0.01 * rng.standard_normal((n, d))
    A, dp, dm = fit_involution(X.T @ Xt)
    assert (dp, dm) == (d_plus, d - d_plus)
    assert np.linalg.norm(A - A_true) / np.linalg.norm(A_true) < 1e-2
    assert np.allclose(A, A.T, atol=1e-12)
    assert np.allclose(A @ A, np.eye(d), atol=1e-10)


def test_random_involution_properties():
    A = random_involution(10, 7, seed=0)
    assert np.allclose(A, A.T)
    assert np.allclose(A @ A, np.eye(10), atol=1e-12)
    assert round(np.trace(A)) == 7 - 3
    assert np.isclose(agreement(A, A), 1.0)


def test_zero_direction_assigned_even():
    """After channel centring the all-ones direction has no signal; it must get +1, not 0."""
    rng = np.random.default_rng(1)
    d = 12
    X = rng.standard_normal((800, d))
    X -= X.mean(1, keepdims=True)
    A, dp, dm, info = fit_involution(X.T @ X, return_info=True)
    assert info["n_zero"] == 1
    assert np.allclose(A @ A, np.eye(d), atol=1e-10)
    ones = np.ones(d) / np.sqrt(d)
    assert np.allclose(A @ ones, ones, atol=1e-8)


def test_shared_operator_from_layers():
    rng = np.random.default_rng(2)
    d = 16
    A_true = random_involution(d, 9, seed=5)
    C = []
    for _ in range(4):
        X = rng.standard_normal((2000, d))
        C.append(X.T @ (X @ A_true + 0.05 * rng.standard_normal((2000, d))))
    C = np.stack(C)
    out = fit_layers(C, (1, 3), sq_x=np.ones(4), sq_xt=np.ones(4))
    assert np.linalg.norm(out["A_shared"] - A_true) / np.linalg.norm(A_true) < 1e-2
    assert all(agreement(A, out["A_shared"]) > 0.99 for A in out["A_layers"])
