"""Test 2: matching recovers a planted pairing of mirror twins plus self-twins."""
import numpy as np

from mirror.matching import candidate_edges, match_many, match_neurons, random_pairing


def planted(h=60, n_pairs=18, seed=0, noise=0.05):
    rng = np.random.default_rng(seed)
    perm = random_pairing(h, n_pairs, seed=seed + 1)
    R = noise * rng.standard_normal((h, h))
    paired = perm != np.arange(h)
    # Twins: neuron i of the mirror correlates with neuron perm[i] of the original.
    R[np.arange(h), perm] += np.where(paired, 0.9, 0.85)
    R[np.arange(h)[paired], np.arange(h)[paired]] += 0.1  # weak self-correlation for paired neurons
    return R, perm


def test_matching_recovers_planted_pairs():
    R, perm = planted()
    res = match_neurons(R, top_k=10)
    assert np.array_equal(res.perm, perm)
    assert res.n_pairs == 18
    assert res.n_self == 60 - 36
    assert np.array_equal(res.perm[res.perm], np.arange(60))  # involution
    assert np.all(res.score > 0.7)


def test_match_many_parallel_equals_serial():
    Rs = [planted(seed=s)[0] for s in range(3)]
    serial = match_many(Rs, workers=1)
    parallel = match_many(Rs, workers=2)
    for a, b in zip(serial, parallel):
        assert np.array_equal(a.perm, b.perm)


def test_candidate_edges_positive_and_unique():
    R, _ = planted()
    S = 0.5 * (R + R.T)
    i, j, w = candidate_edges(S, top_k=10)
    assert np.all(i < j) and np.all(w > 0)
    assert len(set(zip(i.tolist(), j.tolist()))) == len(i)


def test_random_pairing():
    p = random_pairing(20, 7, seed=0)
    assert np.array_equal(p[p], np.arange(20))
    assert int((p != np.arange(20)).sum()) == 14
