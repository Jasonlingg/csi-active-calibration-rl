"""Validate feature invariance and the accelerated learner mathematically."""

import numpy as np
import pytest

from active_calibration.learner import KernelRidgeLearner, PreparedLearner, features


def reference_scores(source_f, source_y, target_f, target_y, pool_f, query_f):
    """Direct weighted kernel-ridge solve using the original fixed pool scale."""
    mu = pool_f.mean(axis=0)
    sd = np.maximum(pool_f.std(axis=0), 1e-6)

    def z(values):
        return np.clip((values - mu) / sd, -8.0, 8.0)

    train = np.vstack([z(source_f), z(target_f)])
    labels = np.concatenate([source_y, target_y])
    weights = np.concatenate([np.ones(len(source_f)), np.full(len(target_f), 8.0)])

    def kernel(left, right):
        distance = ((left[:, None, :] - right[None, :, :]) ** 2).sum(axis=2)
        return np.exp(-0.25 * distance / train.shape[1])

    targets = np.eye(8)[labels]
    alpha = np.linalg.solve(kernel(train, train) + np.diag(0.05 / weights), targets)
    return kernel(z(query_f), train) @ alpha


@pytest.mark.parametrize("selected", [[], [7], [7, 0, 4, 2], list(range(9))])
def test_prepared_updates_match_direct_full_solve(selected):
    """Schur updates must preserve scores with correlated and duplicate rows."""
    rng = np.random.default_rng(63)
    source_f = rng.normal(size=(24, 5))
    source_y = np.tile(np.arange(8), 3)
    pool_f = rng.normal(loc=0.5, scale=0.7, size=(9, 5))
    pool_f[-1] = pool_f[0]
    pool_f[:, -1] = 2.0  # The variance floor is part of the scale contract.
    eval_f = rng.normal(size=(7, 5))
    pool_y = np.arange(9) % 8
    indices = np.asarray(selected, dtype=np.int64)
    prepared = PreparedLearner(source_f, source_y, pool_f, eval_f)
    pool_scores, eval_scores = prepared.scores(indices, pool_y[indices])
    expected = reference_scores(
        source_f, source_y, pool_f[indices], pool_y[indices], pool_f,
        np.vstack([pool_f, eval_f]),
    )
    np.testing.assert_allclose(pool_scores, expected[:len(pool_f)], rtol=2e-9, atol=2e-10)
    np.testing.assert_allclose(eval_scores, expected[len(pool_f):], rtol=2e-9, atol=2e-10)
    ordinary = KernelRidgeLearner().fit(
        source_f, source_y, pool_f[indices], pool_y[indices], pool_f
    )
    np.testing.assert_allclose(ordinary.scores(eval_f), eval_scores, rtol=2e-9, atol=2e-10)


def test_querying_does_not_redefine_the_original_pool_standardizer(episode):
    """The same pool must normalize every acquired-label fit, including refits."""
    prepared = PreparedLearner(
        episode.source_f, episode.source_y, episode.pool_f, episode.eval_f
    )
    original = prepared.scores([], np.empty(0, dtype=np.int64))
    prepared.scores([4, 1, 0], episode.pool_y[[4, 1, 0]])
    repeated = prepared.scores([], np.empty(0, dtype=np.int64))
    for before, after in zip(original, repeated):
        np.testing.assert_allclose(before, after, rtol=0, atol=0)
    pool_scores, eval_scores = prepared.scores([4], episode.pool_y[[4]])
    expected = reference_scores(
        episode.source_f, episode.source_y, episode.pool_f[[4]],
        episode.pool_y[[4]], episode.pool_f,
        np.vstack([episode.pool_f, episode.eval_f]),
    )
    np.testing.assert_allclose(
        np.vstack([pool_scores, eval_scores]), expected, rtol=2e-9, atol=2e-10
    )


def test_features_have_fixed_dimension_and_ignore_subcarrier_identity():
    rng = np.random.default_rng(98)
    windows = rng.uniform(0.2, 4.0, size=(5, 128, 56))
    original = features(windows)
    permuted = features(windows[:, :, rng.permutation(56)])
    assert original.shape == (5, 27)
    assert np.isfinite(original).all()
    np.testing.assert_allclose(original, permuted, rtol=1e-12, atol=1e-12)
    constant = features(np.full((2, 128, 56), 3.0))
    assert constant.shape == (2, 27)
    assert np.isfinite(constant).all()
