"""Fixtures for the standalone active-calibration prototype."""

from pathlib import Path
import sys

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from active_calibration.data import Episode  # noqa: E402


@pytest.fixture
def episode():
    """Small deterministic episode, independent of the recordings on disk."""
    rng = np.random.default_rng(301)
    return Episode(
        source_f=rng.normal(size=(24, 5)),
        source_y=np.tile(np.arange(8), 3),
        pool_f=rng.normal(size=(6, 5)),
        pool_y=np.array([0, 3, 7, 2, 5, 1]),
        eval_f=rng.normal(size=(7, 5)),
        eval_y=np.array([0, 1, 2, 3, 4, 5, 6]),
        receiver="synthetic",
        split="train",
        pool_event_ids=np.arange(100, 106),
        eval_event_ids=np.arange(200, 207),
    )


def assert_observations_equal(left, right):
    assert left.keys() == right.keys()
    for key in left:
        np.testing.assert_array_equal(left[key], right[key], err_msg=key)
