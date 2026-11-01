"""Build development episodes from this project's local CSI recordings.

The loader reads only the nine files in the project's data directory.
Calibration labels come from the recording event manifest and are retained by
the environment, not included in policy observations before a query.
"""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np

from .learner import features


RECEIVERS = tuple("ABCDEFG")
TRAIN_RECEIVERS = tuple("ABCDE")
SPLIT_REPETITIONS = {"train": (20, 21), "eval": (23, 24)}
EXPECTED_SHA256 = {
    "calibration_event_index.csv": "24e36f333bb6c28aed0531b48d240b43ac2bb3cc65f18d6766317948a00c8d49",
    "source.npz": "0e510d97edbee36a75f5c3231f6108275822aab152d5ac2208b33a8bfa4b7b14",
    "target_calib_A.npz": "575c10f5e227c533ae398607564444f01e93c153bcc8cfc36e329e5058821634",
    "target_calib_B.npz": "8bdcbf7434848957282bf21a9649d576921fec0e0b2af38fb7229c931f9b792c",
    "target_calib_C.npz": "a218d66d9db6d6a7f17ec3d0a4cf1d44182a126e33aab2f93f8f2994f667aeb5",
    "target_calib_D.npz": "9f46127951f709591815b53b03e3ec4fc1001395500ce40906737fc1509911d9",
    "target_calib_E.npz": "48a8f8f6085092fa21d9a86d33e8fd080e3a43d2a2e754bcaa7e530997964d85",
    "target_calib_F.npz": "9656b341f4eb04ef1ebd8447d02b23a6f0f96f73fc902dcd20bb95ab6e190dd0",
    "target_calib_G.npz": "f2014c2aced3363aa39e415a9e62a4f8d7a27243681cd6f55cdf4917f69f6594",
}


@dataclass(frozen=True)
class Episode:
    source_f: np.ndarray
    source_y: np.ndarray
    pool_f: np.ndarray
    pool_y: np.ndarray
    eval_f: np.ndarray
    eval_y: np.ndarray
    receiver: str
    split: str
    pool_event_ids: tuple[int, ...]
    eval_event_ids: tuple[int, ...]


def default_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_repo_root() -> Path:
    """Compatibility alias for the standalone project root."""
    return default_project_root()


def _readonly(array: np.ndarray, dtype: np.dtype) -> np.ndarray:
    result = np.array(array, dtype=dtype, copy=True)
    result.flags.writeable = False
    return result


def _check_hash(path: Path) -> None:
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != EXPECTED_SHA256[path.name]:
        raise ValueError(f"Fixture changed: {path}; event alignment must be revalidated")


@lru_cache(maxsize=4)
def _load_source(project_root: Path) -> tuple[np.ndarray, np.ndarray]:
    path = project_root / "data/source.npz"
    _check_hash(path)
    with np.load(path, allow_pickle=False) as data:
        source_f = _readonly(features(data["x_train"]), np.float64)
        source_y = _readonly(data["y_train"], np.int64)
    if source_f.shape != (480, 27) or source_y.shape != (480,):
        raise ValueError("Expected the 480-example source fixture")
    return source_f, source_y


@lru_cache(maxsize=4)
def _event_rows(project_root: Path) -> tuple[tuple[int, int, int], ...]:
    path = project_root / "data/calibration_event_index.csv"
    _check_hash(path)
    with path.open(newline="", encoding="utf-8") as stream:
        rows = tuple(
            (int(row["session_nr"]), int(row["activity_idx"]), int(row["rep_nr"]))
            for row in csv.DictReader(stream) if row["split"] == "calibration"
        )
    if len(rows) != 120 or len({row[0] for row in rows}) != 120:
        raise ValueError("Expected 120 distinct calibration recording events")
    if {row[2] for row in rows} != set(range(20, 25)):
        raise ValueError("Unexpected calibration repetitions")
    return rows


@lru_cache(maxsize=28)
def _load_receiver(project_root: Path, receiver: str, split: str) -> Episode:
    source_f, source_y = _load_source(project_root)
    base_rows = _event_rows(project_root)
    device_index = RECEIVERS.index(receiver) + 1
    # Same row permutation as provenance/build_fixtures_v4.py.
    permutation = np.random.default_rng(0xC51 + device_index * 100).permutation(120)
    rows = [base_rows[int(i)] for i in permutation]
    pool_rep, eval_rep = SPLIT_REPETITIONS[split]
    pool_indices = np.array([i for i, row in enumerate(rows) if row[2] == pool_rep])
    eval_indices = np.array([i for i, row in enumerate(rows) if row[2] == eval_rep])
    path = project_root / f"data/target_calib_{receiver}.npz"
    _check_hash(path)
    with np.load(path, allow_pickle=False) as data:
        target_f = features(data["x"])
    if target_f.shape != (120, 27):
        raise ValueError(f"Unexpected target shape for receiver {receiver}")
    pool_y = np.array([rows[i][1] for i in pool_indices], dtype=np.int64)
    eval_y = np.array([rows[i][1] for i in eval_indices], dtype=np.int64)
    if len(pool_y) != 24 or len(eval_y) != 24:
        raise ValueError("Each episode requires 24 query and 24 reward examples")
    if not all(np.array_equal(np.bincount(y, minlength=8), np.full(8, 3))
               for y in (pool_y, eval_y)):
        raise ValueError("Expected three examples of each activity in each partition")
    pool_ids = tuple(rows[i][0] for i in pool_indices)
    eval_ids = tuple(rows[i][0] for i in eval_indices)
    if set(pool_ids) & set(eval_ids):
        raise ValueError("Query and reward recording events overlap")
    return Episode(
        source_f=source_f, source_y=source_y,
        pool_f=_readonly(target_f[pool_indices], np.float64),
        pool_y=_readonly(pool_y, np.int64),
        eval_f=_readonly(target_f[eval_indices], np.float64),
        eval_y=_readonly(eval_y, np.int64), receiver=receiver, split=split,
        pool_event_ids=pool_ids, eval_event_ids=eval_ids,
    )


def load_episodes(
    project_root: Path | str | None = None,
    split: str = "train",
    receivers: Iterable[str] | None = None,
    *, repo_root: Path | str | None = None,
) -> list[Episode]:
    """Load policy-training A–E or evaluation A–G episodes.

    Repetitions 20/21 are query/reward events in training. Repetitions 23/24
    are query/reward events in evaluation. Repetition 22 is unused. A physical
    event has the same ID on all receivers, so this split is global, rather
    than independently shuffling each receiver's correlated recordings.
    """
    if split not in SPLIT_REPETITIONS:
        raise ValueError("split must be 'train' or 'eval'")
    if project_root is not None and repo_root is not None:
        raise TypeError("provide project_root or its compatibility alias repo_root, not both")
    explicit_root = project_root if project_root is not None else repo_root
    root = (Path(explicit_root) if explicit_root is not None else default_project_root()).resolve()
    selected = tuple(receivers) if receivers is not None else (
        TRAIN_RECEIVERS if split == "train" else RECEIVERS
    )
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("receivers must be a nonempty sequence without duplicates")
    if any(receiver not in RECEIVERS for receiver in selected):
        raise ValueError(f"receivers must be drawn from {RECEIVERS}")
    return [_load_receiver(root, receiver, split) for receiver in selected]
