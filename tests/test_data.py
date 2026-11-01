"""Audit recording splits and ensure the project uses only its local fixtures."""

import builtins
import csv
from os import PathLike
from pathlib import Path

import numpy as np

import active_calibration.data as data_module
from active_calibration.data import load_episodes
from conftest import PROJECT_ROOT


def test_recording_splits_are_disjoint_and_read_only_local_fixtures(monkeypatch, tmp_path):
    """Default loading must work independently of cwd and any former repository."""
    receivers = tuple("ABCDEFG")
    expected_directory = PROJECT_ROOT / "data"
    allowed_npz = {expected_directory / "source.npz"} | {
        expected_directory / f"target_calib_{receiver}.npz" for receiver in receivers
    }
    metadata_path = expected_directory / "calibration_event_index.csv"
    allowed_npz = {path.resolve() for path in allowed_npz}
    allowed_reads = allowed_npz | {metadata_path.resolve()}
    loaded_paths = []
    opened_paths = []
    original_load = np.load
    original_path_open = Path.open
    original_builtin_open = builtins.open

    def audit_read(file, mode):
        path = Path(file).resolve()
        assert path in allowed_reads, f"Loader read outside project fixtures: {path}"
        assert not any(flag in mode for flag in "wax+"), "Loader must not write fixtures"
        opened_paths.append(path)

    def audited_path_open(path, mode="r", *args, **kwargs):
        audit_read(path, mode)
        return original_path_open(path, mode, *args, **kwargs)

    def audited_builtin_open(file, mode="r", *args, **kwargs):
        if isinstance(file, (str, PathLike)):
            audit_read(file, mode)
        return original_builtin_open(file, mode, *args, **kwargs)

    def audited_load(file, *args, **kwargs):
        path = Path(file).resolve()
        assert path in allowed_npz, f"Loader used an unexpected recording file: {path}"
        loaded_paths.append(path)
        return original_load(file, *args, **kwargs)

    # Ensure previous test calls cannot satisfy this read audit from cache.
    for loader in (data_module._load_source, data_module._event_rows, data_module._load_receiver):
        loader.cache_clear()
    monkeypatch.chdir(tmp_path)
    with monkeypatch.context() as guarded:
        guarded.setattr(np, "load", audited_load)
        guarded.setattr(Path, "open", audited_path_open)
        guarded.setattr(builtins, "open", audited_builtin_open)
        train = load_episodes(split="train", receivers=receivers)
        evaluation = load_episodes(project_root=PROJECT_ROOT, split="eval", receivers=receivers)

    assert set(loaded_paths) == allowed_npz
    assert set(opened_paths) == allowed_reads
    assert len(train) == len(evaluation) == 7
    assert {episode.receiver for episode in train} == set(receivers)
    assert {episode.receiver for episode in evaluation} == set(receivers)

    with metadata_path.open(newline="") as stream:
        metadata = {int(row["session_nr"]): row for row in csv.DictReader(stream)}

    roles = {"train_pool": set(), "train_reward": set(),
             "eval_pool": set(), "eval_reward": set()}
    for split, episodes, expected_reps in [
        ("train", train, (20, 21)), ("eval", evaluation, (23, 24))
    ]:
        for episode in episodes:
            assert episode.split == split
            assert episode.source_f.shape == (480, 27)
            assert episode.source_y.shape == (480,)
            assert episode.pool_f.shape == episode.eval_f.shape == (24, 27)
            assert episode.pool_y.shape == episode.eval_y.shape == (24,)
            for values in [episode.source_f, episode.pool_f, episode.eval_f]:
                assert np.isfinite(values).all()
            for values in [episode.source_y, episode.pool_y, episode.eval_y]:
                assert set(np.unique(values)) == set(range(8))
            for ids, labels, role, rep in [
                (episode.pool_event_ids, episode.pool_y, "pool", expected_reps[0]),
                (episode.eval_event_ids, episode.eval_y, "reward", expected_reps[1]),
            ]:
                assert len(ids) == len(set(ids)) == 24
                assert {int(metadata[int(event)]["rep_nr"]) for event in ids} == {rep}
                np.testing.assert_array_equal(
                    labels,
                    [int(metadata[int(event)]["activity_idx"]) for event in ids],
                )
                roles[f"{split}_{role}"].update(ids)
            np.testing.assert_array_equal(episode.source_f, train[0].source_f)
            np.testing.assert_array_equal(episode.source_y, train[0].source_y)

    assert all(len(events) == 24 for events in roles.values())
    assert len(set.union(*roles.values())) == sum(map(len, roles.values()))
