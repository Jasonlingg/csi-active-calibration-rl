#!/usr/bin/env python3
"""Rebuild the seven-target CSI HAR fixtures from two aligned compact NPZs.

The four-receiver compact supplies source and targets A-C. The eight-receiver
compact supplies targets D-G. Both compacts must come from the same raw HAR
recordings; see README.md for the packing and verification commands.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
from pathlib import Path

import numpy as np


FOUR_DEVICES = ("x310", "ESP1", "ESP2", "qca")
EIGHT_DEVICES = FOUR_DEVICES + ("asus1", "asus2", "ax210", "iwl5300")
COUNTS = {"train": 480, "calibration": 120, "test": 120}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_compact(path: Path, expected_devices: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        assert set(archive.files) == {"amplitude", "devices", "session_nr", "labels"}
        amplitude = archive["amplitude"]
        devices = tuple(archive["devices"].tolist())
        sessions = archive["session_nr"]
        labels = archive["labels"]
    assert devices == expected_devices, (path, devices)
    assert amplitude.shape == (len(devices), 720, 128, 56), (path, amplitude.shape)
    assert amplitude.dtype == np.float32 and np.isfinite(amplitude).all(), path
    assert np.array_equal(sessions, np.arange(720)), path
    assert labels.shape == (720,) and labels.dtype == np.int64, path
    return amplitude, labels


def load_split(path: Path, labels: np.ndarray) -> dict[str, np.ndarray]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 720
    assert {row["split"] for row in rows} == set(COUNTS)
    assert {int(row["session_nr"]) for row in rows} == set(range(720))
    assert all(int(row["activity_idx"]) == labels[int(row["session_nr"])] for row in rows)
    for row in rows:
        repetition = int(row["rep_nr"])
        expected = "train" if repetition < 20 else "calibration" if repetition < 25 else "test"
        assert 0 <= repetition < 30 and row["split"] == expected, row
    split_ids = {
        split: np.asarray(
            [int(row["session_nr"]) for row in rows if row["split"] == split],
            dtype=np.int32,
        )
        for split in COUNTS
    }
    assert all(len(split_ids[split]) == count for split, count in COUNTS.items())
    assert len(np.unique(np.concatenate(tuple(split_ids.values())))) == 720
    return split_ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact-four", type=Path, required=True)
    parser.add_argument("--compact-eight", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    four, labels = load_compact(args.compact_four, FOUR_DEVICES)
    eight, eight_labels = load_compact(args.compact_eight, EIGHT_DEVICES)
    assert np.array_equal(labels, eight_labels), "compact labels differ"
    split_ids = load_split(args.split, labels)

    visible = args.out / "environment" / "data"
    hidden = args.out / "tests" / "data"
    public_copy = hidden / "public"
    for directory in (visible, hidden, public_copy):
        directory.mkdir(parents=True, exist_ok=True)

    train_ids = split_ids["train"]
    np.savez_compressed(
        visible / "source.npz",
        x_train=np.ascontiguousarray(four[0, train_ids]),
        y_train=np.ascontiguousarray(labels[train_ids]),
    )

    heldout_labels: dict[str, np.ndarray] = {}
    for device_index in range(1, len(EIGHT_DEVICES)):
        letter = "ABCDEFG"[device_index - 1]
        amplitude = four if device_index < len(FOUR_DEVICES) else eight
        for split, directory, prefix in (
            ("calibration", visible, "target_calib"),
            ("test", hidden, "target_test"),
        ):
            ids = split_ids[split]
            seed = 0xC51 + device_index * 100 + (split == "test")
            shuffled_ids = ids[np.random.default_rng(seed).permutation(len(ids))]
            np.savez_compressed(
                directory / f"{prefix}_{letter}.npz",
                x=np.ascontiguousarray(amplitude[device_index, shuffled_ids]),
            )
            if split == "test":
                heldout_labels[f"y_{letter}"] = np.ascontiguousarray(labels[shuffled_ids])
    np.savez_compressed(hidden / "target_test_labels.npz", **heldout_labels)

    for path in sorted(visible.glob("*.npz")):
        shutil.copyfile(path, public_copy / path.name)
    for path in sorted(args.out.rglob("*.npz")):
        print(f"{path.relative_to(args.out)}  {path.stat().st_size} bytes  sha256:{sha256(path)}")


if __name__ == "__main__":
    main()
