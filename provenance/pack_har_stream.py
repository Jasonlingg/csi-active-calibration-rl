#!/usr/bin/env python3
"""Stream raw 4TU HAR Parquet into aligned, compact real-data CSI windows.

This avoids the release preprocessor's full-file pl.read_parquet on a 16 GB Mac.
It first stores selected devices' first antenna/stream in disk-backed memmaps
indexed by (receiver, session_nr, transmitted sequence number), then retains
only sequence numbers shared by every selected receiver in each event. Uniform
sequence-number subsampling follows the release's preprocessing rule. The NPZ
feeds the fixture builders; no normalization, statistics or train/test labels
are derived from the test signal in this packing step.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


def first_antenna_stream(batch: pa.RecordBatch, column: str, subcarriers: int) -> np.ndarray:
    """Extract the 0th antenna and 0th stream without Python per-row conversion."""
    lane = pc.list_element(pc.list_element(batch.column(column), 0), 0)
    offsets = lane.offsets.to_numpy(zero_copy_only=False)
    lengths = np.diff(offsets)
    if not np.all(lengths == subcarriers):
        raise ValueError(f"Inconsistent {column} lengths: {np.unique(lengths)}")
    values = lane.values.to_numpy(zero_copy_only=False)
    start = int(offsets[0])
    stop = int(offsets[-1])
    return np.asarray(values[start:stop].reshape(len(lane), subcarriers), dtype=np.float32)


def pack(args: argparse.Namespace) -> None:
    meta = pd.read_parquet(args.raw_dir / "meta.parquet")
    devices = list(args.devices)
    if not set(devices).issubset(set(meta.receiver_name)):
        raise ValueError("Unknown receiver in --devices")
    if any(not all(int(v[0]) == 0 for v in meta.loc[meta.receiver_name.eq(d), "antenna_idxs"])
           for d in devices):
        raise ValueError("First antenna is not ID 0")
    subset = meta[meta.receiver_name.isin(devices)].copy()
    if subset.duplicated(["receiver_name", "session_nr"]).any():
        raise ValueError("Duplicate receiver/event metadata")
    events = np.sort(meta.session_nr.unique()).astype(np.int32)
    if len(events) != 720:
        raise ValueError(f"Expected 720 events, got {len(events)}")
    labels_by_event = meta.groupby("session_nr").activity_idx.nunique()
    if not (labels_by_event == 1).all():
        raise ValueError("Activity disagreement within event")
    labels = meta.groupby("session_nr").activity_idx.first().reindex(events).to_numpy(np.int64)
    nsub = meta.subcarrier_idxs.map(len)
    if not (nsub == 56).all():
        raise ValueError("Expected 56 subcarriers for all receivers")
    canonical_sc = tuple(meta.subcarrier_idxs.iloc[0])
    if not all(tuple(x) == canonical_sc for x in meta.subcarrier_idxs):
        raise ValueError("Subcarrier axes differ among receivers")
    lookup = subset.meta_id.tolist()
    value_set = pa.array(lookup)
    d_map = np.array([devices.index(d) for d in subset.receiver_name], dtype=np.int16)
    e_lookup = {int(s): i for i, s in enumerate(events)}
    e_map = np.array([e_lookup[int(s)] for s in subset.session_nr], dtype=np.int16)
    shape = (len(devices), len(events), args.max_sequence, 56)
    scratch = Path(tempfile.mkdtemp(prefix="har-pack-", dir=args.scratch_dir))
    amplitude = np.memmap(scratch / "amplitude.dat", mode="w+", dtype="float32", shape=shape)
    present = np.memmap(scratch / "present.dat", mode="w+", dtype="bool", shape=shape[:3])
    present[:] = False
    total_rows = selected_rows = duplicate_rows = 0
    source = pq.ParquetFile(args.raw_dir / "csi.parquet")
    for batch in source.iter_batches(batch_size=args.batch_size,
                                     columns=["meta_id", "sequence_number", "csi_abs"]):
        total_rows += batch.num_rows
        indices = pc.index_in(batch.column("meta_id"), value_set=value_set)
        mask = pc.is_valid(indices)
        if not pc.any(mask).as_py():
            continue
        chosen = batch.filter(mask)
        idx = pc.filter(indices, mask).to_numpy(zero_copy_only=False).astype(np.int32)
        di, ei = d_map[idx], e_map[idx]
        seq = chosen.column("sequence_number").to_numpy(zero_copy_only=False).astype(np.int32)
        if np.any((seq < 0) | (seq >= args.max_sequence)):
            raise ValueError("Sequence number outside declared packet range")
        existing = present[di, ei, seq]
        duplicate_rows += int(existing.sum())
        key = ((di.astype(np.int64) * len(events) + ei) * args.max_sequence + seq)
        duplicate_rows += len(key) - len(np.unique(key))
        magnitude = first_antenna_stream(chosen, "csi_abs", 56)
        amplitude[di, ei, seq] = magnitude
        present[di, ei, seq] = True
        selected_rows += len(idx)
        if total_rows % (args.batch_size * 100) < args.batch_size:
            print(f"scanned={total_rows:,} selected={selected_rows:,}", flush=True)
    if duplicate_rows:
        raise ValueError(f"Found {duplicate_rows} duplicate (receiver,event,sequence) rows")
    packed_shape = (len(devices), len(events), args.time_steps, 56)
    packed_amp = np.empty(packed_shape, dtype=np.float32)
    common_counts = []
    for event_idx in range(len(events)):
        common = np.flatnonzero(np.all(present[:, event_idx, :], axis=0))
        common_counts.append(len(common))
        if len(common) < args.time_steps:
            raise ValueError(f"Event {events[event_idx]} has only {len(common)} common packets")
        selected = common[np.linspace(0, len(common) - 1, args.time_steps, dtype=int)]
        packed_amp[:, event_idx] = amplitude[:, event_idx, selected, :]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, amplitude=packed_amp,
                        devices=np.asarray(devices), session_nr=events, labels=labels)
    print(f"wrote {args.output} ({args.output.stat().st_size / 1e6:.1f} MB)")
    print(f"rows scanned={total_rows:,}, selected={selected_rows:,}, "
          f"common sequence count min/median/max={min(common_counts)}/"
          f"{int(np.median(common_counts))}/{max(common_counts)}")
    del amplitude, present
    shutil.rmtree(scratch)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scratch-dir", type=Path, required=True)
    parser.add_argument("--devices", nargs="+", default=["x310", "ESP1", "ESP2", "qca"])
    parser.add_argument("--time-steps", type=int, default=128)
    parser.add_argument("--max-sequence", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=8192)
    args = parser.parse_args()
    args.scratch_dir.mkdir(parents=True, exist_ok=True)
    pack(args)


if __name__ == "__main__":
    main()
