# Recording provenance and optional raw-data regeneration

This standalone project uses the `wise-har` recordings in Portner, Fabian;
Asadi, Arash; Gringoli, Francesco; Hollick, Matthias (2025), *Data and Code
for “Same Signal, Different Story: Demystifying Receiver Effects in Wi-Fi
Channel State Information”*, Version 3, 4TU.ResearchData,
[DOI 10.4121/48e1331b-91df-4e1a-906b-b0a0e9d16c74.v3](https://doi.org/10.4121/48e1331b-91df-4e1a-906b-b0a0e9d16c74.v3).
The [version 3 record](https://data.4tu.nl/datasets/48e1331b-91df-4e1a-906b-b0a0e9d16c74/3)
lists the licence as MIT and the `wise-har.zip` MD5 as
`86ba0981d4b96d03b8491525175d58ff`. Preserve this attribution when using or
redistributing the derived recordings.

## Files used by this project

The local `data/` directory contains eight NPZ files and one CSV:

- `source.npz`: 480 labeled source-receiver windows.
- `target_calib_A.npz` through `target_calib_G.npz`: 120 target-receiver
  windows each. Their activity labels are recovered from the event manifest
  by the environment, and revealed to a policy only after a label query.
- `calibration_event_index.csv`: the 120 calibration events, their activity
  IDs, setting IDs, repetition numbers, and split membership.

These are copies of the source and calibration fixtures from the earlier
CSI HAR experiment. The standalone loader reads only these nine local files.
It does not read the earlier experiment's test recordings or test-label archive.
Expected SHA-256 hashes for all nine local files are recorded and checked in
[`active_calibration/data.py`](../active_calibration/data.py).

The original split assigned repetitions 0–19 to source training, 20–24 to
calibration, and 25–29 to the earlier test set, within each activity and
setting. It contains 480/120/120 distinct physical events. Each event has
the same `session_nr` on all eight synchronized receivers, so the split is
global across receivers.

The RL episodes further divide the 120 calibration events: repetitions 20/21
are the training query/reward sets, repetitions 23/24 are the evaluation
query/reward sets, and repetition 22 is unused. Each repetition supplies 24
events. Training defaults to receivers A–E; evaluation includes A–G. The
recordings remain public development data, rather than a newly collected or
prospectively untouched dataset.

## Window construction and receivers

[`pack_har_stream.py`](pack_har_stream.py) selects the first antenna and
stream, intersects packet sequence numbers across the selected receivers
within each event, and samples 128 positions uniformly from that intersection.
It keeps 56 subcarriers and stores float32 amplitudes.

The source (`x310`) and targets A–C come from a four-receiver compact containing
`x310`, `ESP1`, `ESP2`, and `qca`. Targets D–G come from an eight-receiver compact
that also contains `asus1`, `asus2`, `ax210`, and `iwl5300`. The two compacts
therefore use different shared-packet intersections; the corresponding windows
must not be described as exactly packet-aligned across all eight receivers.

| Local receiver | Recording receiver | Compact used |
| --- | --- | --- |
| Source | x310 | Four receivers |
| A | ESP1 | Four receivers |
| B | ESP2 | Four receivers |
| C | qca | Four receivers |
| D | asus1 | Eight receivers |
| E | asus2 | Eight receivers |
| F | ax210 | Eight receivers |
| G | iwl5300 | Eight receivers |

The fixture builder independently shuffles each receiver and partition using
fixed seeds. The loader reconstructs that ordering from the event manifest.

## Optional regeneration from the original raw release

The supplied local files are sufficient to run the RL project. A raw rebuild
additionally requires the original `meta.parquet`, `csi.parquet`, and the full
**720-row** `har_event_split.csv`. These raw inputs and the full split CSV are
not included here. The local **120-row** `data/calibration_event_index.csv`
is not a valid value for the builder's `--split` argument.

Recorded SHA-256 hashes of the original regeneration inputs are:

| Input | SHA-256 |
| --- | --- |
| `meta.parquet` | `72c1bb52fa02f497676a94b655acd5dbebbbdefd7837171bb5041ac9679f1ef6` |
| `csi.parquet` | `37bcf1578abe3f038ac47e837c7701d312ec04741262ff95eaa2ed0cf8b90bf7` |
| Full `har_event_split.csv` | `899aa30736bd744d678699c3273863811d1178d0f17a8b83052ab10671cb9023` |

The following commands run from this standalone project's root and keep the
generated files in a fresh temporary directory. Replace the two `/path/to/`
placeholders with the original inputs. The historical packing environment used
PyArrow 26.0.0, pandas 3.0.6, and NumPy 2.5.4; the fixture builder used NumPy
2.1.2. These optional regeneration environments are separate from the RL
runtime dependencies.

```sh
CSI_PROJECT="$PWD"
CSI_RAW=/path/to/wise-har
CSI_SPLIT=/path/to/har_event_split.csv
CSI_REBUILD="$(mktemp -d "${TMPDIR:-/tmp}/csi-fixtures.XXXXXX")"

uv run --no-project --with pyarrow==26.0.0 --with pandas==3.0.6 \
  --with numpy==2.5.4 python \
  provenance/pack_har_stream.py \
  --raw-dir "$CSI_RAW" --scratch-dir "$CSI_REBUILD/scratch" \
  --output "$CSI_REBUILD/four-receivers.npz" \
  --devices x310 ESP1 ESP2 qca

uv run --no-project --with pyarrow==26.0.0 --with pandas==3.0.6 \
  --with numpy==2.5.4 python \
  provenance/pack_har_stream.py \
  --raw-dir "$CSI_RAW" --scratch-dir "$CSI_REBUILD/scratch" \
  --output "$CSI_REBUILD/eight-receivers.npz" \
  --devices x310 ESP1 ESP2 qca asus1 asus2 ax210 iwl5300

uv run --no-project --with numpy==2.1.2 python \
  provenance/build_fixtures_v4.py \
  --compact-four "$CSI_REBUILD/four-receivers.npz" \
  --compact-eight "$CSI_REBUILD/eight-receivers.npz" \
  --split "$CSI_SPLIT" --out "$CSI_REBUILD/fixtures"

(cd "$CSI_REBUILD/fixtures" && \
  shasum -a 256 -c "$CSI_PROJECT/provenance/SHA256SUMS.v4")
```

The recorded compact SHA-256 hashes are
`1ca600999c99ce40e9d7e631afd6427a2b2d54816616919e671be5c571c0394b`
for four receivers and
`a2a5da99c8c2b62125f97f331aa5907fa50e3f0a3eba9d9dafe1eaa0cffa8193`
for eight receivers. A historical isolated rebuild matched all 24 original
fixture NPZs byte-for-byte; no raw rebuild is implied by the standalone copy.

The copied builder retains the **original fixture layout** under its temporary
output directory: `environment/data/`, `tests/data/public/`, and `tests/data/`.
The historical [`SHA256SUMS.v4`](SHA256SUMS.v4) covers all 24 of those files:
eight source/calibration NPZs, eight byte-identical verifier copies, seven
earlier test inputs, and one earlier test-label archive. It is not a manifest
of this standalone project's nine local data files.

After checking the regenerated hashes, the optional commands below replace
only the eight source/calibration NPZs used by this project. Keep the supplied
calibration event CSV. The temporary builder's test data and test-label archive
are not copied into the project.

```sh
cp "$CSI_REBUILD/fixtures/environment/data/source.npz" data/source.npz
for CSI_RECEIVER in A B C D E F G; do
  cp "$CSI_REBUILD/fixtures/environment/data/target_calib_${CSI_RECEIVER}.npz" \
    "data/target_calib_${CSI_RECEIVER}.npz"
done
```
