# Standalone project

Active development now lives in this local Git repository:
`/Users/jasonling/Documents/GitHub/csi-active-calibration-rl`.

The initial prototype was developed under
`terminal-bench/experiments/csi-active-calibration-rl`. It was copied here
with its code, tests, checkpoint, results, and unpublished post draft. All
earlier experiments and submission files remain in the Terminal-Bench
checkout. The original prototype README points here.

## Independent runtime

The eight source/calibration NPZs and the 120-event calibration manifest were
copied byte-for-byte into `data/`. The loader now resolves this project's
root and reads those local files. CLI commands accept `--project-root`;
`--repo-root` remains a compatibility alias. No Terminal-Bench installation,
checkout, hidden test files, or network access is needed to run episodes.

Tests audit NPZ, hash, and metadata reads. They exercise default loading from
an unrelated working directory, allowing only the nine local fixture files.
At migration, the standalone suite passed **23 tests**. Harbor support was
added afterward; its current checks are documented in [HARBOR.md](HARBOR.md).
The receiver F uncertainty demo
also runs from `/private/tmp`, reaching 14/24 correct from 3/24 initially.

The original trained checkpoint was reused without training another policy.
The standalone evaluation repeats the existing fixed seed protocol to check
that moving paths did not change query traces or scores. Its comparison and
copy hashes are recorded in
[`runs/standalone-migration/verification.json`](runs/standalone-migration/verification.json).

## Historical evidence

`runs/2026-10-17-seed0/` is preserved as historical evidence. Its recorded
commands, source hashes, and absolute paths describe the original experiment,
before migration. They should not be interpreted as hashes of the subsequently
adapted standalone path-loading code. Use the current README's commands to
run this project; use the migration verification for the new code hashes.

`provenance/` retains the dataset attribution and the optional raw-data
packing/builder scripts. The included local fixtures are sufficient for this
project; rebuilding them from raw recordings is optional.

## Relationship to a possible benchmark submission

This repository supports the RL environment, capstone work, and writing.
Whether a Terminal-Bench task is submitted is a separate decision. A future
submission can package a fixed, documented version of the environment and
add a task-specific instruction and verifier. The research code can continue
changing here while a submitted benchmark version remains fixed.

No Git remote was added, nothing was published or submitted, and no frontier
model trial was launched during this migration.
