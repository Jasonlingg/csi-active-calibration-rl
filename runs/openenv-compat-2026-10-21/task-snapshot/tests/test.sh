#!/bin/bash
set -euo pipefail
python3 -m active_calibration.harbor_grader --bundle /tests/episode.npz --history /state/query_history.json --out-dir /logs/verifier
