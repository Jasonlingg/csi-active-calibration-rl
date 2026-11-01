"""Package a CSI label-query episode as a self-contained Harbor task."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np

from .data import TRAIN_RECEIVERS, default_project_root, load_episodes


BASE_IMAGE = "python:3.11-slim@sha256:da047cb8f9d1d98e5c070f5300ba9f7274e33b8fc0e5be5ed88740aed1b95ba9"
RUNTIME_MODULES = ("__init__.py", "data.py", "learner.py", "env.py")

TASK_CONFIG = '''schema_version = "1.4"
artifacts = [{ source = "/state/query_history.json", service = "calibration" }]

[task]
name = "csi-calibration/label-selection"
version = "0.1.0"
description = "Choose 16 activity labels to improve a fixed CSI classifier."

[metadata]
category = "ML"
tags = ["wifi", "csi", "active-learning", "rl"]

[verifier]
timeout_sec = 120.0
environment_mode = "separate"

[verifier.environment]
build_timeout_sec = 600.0
cpus = 1
memory_mb = 1024
storage_mb = 2048

[agent]
timeout_sec = 900.0

[environment]
build_timeout_sec = 600.0
network_mode = "public"
cpus = 1
memory_mb = 2048
storage_mb = 4096
'''

INSTRUCTION = '''Improve a fixed Wi-Fi activity classifier by choosing which target recordings
to label. Request exactly 16 distinct labels from the 24-row query pool.

Start with `csi-calibration state --compact --features` to inspect the pool.
Then use `csi-calibration query INDEX --compact` to label one available row
and read its updated scores. These commands return JSON. The compact view
lists available rows as [index, class_0_score, ..., class_7_score], and revealed
labels as [index, label]. Use the explicit index, not a row's position in the
shrinking list. Features are static, so request them once. Displayed numbers
are rounded to four decimals; omit --compact for the full-precision observation.
The service URL is in `CSI_CALIBRATION_URL`; its public API is
`GET /state` and `POST /query` with body `{"index": INDEX}`. There is one
episode and no reset. The service owns the classifier and enforces the budget.
You may automate your query strategy.

The observation contains normalized window features, eight classifier scores
per row, requested labels (`-1` for unknown), action and queried masks, and
remaining budget as a fraction. Classes 0–7 are empty room, jumping, standing,
walking, running, squat, clapping, and waving arms. Scores are raw kernel-ridge
outputs. Each label request updates the classifier's scores. Previously
requested rows are unavailable.

The final reward is accuracy after your 16 requests minus source-only accuracy
on 24 separate recording events. It can be negative. An incomplete or invalid
episode receives -1. Reward examples and their labels are private to the
service and verifier. Maximize this accuracy gain through your query choices.
No prediction program or answer file is required: the service records your
accepted requests, and Harbor collects that history for verification.

Continue taking actions until the service reports `terminated: true`,
`queries_used: 16`, and `queries_remaining: 0`. A plan or final text response
does not request labels. Each query response already contains the updated
state; you do not need to fetch state again between requests. After the last
query, report completion and stop. Keep terminal output concise.
'''

COMPOSE = '''services:
  main:
    depends_on:
      calibration:
        condition: service_healthy
    environment:
      CSI_CALIBRATION_URL: http://calibration:8080
  calibration:
    build:
      context: ./calibration
    expose:
      - "8080"
    tmpfs:
      - /tmp:size=16m
    cpus: 1
    mem_limit: 512m
    healthcheck:
      test: ["CMD", "python3", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8080/health', timeout=2).read()"]
      interval: 2s
      timeout: 3s
      retries: 30
'''

REFERENCE = '''"""A smallest-margin baseline; this is not an optimal query oracle."""
from harbor_client import fetch_state, query

state = fetch_state()
while not state["terminated"]:
    observation = state["observation"]
    candidates = []
    for index, valid in enumerate(observation["action_mask"]):
        if valid:
            ordered = sorted(observation["scores"][index])
            candidates.append((ordered[-1] - ordered[-2], index))
    _, selected = min(candidates)
    state = query(selected)
print("Completed", state["queries_used"], "requests; reward", state["reward"])
'''


def _write(path: Path, text: str, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    if executable:
        path.chmod(0o755)


def build_task(output: Path, project_root: Path | str | None = None,
               receiver: str = "F", seed: int = 0, budget: int = 16,
               *, split: str = "eval") -> Path:
    """Build once into a new directory; never overwrite an existing task."""
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    if type(budget) is not int or budget != 16:
        raise ValueError("the Harbor task contract requires an integer budget of exactly 16 queries")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if split not in ("train", "eval"):
        raise ValueError("split must be 'train' or 'eval'")
    if split == "train" and receiver not in TRAIN_RECEIVERS:
        raise ValueError("training tasks require a receiver from A-E; F/G are evaluation-only")
    episodes = load_episodes(project_root, split=split, receivers=[receiver])
    episode = episodes[0]
    source_modules = Path(__file__).resolve().parent
    required = (*RUNTIME_MODULES, "harbor_client.py", "harbor_service.py", "harbor_grader.py")
    if any(not (source_modules / name).is_file() for name in required):
        raise FileNotFoundError("Harbor adapter modules are incomplete")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Stage the complete task before making it visible at the requested path.
    with tempfile.TemporaryDirectory(prefix=".csi-harbor-", dir=output.parent) as temporary:
        task = Path(temporary) / "task"
        private = task / "environment/calibration"
        verifier = task / "tests"
        private.mkdir(parents=True)
        verifier.mkdir()
        np.savez_compressed(
            private / "episode.npz", source_f=episode.source_f, source_y=episode.source_y,
            pool_f=episode.pool_f, pool_y=episode.pool_y, eval_f=episode.eval_f,
            eval_y=episode.eval_y, pool_event_ids=np.asarray(episode.pool_event_ids),
            eval_event_ids=np.asarray(episode.eval_event_ids), receiver=episode.receiver,
            split=episode.split, seed=seed, budget=budget,
        )
        shutil.copy2(private / "episode.npz", verifier / "episode.npz")
        for destination, extra in ((private, "harbor_service.py"), (verifier, "harbor_grader.py")):
            package = destination / "active_calibration"
            package.mkdir()
            for name in (*RUNTIME_MODULES, extra):
                shutil.copy2(source_modules / name, package / name)
        shutil.copy2(source_modules / "harbor_client.py", task / "environment/client.py")
        solution = task / "solution"
        solution.mkdir()
        shutil.copy2(source_modules / "harbor_client.py", solution / "harbor_client.py")
        _write(task / "task.toml", TASK_CONFIG)
        _write(task / "instruction.md", INSTRUCTION)
        _write(task / "environment/docker-compose.yaml", COMPOSE)
        _write(task / "environment/Dockerfile", f'''FROM {BASE_IMAGE}
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates git && rm -rf /var/lib/apt/lists/*
COPY --chmod=755 client.py /usr/local/bin/csi-calibration
WORKDIR /app
RUN mkdir -p /app /logs/artifacts
''')
        runtime = f'''FROM {BASE_IMAGE}
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/opt/csi OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
RUN pip install --no-cache-dir numpy==2.4.3 gymnasium==1.4.0
COPY active_calibration/ /opt/csi/active_calibration/
'''
        _write(private / "Dockerfile", runtime + '''COPY episode.npz /srv/private/episode.npz
RUN mkdir -p /state
CMD ["python3", "-m", "active_calibration.harbor_service", "--bundle", "/srv/private/episode.npz", "--history", "/state/query_history.json", "--host", "0.0.0.0", "--port", "8080"]
''')
        _write(verifier / "Dockerfile", runtime + '''COPY episode.npz /tests/episode.npz
COPY test.sh /tests/test.sh
RUN mkdir -p /state /logs/verifier
WORKDIR /tests
''')
        _write(verifier / "test.sh", '''#!/bin/bash
set -euo pipefail
python3 -m active_calibration.harbor_grader --bundle /tests/episode.npz --history /state/query_history.json --out-dir /logs/verifier
''', executable=True)
        _write(solution / "solve.sh", '#!/bin/bash\nset -euo pipefail\npython3 /solution/run_baseline.py\n', executable=True)
        _write(solution / "run_baseline.py", REFERENCE)
        hashes = {str(path.relative_to(task)): hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in sorted(task.rglob("*")) if path.is_file()}
        _write(task / "build_manifest.json", json.dumps({
            "receiver": receiver, "split": split, "seed": seed, "budget": budget,
            "project_root": str(Path(project_root).resolve()) if project_root else str(default_project_root()),
            "harbor_version_target": "0.24.0", "sha256": hashes,
            "purpose": "Interactive research task; not a Terminal-Bench submission or proven cheat-resistant benchmark",
        }, indent=2) + "\n")
        task.rename(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("harbor_tasks/csi-active-calibration"))
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--receiver", choices=tuple("ABCDEFG"), default="F")
    parser.add_argument("--split", choices=("train", "eval"), default="eval",
                        help="recording split; training tasks require --receiver A-E")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    result = build_task(args.out, args.project_root, args.receiver, args.seed, split=args.split)
    print(f"Built {result}")
    print(f"harbor run -p {result} -a oracle -o {Path('harbor_runs').resolve()} --job-name scripted-reference")


if __name__ == "__main__":
    main()
