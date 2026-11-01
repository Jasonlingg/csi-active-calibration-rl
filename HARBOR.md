# Run interactive calibration with Harbor

The standalone project now includes a Harbor task. A terminal agent chooses
which recordings to label through a command-line client. The private service
runs the same fixed learner and Gymnasium transition logic as the Python
environment. Harbor manages containers, agent execution, the saved query
history, and a separate verifier.

This supports an interactive terminal agent as well as the existing local
Torch-policy experiments. It does not require a Terminal-Bench checkout or
submission. Nothing has been published. The first model-agent attempts used
the optional Verifiers adapter; see the [LLM smoke record](runs/llm-smoke-2026-10-26/README.md).

## Run the packaged task

Use Docker with Compose and **Harbor 0.24.0**, the version checked locally.
Harbor itself requires Python 3.12 or newer; the CSI environment and its
containers use Python 3.11. Install Harbor as a separate tool so those Python
requirements do not interfere:

```sh
uv tool install 'harbor==0.24.0' --python 3.13
harbor --version
docker version
```

From the standalone project directory, run the included scripted baseline:

If `harbor_tasks/csi-active-calibration/` is absent in a fresh checkout, first
generate it with `.venv/bin/python -m active_calibration.harbor_task`. Generated
task bundles and Harbor logs stay local and are excluded from Git by default.

```sh
harbor run \
  -p "$PWD/harbor_tasks/csi-active-calibration" \
  -a oracle \
  --jobs-dir "$PWD/harbor_runs" \
  --job-name my-scripted-reference --n-concurrent 1
```

Use a new job name for another run. The `oracle` agent executes the provided
script; it makes no language-model call. Here that script implements the
smallest-score-margin baseline, not an optimal query strategy. The negative
control uses `-a nop` and another job name.

Supply an **absolute jobs directory**, as above. The first local packaging
check exposed relative-path problems in Harbor's Docker Compose artifact-copy
step. The working commands use absolute task and jobs paths.

The packaged episode is receiver F, evaluation split, seed 0. To generate a
different receiver or row permutation into a new, self-contained directory:

```sh
.venv/bin/python -m active_calibration.harbor_task \
  --receiver G --seed 1 --out /tmp/csi-calibration-G-seed1
```

To prepare a task for future LLM training, explicitly select the training
recordings and a receiver from A-E:

```sh
.venv/bin/python -m active_calibration.harbor_task \
  --split train --receiver A --seed 0 --out /tmp/csi-calibration-train-A-seed0
```

This command only packages a task; it does not run or fine-tune a model.
Training tasks use repetitions 20/21, while the default `--split eval` uses
23/24. F/G are rejected for training. The manifest records the selected split,
and the service and verifier receive the same frozen episode. Row-shuffle
seeds reuse the recordings and must not be counted as independent data.
The small Torch-policy trainer is separate from an LLM fine-tuning pipeline;
the latter has not yet been implemented or run in this project.

The builder refuses to overwrite an existing directory. It freezes features,
labels, the seed, and the 16-query budget; copies the needed runtime code; and
records all generated input hashes in `build_manifest.json`. The resulting
task can run independently of the project that generated it. Its main agent
image includes only the standard-library client. Private episode bundles are
copied into the service and verifier images.

## What the agent does

Inside the Harbor agent container:

```sh
csi-calibration state
csi-calibration query 13
```

For a smaller LLM transcript, newly generated tasks recommend:

```sh
csi-calibration state --compact --features
csi-calibration query 13 --compact
```

The compact JSON lists available rows as `[index, class_0_score, ...,
class_7_score]`, and revealed labels as `[index, label]`. Indices retain their
original meaning when queried rows disappear from the available list. Scores
and features are rounded to four decimal places for display; classifier
updates and rewards retain full precision. Request static features once with
`--features`; later query responses include updated scores and integer counts
of queries used and remaining. The original commands and HTTP API still return
the full observation. Stop only after `terminated` is true and 16 queries are
recorded. A final text response cannot complete the episode.

Existing generated bundles retain their copied client and instructions.
Build into a new directory to use this interface; earlier runs are preserved.
See [the CPU and Docker checks](runs/llm-interface-2026-10-28/README.md).

Each command prints JSON. `state` returns the current features, classifier
scores, requested labels, masks, and budget. `query INDEX` requests one label
and returns the updated observation. The URL is
`http://calibration:8080`, also set in `CSI_CALIBRATION_URL`. An explicit
`--url` option overrides it.

The service exposes `GET /health`, `GET /state`, and `POST /query`. There is
one episode per trial, with no reset, new-session, arbitrary-file, or
reward-data endpoint. Invalid or repeated requests do not consume budget;
concurrent requests are serialized. The client can be used from a script or
the agent can make decisions between commands. No answer file or prediction
program needs to be submitted.

After 16 valid requests, the service ends the episode and returns the terminal
reward. It records accepted action indices in an atomic private journal.
Process recovery replays an existing journal without replenishing the budget.
Deleting or recreating containers is controlled by Harbor, not exposed as an
agent action.

## Grading and reward

The task declares one authoritative artifact:

```toml
artifacts = [{ source = "/state/query_history.json", service = "calibration" }]

[verifier]
environment_mode = "separate"
```

Harbor stops the agent and collects that file directly from the service's
filesystem. The agent cannot substitute its own answer file for this service
artifact. The verifier starts in a fresh container with its own frozen episode
bundle and the copied history. It validates the bundle hash, seed, budget,
and exactly 16 distinct legal actions, then replays them from a clean reset.
It accepts no claimed scores or labels from the agent.

For a completed valid episode:

```text
Harbor reward = final accuracy − source-only accuracy
```

This is the same terminal reward as the Gymnasium environment, including
negative gains. An invalid or incomplete trial receives **−1**, the minimum
possible accuracy-gain reward. It is a continuous research objective with no
classification cutoff or binary Terminal-Bench qualification rule.

`reward.txt` and `report.json` appear under each trial's `verifier/` directory.
The trusted journal and its collection manifest are in `artifacts/`.

## Checked locally

- The Python suite passes **60 tests**, including training/evaluation task
  separation and builder budget validation,
  HTTP budget enforcement, action-journal replay, hidden-label separation, invalid-history rejection,
  and a client running without site packages or project data.
- The fixed scripted reference completed all 16 requests on receiver F:
  **3/24 → 14/24 correct**, reward **0.45833333333333337**, zero Harbor
  exceptions. The trusted sidecar artifact was collected successfully.
- The no-action control produces an incomplete episode and reward **−1**.

Evidence is under `harbor_runs/smoke-reference-fixed/` and
`harbor_runs/smoke-nop/`, with a compact hash/result record in
[`runs/harbor-smoke-2026-10-17/verification.json`](runs/harbor-smoke-2026-10-17/verification.json).
The initial packaging failure and exact original task
snapshot remain under `harbor_runs/smoke-reference/`; that failure involved no
model. It exposed a Python module-name collision and the artifact handoff.
The reference script now uses a nonconflicting filename, the journal lives in
the service's writable filesystem, and run instructions use absolute paths.

The Harbor smoke result is one receiver episode. The seven-receiver local
Torch-policy comparison in the main README is a separate experiment.

An additional [OpenEnv compatibility check](runs/openenv-compat-2026-10-21/README.md)
ran the existing scripted baseline and no-action control through OpenEnv's
native Harbor trial path in real Docker containers. Both collected the private
sidecar journal and used the separate verifier; OpenEnv preserved rewards
**0.4583** and **-1**, respectively. Both captures correctly returned
`ok=false` because the scripts made no model calls. This validates lifecycle
and reward handling, not LLM token capture, TRL training, or model performance.
The check changed no runtime code and started no model trial.

The optional [Verifiers package](integrations/verifiers/README.md) also loads
prepared copies of these tasks through its existing Harbor adapter. Its local
Docker checks retain the service journal and separate verifier. The original
Harbor builder and runner remain available as documented above.

## Scope

Isolation keeps the private labels out of the agent image and stops ordinary
file tampering from changing grading. The dataset is public and was used in
earlier development, so this remains a research prototype with possible
lookup and memorization shortcuts. No adversarial resistance or frontier-model
difficulty has been established. Using Harbor does not make it a qualified
Terminal-Bench submission.

The implementation follows Harbor's official documentation on
[multi-container tasks](https://docs.harborframework.com/tasks/multi-container),
[service artifact collection](https://docs.harborframework.com/jobs/artifact-collection),
and [separate verifiers](https://docs.harborframework.com/tasks/separate-verifier).
