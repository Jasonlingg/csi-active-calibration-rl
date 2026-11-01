# CSI active-calibration RL

A runnable Gymnasium environment for learning **which Wi-Fi activity samples
to ask a person to label**. A fixed classifier uses those labels; the policy
receives reward for its accuracy improvement on separate recording events.

This standalone local project is an offline research prototype using the
recordings in its own `data/` directory. It can support a capstone about
calibration with limited labels.
It does not collect live CSI or establish transfer to new rooms or people.

## Run it

From this project directory, with Python 3.11 or newer:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[train,test]'

.venv/bin/python \
  -m active_calibration.demo --receiver F --policy uncertainty

.venv/bin/python \
  -m pytest tests -q
```

Torch is optional for environment/baseline use: install the project without
`[train]` if you only need Gymnasium and the fixed learner. The loader depends
on the nine source/calibration files in `data/`; they are not bundled in a
wheel. Editable installation works directly. A wheel installation needs an
explicit `--project-root /path/to/csi-active-calibration-rl` (or
`load_episodes(project_root=...)`). `--repo-root` and the loader's `repo_root`
keyword remain compatibility aliases. Terminal-Bench is not a runtime dependency.

The latest local core check used Python 3.11.9, NumPy 2.4.3, Gymnasium 1.4.0,
Torch 2.10.0, and pytest 9.2.0.
All 60 tests pass, covering Gymnasium compatibility, label separation, reward
arithmetic, global event splits, kernel-update equivalence, and policy masks
and permutation symmetry, plus preparation for Verifiers. Gymnasium emits two generic warnings about the
unbounded classifier-score observation space.

## Run with Harbor

The project also has a working **Harbor 0.24.0 task** for a terminal agent that
chooses labels interactively. Its private service enforces the same 16-query
budget, and a separate verifier replays the accepted requests to calculate
the accuracy-improvement reward. It runs from this standalone repository.

See [HARBOR.md](HARBOR.md) for setup, commands, and evidence. The scripted
reference check reached **12.5% → 58.3% accuracy**, reward **0.4583**, on one
receiver-F episode. This check calls no model. The first LLM attempts through
Verifiers are recorded below. The Torch-policy experiment is a separate way to use the
same environment.

## Run with Verifiers

The optional [Verifiers package](integrations/verifiers/README.md) loads the
existing Harbor task format as the `csi-label-selection` taskset. Preparation
creates five training tasks (A-E) and seven evaluation tasks (A-G), with local
grader images and checked bundle hashes. The private service, separate grading,
16-label budget, and accuracy-gain reward stay the same.

Real Docker checks through Verifiers reproduced the margin baseline rewards
of **0.125** on training A and **0.4583** on evaluation F; both no-action
controls scored **-1**. Another **13 integration tests** pass. These scripted
traces are marked non-training and contain no model calls. LLM token capture
and fine-tuning remain unvalidated. Nothing has been published or submitted.

The first [Qwen3-8B inference check](runs/llm-smoke-2026-10-26/README.md)
used a Runpod A40 with the same local tasks and separate grader. The initial
attempt requested three labels before its context overflowed; a retry with
native context compaction requested one label and then stopped. Both correctly
received **-1** for incomplete episodes, confirmed by independent journal replay.
Model serving and tool calls work; a complete LLM episode and training-ready
token capture have not been demonstrated. The temporary GPU was terminated.

The next local change adds an optional compact terminal observation and
explicit completion instructions. A CPU tokenizer audit reduces a complete
episode's observation text to 20–22k Qwen tokens; the full observation remains
available. Core tests and fresh scripted Harbor checks pass. This is an
interface fix, not a successful LLM rollout or training result. Evidence and
reproduction are in [the interface check](runs/llm-interface-2026-10-28/README.md).
The remaining training milestones are in [LLM_TRAINING.md](LLM_TRAINING.md).

## Episode contract

Each episode has 480 labeled source examples, 24 unlabeled target query
examples, and 24 separate target reward examples. The policy receives 27
summary features and eight classifier scores for each query example, acquired
labels (`-1` until requested), action/queried masks, and remaining budget.
The source data is used by the fixed learner rather than given as another
policy input. Receiver names and recording event IDs are absent from the
observation.

An action is the row index of one available query example. Its recorded
activity label is revealed and the fixed learner updates. There are eight
classes: empty room, jumping, standing, walking, running, squat, clapping,
and waving arms. A repeat or padding action raises `ValueError` without
consuming budget. Mask invalid actions when sampling.

The default budget is 16 queries. Rewards are zero before the last query.
The terminal reward is:

```text
accuracy on reward examples after 16 queries
− accuracy on those same examples before any queries
```

Reward can be negative. The policy must use the complete fixed budget;
there is no stop action or variable label-price objective. Reward features,
labels, predictions, and accuracy are absent from policy observations and
intermediate info. Final info reports initial/final accuracy, reward, and
query count. These statistics are feedback during policy training; the
evaluation runner freezes the trained policy and performs no updates.

```python
import numpy as np
from active_calibration import ActiveCalibrationEnv
from active_calibration.data import load_episodes

env = ActiveCalibrationEnv(load_episodes(split="train"), budget=16)
obs, info = env.reset(seed=0)
rng = np.random.default_rng(0)
while True:
    action = int(rng.choice(np.flatnonzero(obs["action_mask"])))
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        print(reward, info)
        break
```

## Fixed learner and data split

The original feature extractor summarizes overall level, dynamic energy,
motion-envelope variation, 16 Fourier magnitudes, and eight autocorrelation
lags. It ignores subcarrier identity; that invariance alone does not guarantee
receiver independence. The learner is weighted RBF kernel ridge with
`gamma=0.25`, `ridge=0.05`, and target-label weight `8`. It standardizes using
the complete unlabeled query pool, with the scaler fixed during an episode.
A cached Schur complement produces the same scores as full refitting while
solving only a small system after each query.

The underlying [wise-har release](https://doi.org/10.4121/48e1331b-91df-4e1a-906b-b0a0e9d16c74.v3)
contains synchronized recordings across receivers. Their shared physical
event IDs determine splits globally:

| Use | Receivers | Query repetition | Reward repetition |
| --- | --- | --- | --- |
| Policy training | A–E | 20 | 21 |
| Development evaluation | A–G | 23 | 24 |

Source repetitions are 0–19; repetition 22 is unused. Each target repetition
has eight classes × three settings = 24 events. F/G are excluded from policy
training. The train/evaluation physical event sets are disjoint across all
receivers, including the synchronized observations. Fixture hashes and a
manifest hash guard against accidental row/label misalignment.

Only `data/source.npz`, `data/target_calib_A.npz` through
`data/target_calib_G.npz`, and `data/calibration_event_index.csv` are read.
No Terminal-Bench files are used at runtime or by the test suite. Raw-to-fixture
provenance is documented in [the provenance README](provenance/README.md).

## First measured run

One policy was trained for 400 episodes with seed 0, on A–E only, using
terminal-return REINFORCE, a value baseline, and an entropy bonus. Its shared
candidate network has 64 hidden units and pools context over the query set.
It uses queried labels and current scores without learning a row position.
The frozen policy chooses its largest masked logit in evaluation.

All strategies were evaluated on the same seven receiver episodes with five
pool-shuffle seeds (0–4), and the same 16-label budget:

| Strategy | Final accuracy | Gain over source only |
| --- | ---: | ---: |
| Source only | 14.88% | — |
| Random queries | 40.83% | +25.95 percentage points |
| Smallest top-two score margin | 42.26% | +27.38 points |
| Farthest-first feature diversity | 42.86% | +27.98 points |
| First learned RL policy | 33.33% | +18.45 points |

The first learned policy trails all query baselines. These results establish
an executable experiment and a starting comparison; one short training run
does not establish whether RL can outperform these rules. There was no
evaluation-driven hyperparameter search. Raw results, the checkpoint,
configuration, and training history are in
[`runs/2026-10-17-seed0/`](runs/2026-10-17-seed0/).

The evaluation contains **24 unique physical reward events**, each observed
on seven receivers (168 receiver/event pairs). Seed repeats reuse those
recordings; they are not extra independent examples. Deterministic policies
chose the same sets under these row permutations. The JSON's standard
deviations are descriptive spread across receiver/seed episodes, not
confidence intervals.

To produce another explicitly named run without overwriting this one:

```sh
.venv/bin/python \
  -m active_calibration.train --episodes 400 --seed 0 --budget 16 \
  --out /tmp/csi-rl-my-run/train

.venv/bin/python \
  -m active_calibration.evaluate --budget 16 \
  --checkpoint /tmp/csi-rl-my-run/train/policy.pt \
  --out /tmp/csi-rl-my-run/evaluation.json
```

## Limits and capstone use

This finite, public dataset was inspected during earlier author development.
Its evaluation partition is a development check, not an untouched prospective
holdout. There is one participant and one office; receiver variation does not
demonstrate bathroom, room-size, or new-person generalization. A policy could
memorize a small repeated training set. Ordinary Python attributes and public
dataset labels are accessible to an adversarial program; this environment is
not a secure benchmark sandbox.

For a capstone, the environment supplies the calibration experiment: fix an
activity classifier, give every strategy the same labeling budget, and measure
how its query choices change accuracy. A physical capstone would add CSI
acquisition, signal-quality checks, and a usable activity-classification
pipeline. Claims about new rooms or people need recording events from those
conditions and a split fixed before tuning. Existing recordings are sufficient
to build and study this prototype.

A [dev-post draft](DEVPOST_DRAFT.md) describes the implementation and the
initial negative comparison. It has not been published. Original experiments
and submission files were preserved in the earlier
Terminal-Bench checkout. See [MIGRATION.md](MIGRATION.md) for the copy and
verification record. This local Git repository has no remote and has not been
published.
