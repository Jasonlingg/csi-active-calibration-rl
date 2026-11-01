# CSI label selection for Verifiers

An optional Verifiers v1 taskset over this project's existing Harbor tasks.
The agent selects 16 labels; the fixed classifier updates and a separate grader
replays the private service journal to calculate accuracy gain. Incomplete
episodes receive -1. The Harbor runner remains available independently.

This package targets Verifiers commit
`c4e2f24a9ae0434ab55f5627c9086633b9391ad6` and Harbor 0.24.0.
It is local and unpublished. Initial LLM rollouts have run, but none completed
the 16-label task; fine-tuning has not run.
See the project [README](../../README.md) and
[recording provenance](../../provenance/README.md) for dataset and study limits.

## Install and prepare

From the CSI project root, with Docker Compose and Python 3.13 available:

```sh
uv venv --python 3.13 /tmp/csi-verifiers-env
uv pip install --python /tmp/csi-verifiers-env/bin/python \
  -e . -e integrations/verifiers

/tmp/csi-verifiers-env/bin/python -m active_calibration.verifiers_task \
  --split train --out "$PWD/verifiers_tasks/train"
/tmp/csi-verifiers-env/bin/python -m active_calibration.verifiers_task \
  --split eval --out "$PWD/verifiers_tasks/eval"
```

Preparation freezes five training tasks (A-E) or seven evaluation tasks (A-G),
using the existing globally disjoint recording-event partitions and seed 0.
For a smaller check, pass `--receivers A` for train or `--receivers F` for eval.
Each output directory must be new. Preparation builds each grader image into
the local Docker daemon, records its immutable image ID, and never pushes it.
Prepared directories therefore require their grader images on that Docker host;
moving to another host requires preparing there again. Copying the Python
package alone does not include the recordings or Docker images.

The taskset checks bundle hashes and requires the requested split to match the
prepared split. Training rejects F/G. Receiver/seed identities remain distinct
even though the agent instructions are identical. Row permutations reuse the
recordings and are not independent training data.

## Check without a model

```sh
/tmp/csi-verifiers-env/bin/csi-label-check \
  --tasks-dir "$PWD/verifiers_tasks/eval" --split eval \
  --out /tmp/csi-verifiers-check
```

This runs the existing margin reference and a no-action control through
Verifiers' native agent, Compose, artifact, and isolated-grading lifecycle.
The scripted harness makes no inference calls and produces no training tokens.
Outputs include native traces marked `trainable=false`, grader reports, private
journals, and rewards. Use a new output directory for each run. `--limit`
selects how many tasks to check; `--task-id csi-eval-F-seed0` selects receiver F.

Validate the included model-run configuration without calling a model:

```sh
/tmp/csi-verifiers-env/bin/vf-eval @ integrations/verifiers/eval.toml \
  --dry-run --output-dir /tmp/csi-verifiers-config-check
```

The file sets `push=false` and points to a local model-server placeholder.
Omitting `--dry-run` would attempt model calls and requires a running server.

## Checked locally

The core suite passes 60 tests, and the optional package passes 13 integration
tests covering native plugin discovery, config round trips, split enforcement,
asset hashes, and episode identity. Four real Docker rollouts passed:

| Split / receiver | Script | Queries | Reward |
| --- | --- | ---: | ---: |
| Train A | Margin baseline | 16 | 0.125 |
| Train A | No action | 0 | -1 |
| Eval F | Margin baseline | 16 | 0.45833333333333337 |
| Eval F | No action | 0 | -1 |

The saved service journals replay to exactly the same grader reports. The wheel
loads all five training and seven evaluation tasks without the main CSI Python
package installed. These checks validate local execution and packaging;
they do not establish model performance, learning, or hosted Prime compatibility.
Evidence is in [the integration record](../../runs/verifiers-integration-2026-10-22/README.md).

To run the package's tests after installing the main CSI project in the same
environment:

```sh
uv pip install --python /tmp/csi-verifiers-env/bin/python \
  -e 'integrations/verifiers[test]'
/tmp/csi-verifiers-env/bin/python -m pytest integrations/verifiers/tests -q
```

## First model check

The [2026-10-26 Qwen3-8B check](../../runs/llm-smoke-2026-10-26/README.md)
served a pinned model revision with vLLM on one temporary Runpod A40 and ran
the native `vf-eval` CLI locally. A tool-calling preflight passed. On training
receiver A, the initial attempt exhausted its 32k context after three queries;
a retry using native compaction stopped voluntarily after one query. Both
received -1, with the separate grader and independent journal replay agreeing.
No successful completion or learning improvement is claimed. The temporary
pod was deleted, and the three existing pods remain stopped.

The native trace's `ok=true` indicates lifecycle completion, not task validity:
inspect the grader report and the accepted-query count too. These API traces
contain messages and usage but no sampled token IDs, so exact training capture
remains unvalidated. Repeated full JSON observations are expensive in context;
the first attempt's compaction-disabled setup should not be treated as a
working model configuration. Both attempts and their exact configs are saved.

## Model-run configuration

For the next LLM episode, `smoke.toml` selects training receiver A, seed 0,
with one rollout, concurrency one, 24 model turns, 2,048 tokens per response,
an 8,192 output-token stopping threshold, and a ten-minute rollout timeout.
It uses compact-observation instructions from a newly prepared bundle,
disables automatic compaction, and permits 512k cumulative input tokens
(not a 512k context window). The CPU audit measured 227–259k cumulative
observation-text inputs when repeatedly sending the growing transcript.
Prompts and model outputs add to that. These settings have passed a dry run;
they have not been validated by a new LLM rollout.
Retries and publishing are disabled. These are execution limits, not a dollar
budget; cost depends on the chosen model endpoint and its billing.

Prepare the fresh training-A bundle and validate without calling a model:

```sh
/tmp/csi-verifiers-env/bin/python -m active_calibration.verifiers_task \
  --split train --receivers A --out "$PWD/verifiers_tasks/compact/train"

/tmp/csi-verifiers-env/bin/vf-eval @ integrations/verifiers/smoke.toml \
  --dry-run --output-dir /tmp/csi-verifiers-smoke-config
```

Use a new bundle directory if that path already exists, and change `tasks_dir`
in a copy of the config to match. Old bundles keep their original instructions
and client; installing updated Python code does not update them.

The model name and localhost endpoint are placeholders. Before an actual run,
supply a working OpenAI-compatible model endpoint, its model ID, and credentials
through the configured environment variable. A successful first episode should
contain real model calls and tool use, exactly 16 accepted queries, and an
independently replayable verifier reward. Ordinary API traces alone do not
establish exact training-token capture or trainer compatibility.

For a future LLM run, use the regular `bash` harness with this taskset ID:

```toml
[env.taskset]
id = "csi-label-selection"
tasks_dir = "/absolute/path/to/verifiers_tasks/train"
split = "train"

[env.agent.harness]
id = "bash"

[env.agent.runtime]
type = "docker"

[env.agent.deployment]
trust_compose = true
```

This is an environment configuration fragment, not a full training recipe.
Model serving, exact token capture, trainer configuration, and learning quality
still need validation. Use explicit evaluation bundles for frozen comparisons;
the release is public development data from one person in one office.
