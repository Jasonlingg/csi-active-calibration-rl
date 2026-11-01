# Compact LLM interface check — 2026-10-28

Local CPU and scripted Docker checks only. No model inference, GPU resource,
fine-tuning, publication, or submission was started. Earlier runs and data
were preserved.

## Change

The standard-library Harbor client has an optional `--compact` display.
It returns stable row indices, available rows' eight classifier scores,
acquired labels, and integer query counts. `--features` includes static
features, normally requested once at the start. Display numbers are rounded
to four decimal places. The raw CLI, Python client functions, HTTP service,
classifier precision, reward, and separate grading contract are unchanged.

New task instructions describe the compact commands and explicitly require
16 accepted actions before the model finishes. Each query response includes
the updated observation, so another state call is unnecessary. Existing
generated tasks keep their original client and instructions; use a new build
directory rather than editing a historical bundle.

## Checks

- **60 core tests pass**, including three new checks for the compact client,
  stable indices, revealed labels, padding, budget, error handling, and grader
  agreement. The CLI is executed without site packages using `python -I -S`.
  Gymnasium emits its two existing generic unbounded-score warnings.
- **13 optional Verifiers integration tests pass.**
- Fresh **Harbor 0.24.0 Docker** reference: 16 queries, accuracy
  0.125 → 0.5833333333333334, reward **0.45833333333333337**.
- Fresh no-action control: 0 queries, invalid, reward **−1**.
- Both collected journals independently replay to the exact verifier report.
- Fresh training-A preparation and both smoke-config dry runs passed. No model
  endpoint was contacted by a dry run.

The Docker reference uses the existing full-precision margin script. The
compact CLI itself is exercised by the HTTP/subprocess tests. These checks
do not establish that a model follows the new instructions.

## Observation-size audit

`observation-sizes.json` compares identical full-precision margin action
sequences over all five training and seven evaluation episodes, seed 0.
It renders the initial state plus 16 query responses, with features included
only in the first compact response.

Token counts use Qwen3-8B's tokenizer from revision
`b968826d9c46dd6066d109eabc6255188de91218` and `tokenizers==0.21.4`.
The tokenizer file's SHA-256 is in the report. Only that tokenizer was
downloaded; model weights were not downloaded.

| Display | Observation text tokens per complete episode |
| --- | ---: |
| Original indented full JSON | 316,915–318,570 |
| Minified full JSON | 281,758–283,413 |
| Compact, initial features once | 19,628–22,044 |

This is approximately a **93–94% reduction** from the original display.
Prompts, tool schemas, chat templates, and model responses are excluded.
The largest compact transcript leaves roughly 10.7k tokens within a 32,768
context for that additional material; an actual model run must verify fit.

Repeatedly sending the growing compact observation history costs
227,267–259,322 cumulative text-input tokens. The prepared smoke config
therefore allows 512k cumulative inputs and disables automatic compaction;
that limit is different from the model's context window. These settings are
unvalidated by inference. The counts are formatting diagnostics, not captured
engine token IDs or trainable rollouts.

Reproduce without a model, from the project root after installing the project
and `tokenizers==0.21.4` in your chosen environment:

```sh
python experiments/measure_llm_observations.py \
  --tokenizer /path/to/the/pinned/tokenizer.json \
  --out /absolute/path/to/a/new/observation-sizes.json
```

Omit `--tokenizer` to measure bytes without the optional dependency.

## Saved evidence and next step

`verification.json`, `harbor-reference/`, and `harbor-nop/` retain reports,
journals, and native job results. `episode.npz` and `build_manifest.json`
freeze the evaluation-F task used for the Docker checks.

`next-smoke.toml` and `next-smoke-plan.json` prepare, but do not execute, one
Qwen3-8B training-A inference check using a fresh compact task. The plan
proposes at most $1 of new GPU spend and 20 minutes of GPU wall time, with
temporary-resource cleanup. It requires new authorization under the original
no-model-trials instruction. It is not an RL training configuration.

Full LLM completion, exact training capture, optimizer updates, checkpoint
recovery, and policy comparisons remain outstanding. The intended upstream
direction and gates are in [LLM_TRAINING.md](../../LLM_TRAINING.md).

The recordings retain the attribution and limits in
[provenance/README.md](../../provenance/README.md). These are public development
events from one person and one office. No new generalization claim follows
from smaller observations or scripted completion.
