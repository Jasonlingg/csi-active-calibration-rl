# Local Verifiers integration check — 2026-10-22

The optional `csi-label-selection` package runs the existing CSI Harbor tasks
through Verifiers' native agent, Docker Compose, artifact collection, and
isolated verifier. It targets Verifiers commit
`c4e2f24a9ae0434ab55f5627c9086633b9391ad6` (0.4.1.dev68) and Harbor 0.24.0.
The original Harbor runner, classifier, recordings, and earlier results remain
unchanged. This is a local integration, not an upstream PR or published environment.

See [setup and reproduction](../../integrations/verifiers/README.md) and
[recording provenance](../../provenance/README.md).

## Results

| Episode | Script | Labels requested | Initial accuracy | Final accuracy | Reward |
| --- | --- | ---: | ---: | ---: | ---: |
| Train A, seed 0 | Margin | 16 | 0.125 | 0.25 | 0.125 |
| Train A, seed 0 | No action | 0 | — | — | -1 |
| Eval F, seed 0 | Margin | 16 | 0.125 | 0.5833333333333334 | 0.45833333333333337 |
| Eval F, seed 0 | No action | 0 | — | — | -1 |

All four native traces completed successfully. The no-action control was
correctly graded as an invalid, incomplete episode. Each saved private journal
was independently replayed with the existing grader; the complete report
matched exactly. Traces contain zero model calls and zero output tokens and
are marked `trainable=false` using Verifiers' trace callback.

The core suite passed 57 tests (two existing Gymnasium warnings about unbounded
score observations); the optional package passed 13 tests. Ruff checks and
formatting passed for the new Python files. The wheel built successfully and
loaded all five training and seven evaluation tasks from copied bundles,
outside the repository, without the main CSI Python package installed.
The included model-run configuration passed `vf-eval --dry-run` with `push=false`.

All 12 prepared tasks passed file-hash and partition checks. Across receivers,
the training query/reward data uses 48 distinct recording events; evaluation
uses a separate 48. These sets are disjoint. Multiple receivers or row
permutations do not constitute independent recording events.

## Saved evidence

- `train/` and `eval-F/`: summaries, native traces, collected journal artifacts.
- `prepared/`: indexes and per-task build manifests for the 12 local tasks.
- `verification.json`: versions, checks, independent replay reports, event IDs.
- `wheel-isolation.json`: standalone wheel load results.
- `resolved-eval.json`: dry-run configuration; no model was called.
- `csi_label_selection-0.1.0-py3-none-any.whl`: optional package build.
- `SHA256SUMS.json`: hashes of this record's files.

Raw runs remain in the ignored `verifiers_runs/train-check-v3` and
`verifiers_runs/eval-F-check-v2` directories. Earlier attempts remain there too;
the saved copies here include the corrected non-training trace metadata.
Prepared tasks remain in the ignored `verifiers_tasks/{train,eval}` directories.
No Verifiers containers remained after the checks.

## Scope

No LLM rollout, token-capture validation, fine-tuning, model comparison, hosted
Prime compatibility check, publication, submission, or cloud spending occurred.
The scripted margin result checks infrastructure, not LLM ability or learning.
Grader images exist only in the local Docker daemon; these records and the
wheel do not distribute them. Prepare tasks again on another Docker host.
This is public development data from one person in one office, not a proven
cheat-resistant benchmark.
