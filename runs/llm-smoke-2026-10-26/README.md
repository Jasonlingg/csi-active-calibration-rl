# First CSI LLM inference check — 2026-10-26

Qwen3-8B made real tool calls against the existing CSI task through Verifiers'
native `vf-eval` CLI, Docker Compose, and separate verifier. Neither attempt
completed the required 16 label requests. No fine-tuning, publishing, or
submission occurred. The Harbor runner, classifier, reward, recording splits,
and earlier results were unchanged.

## Results

Both attempts used training receiver A, seed 0, the same frozen episode,
greedy decoding, and Qwen3-8B in non-thinking mode.

| Attempt | Accepted queries | Reward | Outcome |
| --- | ---: | ---: | --- |
| Initial, compaction disabled | 3 | -1 | Next request had 43,654 input tokens, exceeding the 32,768-token server context. |
| Retry, native compaction at 16k | 1 | -1 | Model summarized context, echoed an intention to continue, then returned text and stopped. |

The initial attempt made five successful model calls plus one rejected
request, producing 140 output tokens. The retry made six successful model
calls, including its summary, producing 716 output tokens. A separate
tool-calling preflight generated 19 tokens and returned the requested valid
`ping` function call.

Each accepted-query journal was independently replayed using the existing
CPU grader. The complete reports matched the separate verifier exactly.
The native traces say `ok=true` and `agent_completed`; those flags describe
lifecycle completion, not successful task completion. The authoritative
reports say `valid=false` and reward -1.

The trace nodes contain no sampled token IDs. Messages, tool calls, usage,
and journals are available, but training-ready capture is not validated.
These results establish neither an effective label-selection strategy nor
learning improvement. The next useful work is to make observation handling
and completion behavior reliable before adding an RL trainer.

## Setup and resource use

- Model: `Qwen/Qwen3-8B`, revision
  `b968826d9c46dd6066d109eabc6255188de91218`, BF16.
- Server: vLLM 0.10.2, Transformers 4.55.2, Torch 2.8.0, Python 3.12.3.
- Hardware: one NVIDIA A40, 46,068 MiB reported by `nvidia-smi`.
  The server reserved 36,597 MiB, including its configured KV cache.
- Serving: 32,768-token context, one sequence, eager mode, Hermes tool parser.
  `nonthinking.jinja` is the model's own chat template with
  `enable_thinking=false` set explicitly. The server bound only to loopback
  and was reached from the Mac through an SSH tunnel.
- Runtime: Verifiers commit
  `c4e2f24a9ae0434ab55f5627c9086633b9391ad6`, Harbor 0.24.0.
- Both attempts had one rollout, concurrency one, a 600-second rollout limit,
  2,048 tokens per response, an 8,192 output-token threshold, no rollout/scoring
  retries, and `push=false`. The retry allowed 40 turns and a 512k cumulative
  input-token threshold to accommodate native summaries; the initial limits
  were 24 turns and 128k input tokens. Neither hit those aggregate limits.

The three existing A40 pods were unavailable: two had no free GPU on their
hosts, and the third failed container creation with `GPU requires reset`.
The third was returned to its original stopped state. A temporary pod,
`luleoifj03tqto`, was created in EU-SE-1 at the catalog rate of $0.59/hour.
It ran for approximately eight minutes and was **terminated**, verified by
a subsequent 404 and account inventory. All three original pods remain
`EXITED`; no Verifiers containers or SSH tunnel remain.

Temporary-pod compute is approximately $0.08 from elapsed time times the
quoted rate. Including the brief failed existing-pod resume, compute is
roughly $0.08–$0.10, excluding existing storage. The billing API had no
records for either pod yet; these are estimates, not a finalized bill.
The automatic cleanup timer was cancelled after deletion was verified.

## Files and reproduction

- `run.toml` / `run-compaction.toml`: exact experiment configurations.
- `native/` / `native-compaction/`: native configs, traces, logs, and journals.
- `verification.json`: outcomes, capture audit, replay checks, and cleanup.
- `initial-replay.json` / `native-compaction-replay.json`: independent reports.
- `episode.npz` / `build_manifest.json`: frozen episode and preparation record.
- `server-command.json`, `server-requirements.txt`, `server-environment.json`,
  `server.log`, and `nonthinking.jinja`: model-serving evidence.
- `model.json`, `server-models.json`, `tool-preflight.json`: model identity and
  successful tool preflight.
- `SHA256SUMS.json`: hashes of this run's files.

The configs require an installed optional package, the prepared training
bundle, Docker, and a live compatible model server behind local port 18000.
The temporary server has been deleted; the old pod address is not reusable.
From the project root, after setting up a new server and SSH tunnel:

```sh
vf-eval @ runs/llm-smoke-2026-10-26/run-compaction.toml \
  --output-dir /absolute/path/to/a/new/output-directory --rich None
```

To reproduce only the CPU grading, use the saved `episode.npz` and a journal
path listed in `verification.json` with
`python -m active_calibration.harbor_grader --bundle ... --history ... --out-dir ...`.
This calls no model. See [PROVENANCE.md](PROVENANCE.md) for attribution and
dataset limits. The observations come from public development recordings,
not a prospective holdout.
