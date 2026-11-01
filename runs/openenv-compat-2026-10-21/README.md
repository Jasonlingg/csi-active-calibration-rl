# OpenEnv / Harbor CSI compatibility check

Checked on 2026-10-21 with real local Docker containers and no model calls.
The existing CSI task ran through `openenv.harbor.rollout.run_rollout`, which
delegates to Harbor's native `Trial.create()` and `Trial.run()`.

| Scripted agent | Accepted requests | Separate-verifier reward | OpenEnv reward |
| --- | ---: | ---: | ---: |
| `oracle` (existing margin baseline) | 16 | 0.45833333333333337 | 0.45833333333333337 |
| `nop` | 0 | -1.0 | -1.0 |

Both checks collected `/state/query_history.json` from the private
`calibration` service, used the separate verifier, had no Harbor exception,
and released their capture session. No check containers remain.
The completed baseline reproduced the existing receiver-F result:
3/24 correct initially and 14/24 after labeling.

**Both OpenEnv results have `ok=false`.** This is expected: the scripts make
no model calls, so capture validation reports `[FATAL] no_turns`. A correct
verifier reward alone does not make a usable LLM training trajectory.
The probe asserts this rejection rather than hiding or bypassing it.

The probe adds test-only, in-process seam entries selecting Harbor's existing
`oracle` and `nop` agents. It changes neither framework's source, the task,
nor its grading contract. No capture server or model endpoint runs; the
intercept URL is loopback port 1. This checks lifecycle and reward plumbing,
not a stock LLM harness, hosted OpenEnv transport, exact token/logprob capture,
TRL ingestion, or learning quality. No new upstream bug was established.

## Evidence

- `verification.json`: scope, outcomes, versions, source commits, and checks.
- `oracle/` and `nop/`: raw OpenEnv/native results, verifier reports, artifact
  collection manifests, and accepted-query journals.
- `task-snapshot/`: the frozen evaluation task used for both checks.
- `probe_openenv.py`: the executed reproduction script.
- `dependencies.json`: observed Python and installed dependency versions.
- `sha256.json`: evidence hashes, excluding this README and the hash file itself.

The CSI suite passed **47 tests**. The selected upstream OpenEnv rollout,
session-factory, runner-lifecycle, and capture-validation tests passed
**90 tests**, with one skipped because it imports TRL. These upstream tests
include simulated components; the two CSI checks above used real containers.

Sources were clean at:

- [OpenEnv c696fe24](https://github.com/huggingface/OpenEnv/tree/c696fe24f754d40dfed5f93a0c3c0fbd53d08c6c)
- [Harbor 38b9be01](https://github.com/harbor-framework/harbor/tree/38b9be0164a910015a0be9a37ba6e8e78c562dfe)

## Repeat the no-model checks

From the CSI project root, with Docker running and `uv` installed:

```sh
CSI_CHECK="$(mktemp -d /private/tmp/csi-openenv-repeat.XXXXXX)"
uv venv --python 3.13 "$CSI_CHECK/venv"
uv pip install --python "$CSI_CHECK/venv/bin/python" \
  'openenv @ git+https://github.com/huggingface/OpenEnv.git@c696fe24f754d40dfed5f93a0c3c0fbd53d08c6c' \
  'harbor @ git+https://github.com/harbor-framework/harbor.git@38b9be0164a910015a0be9a37ba6e8e78c562dfe'

"$CSI_CHECK/venv/bin/python" \
  runs/openenv-compat-2026-10-21/probe_openenv.py \
  --agent oracle \
  --task "$PWD/runs/openenv-compat-2026-10-21/task-snapshot" \
  --out "$CSI_CHECK/oracle"

"$CSI_CHECK/venv/bin/python" \
  runs/openenv-compat-2026-10-21/probe_openenv.py \
  --agent nop \
  --task "$PWD/runs/openenv-compat-2026-10-21/task-snapshot" \
  --out "$CSI_CHECK/nop"
```

Each output directory must be new. This adds an optional compatibility check;
the existing Harbor runner and scripted checks remain the normal task workflow.
This uses one existing evaluation episode, not additional independent data.
No model trials, fine-tuning, GPU provisioning, publication, or submission
occurred during this check.
