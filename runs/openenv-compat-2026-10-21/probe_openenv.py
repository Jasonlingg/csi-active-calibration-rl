"""Exercise OpenEnv's real Harbor lifecycle with CSI's existing scripted agents.

No inference server runs. The empty model capture is intentionally rejected;
this probe checks task execution, artifact transfer and reward preservation.
"""

import argparse
import asyncio
import hashlib
import json
from pathlib import Path

from openenv.core.harness.capture.sessions import SessionRegistry
from openenv.harbor import seams
from openenv.harbor.rollout import run_rollout


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", choices=("oracle", "nop"), required=True)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)

    # Test-only seam configuration selects Harbor's unmodified built-in agents.
    # OpenEnv has no built-in seam for agents that never invoke a model.
    seams.SEAMS[args.agent] = seams.Seam(
        name=args.agent, dialect="scripted-no-inference", model_fmt=""
    )
    registry = SessionRegistry()
    sessions = []
    result = await run_rollout(
        task_dir=args.task.resolve(),
        harness=args.agent,
        sandbox="docker",
        registry=registry,
        intercept_url="http://127.0.0.1:1",
        model="scripted-no-model",
        trials_dir=args.out.resolve() / "trials",
        agent_timeout_sec=120,
        capture_level="text",
        purpose="eval",
        session_prefix="csi-openenv-check",
        on_session_created=lambda sid: sessions.append(registry.get(sid)),
    )
    (args.out / "openenv-result.json").write_text(result.model_dump_json(indent=2) + "\n")
    trial = args.out / "trials" / result.trial_name
    native = json.loads((trial / "result.json").read_text())
    report = json.loads((trial / "verifier/report.json").read_text())
    manifest = json.loads((trial / "artifacts/manifest.json").read_text())
    journal_path = trial / "artifacts/state/query_history.json"
    journal = json.loads(journal_path.read_text())
    artifact = next(entry for entry in manifest if entry["service"] == "calibration")
    assert artifact["source"] == "/state/query_history.json"
    assert artifact["status"] == "ok"
    assert native["exception_info"] is None, native["exception_info"]
    assert result.reward == report["reward"] == native["verifier_result"]["rewards"]["reward"]
    assert report["valid"] == (args.agent == "oracle")
    assert len(journal["actions"]) == (16 if args.agent == "oracle" else 0)
    assert result.reward == (0.45833333333333337 if args.agent == "oracle" else -1.0)
    assert len(sessions) == 1 and sessions[0].model_calls == 0
    assert registry.list_ids() == []
    assert result.n_turns == result.n_trainable_tokens == 0
    assert not result.ok and "no model calls" in result.error.lower(), result.error
    summary = {
        "agent": args.agent,
        "task": str(args.task.resolve()),
        "trial": str(trial.resolve()),
        "report": report,
        "reward_preserved": True,
        "private_sidecar_artifact_collected": True,
        "journal_sha256": hashlib.sha256(journal_path.read_bytes()).hexdigest(),
        "queries": len(journal["actions"]),
        "model_calls": sessions[0].model_calls,
        "session_cleaned_up": True,
        "openenv_ok": result.ok,
        "expected_capture_rejection": result.error,
        "scope": "Harbor lifecycle and reward check; no LLM capture or training validated",
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
