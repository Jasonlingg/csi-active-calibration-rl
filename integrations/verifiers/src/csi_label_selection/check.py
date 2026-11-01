"""Run CSI's existing scripted reference through the native Verifiers lifecycle."""

import argparse
import asyncio
import json
from pathlib import Path

import verifiers.v1 as vf
from verifiers.v1.configs.client import EvalClientConfig
from verifiers.v1.utils.artifacts import save_artifacts

from .scripted import ScriptedConfig
from .taskset import CsiConfig, CsiLabelSelectionTaskset


async def check(args):
    args.out.mkdir(parents=True, exist_ok=False)
    taskset = CsiLabelSelectionTaskset(
        CsiConfig(tasks_dir=args.tasks_dir, split=args.split)
    )
    if args.task_id is not None:
        taskset = taskset.include(ids=[args.task_id])
    outcomes = []
    for task in taskset.take(args.limit):
        for strategy in ("margin", "nop"):
            config = vf.AgentConfig(
                model="scripted-no-model",
                client=EvalClientConfig(
                    base_url="http://127.0.0.1:1/v1", api_key_var="CSI_NO_MODEL_KEY"
                ),
                harness=ScriptedConfig(id="csi-label-selection", strategy=strategy),
                runtime=vf.DockerConfig(),
                deployment={"trust_compose": True},
                timeout={"setup": 600, "rollout": 120, "finalize": 120, "scoring": 120},
                retries={"max_retries": 0},
                scoring={"retries": {"max_retries": 0}},
            )

            async def capture_report(trace, runtime):
                trace.info["verifier_report"] = json.loads(
                    await runtime.read("/logs/verifier/report.json")
                )

            def mark_scripted(trace):
                trace.agent.trainable = False

            async with vf.Agent(config) as agent:
                trace = await agent.run(
                    task,
                    collect_artifacts=True,
                    on_score=capture_report,
                    on_trace=mark_scripted,
                )
            directory = args.out / f"{task.data.id}-{strategy}"
            directory.mkdir()
            (directory / "trace.json").write_text(
                json.dumps(trace.to_record(), indent=2) + "\n"
            )
            if not trace.ok:
                raise RuntimeError(f"{directory}: {trace.errors}")
            report = trace.info["verifier_report"]
            expected_queries = 16 if strategy == "margin" else 0
            if (
                trace.calls
                or trace.num_output_tokens
                or trace.agent.trainable
                or report["queries"] != expected_queries
                or report["valid"] != (strategy == "margin")
                or trace.reward != report["reward"]
            ):
                raise AssertionError("scripted rollout, journal, and grader disagree")
            save_artifacts(
                directory / "artifacts",
                trace.state.artifacts,
                task.data.artifacts,
                workdir="/app",
            )
            outcomes.append(
                {
                    "task": task.data.id,
                    "strategy": strategy,
                    "reward": trace.reward,
                    "queries": report["queries"],
                    "model_calls": len(trace.calls),
                    "trainable": trace.agent.trainable,
                    "ok": trace.ok,
                }
            )
            print(json.dumps(outcomes[-1]), flush=True)
    if not outcomes:
        raise ValueError("no tasks matched the requested check")
    (args.out / "summary.json").write_text(
        json.dumps(
            {
                "checks": outcomes,
                "scope": "Scripted lifecycle checks; LLM capture and training are unvalidated",
                "published": False,
            },
            indent=2,
        )
        + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "eval"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument(
        "--task-id", help="check one prepared episode by its native task ID"
    )
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
    asyncio.run(check(args))


if __name__ == "__main__":
    main()
