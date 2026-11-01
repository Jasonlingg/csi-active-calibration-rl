"""Deterministic reference harness for infrastructure checks, without inference."""

import json
from pathlib import Path
from typing import Literal

import verifiers.v1 as vf


class ScriptedConfig(vf.HarnessConfig):
    strategy: Literal["margin", "nop"] = "margin"


class ScriptedHarness(vf.Harness[ScriptedConfig]):
    async def launch(
        self,
        ctx,
        trace,
        runtime,
        endpoint,
        secret,
        mcp_urls,
        data,
        tool_interception_url=None,
    ):
        if self.config.strategy == "margin":
            solution = Path(data.task_dir) / "solution"
            for name in ("harbor_client.py", "run_baseline.py"):
                await runtime.write(
                    f"/tmp/csi-scripted/{name}", (solution / name).read_bytes()
                )
            result = await runtime.run(
                ["python3", "/tmp/csi-scripted/run_baseline.py"], {}
            )
            if result.exit_code:
                return result
        state = await runtime.run(["csi-calibration", "state"], {})
        if state.exit_code == 0:
            trace.info["scripted_state"] = json.loads(state.stdout)
        trace.info["scripted_check"] = True
        return state
