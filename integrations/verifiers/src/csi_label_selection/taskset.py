"""Load prepared CSI tasks through Verifiers' existing Harbor adapter."""

import hashlib
import json
from pathlib import Path
from typing import Literal

import verifiers.v1 as vf
from verifiers.v1.tasksets.harbor import HarborConfig, HarborTask
from verifiers.v1.tasksets.harbor.taskset import HarborTaskConfig, parse_task


class CsiConfig(vf.TasksetConfig):
    task: HarborTaskConfig = HarborTaskConfig()
    tasks_dir: Path = Path("verifiers_tasks/train")
    split: Literal["train", "eval"] = "train"


def checked_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError(f"bundle path escapes its root: {relative!r}")
    return path


class CsiLabelSelectionTaskset(vf.Taskset[HarborTask, CsiConfig]):
    def load(self):
        root = self.config.tasks_dir.resolve()
        index = json.loads((root / "index.json").read_text())
        if index["schema_version"] != 1 or index["split"] != self.config.split:
            raise ValueError(
                "prepared bundle must match the requested train/eval split"
            )
        if not index["tasks"]:
            raise ValueError("prepared bundle contains no tasks")
        config = HarborConfig(ignore_timeouts=False, task=self.config.task)
        seen = set()
        for idx, entry in enumerate(index["tasks"]):
            task_dir = checked_path(root, entry["path"])
            manifest_path = task_dir / "build_manifest.json"
            if (
                hashlib.sha256(manifest_path.read_bytes()).hexdigest()
                != entry["manifest_sha256"]
            ):
                raise ValueError("task manifest hash mismatch")
            manifest = json.loads(manifest_path.read_text())
            allowed = tuple("ABCDE" if self.config.split == "train" else "ABCDEFG")
            if (
                manifest["split"] != self.config.split
                or manifest["receiver"] not in allowed
                or type(manifest["budget"]) is not int
                or manifest["budget"] != 16
                or type(manifest["seed"]) is not int
                or manifest["seed"] < 0
            ):
                raise ValueError("invalid CSI partition, seed, or label budget")
            for name, digest in manifest["sha256"].items():
                if (
                    hashlib.sha256(
                        checked_path(task_dir, name).read_bytes()
                    ).hexdigest()
                    != digest
                ):
                    raise ValueError(f"task file hash mismatch: {name}")
            data = parse_task(task_dir, idx, config)
            if data.verifier is None or data.scoring.mode != "isolated":
                raise ValueError("CSI requires a separate verifier")
            if data.verifier_image != manifest["verifiers"]["verifier_image"]:
                raise ValueError("grader image does not match the prepared task")
            identity = (
                f"csi-{manifest['split']}-{manifest['receiver']}-seed{manifest['seed']}"
            )
            if identity in seen:
                raise ValueError(f"duplicate CSI episode: {identity}")
            seen.add(identity)
            yield HarborTask(
                data.model_copy(update={"id": identity, "name": identity}),
                self.config.task,
            )
