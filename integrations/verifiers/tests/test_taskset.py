"""Exercise native plugin loading and prevent split/asset mixups before rollouts."""

import hashlib
import json
import subprocess
from pathlib import Path

import pytest
from csi_label_selection import CsiConfig, CsiLabelSelectionTaskset

from active_calibration.verifiers_task import prepare_tasks


@pytest.fixture
def bundles(tmp_path, monkeypatch):
    def build(argv, check):
        context = Path(argv[-1])
        image = (
            "sha256:"
            + hashlib.sha256((context / "episode.npz").read_bytes()).hexdigest()
        )
        Path(argv[argv.index("--iidfile") + 1]).write_text(image + "\n")
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr("active_calibration.verifiers_task.subprocess.run", build)
    return {
        split: prepare_tasks(tmp_path / split, split=split, receivers=["A"])
        for split in ("train", "eval")
    }


def test_native_loader_resolves_taskset_and_separate_grader(bundles):
    from verifiers.v1.tasksets.harbor import HarborTask
    from verifiers.v1.utils.loaders import load_taskset, task_type, taskset_config_type

    assert taskset_config_type("csi-label-selection") is CsiConfig
    assert task_type("csi-label-selection") is HarborTask
    config = CsiConfig(id="csi-label-selection", tasks_dir=bundles["train"])
    task = next(iter(load_taskset(config)))
    assert task.data.id == "csi-train-A-seed0"
    assert task.scoring.mode == "isolated"
    assert task.data.verifier_image.startswith("sha256:")
    assert task.data.timeout.agent == 900 and task.data.timeout.scoring == 120
    assert [(a.source, a.service) for a in task.data.artifacts] == [
        ("/state/query_history.json", "calibration")
    ]
    assert "exactly 16" in task.data.prompt


def test_train_default_rejects_eval_bundle(bundles):
    with pytest.raises(ValueError, match="train/eval split"):
        list(CsiLabelSelectionTaskset(CsiConfig(tasks_dir=bundles["eval"])))
    task = next(
        iter(
            CsiLabelSelectionTaskset(CsiConfig(tasks_dir=bundles["eval"], split="eval"))
        )
    )
    assert task.data.id == "csi-eval-A-seed0"


@pytest.mark.parametrize("target", ["manifest", "instruction", "grader"])
def test_modified_task_assets_are_rejected(bundles, target):
    root = bundles["train"]
    entry = json.loads((root / "index.json").read_text())["tasks"][0]
    task = root / entry["path"]
    name = {
        "manifest": "build_manifest.json",
        "instruction": "instruction.md",
        "grader": "tests/test.sh",
    }[target]
    (task / name).write_text((task / name).read_text() + "\nmodified\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        list(CsiLabelSelectionTaskset(CsiConfig(tasks_dir=root)))


def test_task_path_cannot_escape_bundle(bundles):
    root = bundles["train"]
    path = root / "index.json"
    index = json.loads(path.read_text())
    index["tasks"][0]["path"] = "../eval"
    path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match="escapes"):
        list(CsiLabelSelectionTaskset(CsiConfig(tasks_dir=root)))


def test_duplicate_episode_is_rejected(bundles):
    root = bundles["train"]
    path = root / "index.json"
    index = json.loads(path.read_text())
    index["tasks"] *= 2
    path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match="duplicate CSI episode"):
        list(CsiLabelSelectionTaskset(CsiConfig(tasks_dir=root)))


@pytest.mark.parametrize(
    "field,value",
    [
        ("receiver", "F"),
        ("receiver", "AB"),
        ("budget", 16.0),
        ("seed", True),
    ],
)
def test_invalid_episode_metadata_is_rejected_even_with_updated_index(
    bundles, field, value
):
    root = bundles["train"]
    index_path = root / "index.json"
    index = json.loads(index_path.read_text())
    manifest_path = root / index["tasks"][0]["path"] / "build_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest[field] = value
    manifest_path.write_text(json.dumps(manifest))
    index["tasks"][0]["manifest_sha256"] = hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()
    index_path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match="partition, seed, or label budget"):
        list(CsiLabelSelectionTaskset(CsiConfig(tasks_dir=root)))


def test_identical_prompts_do_not_merge_receiver_identities(bundles):
    root = prepare_tasks(
        bundles["train"].parent / "two-receivers", receivers=["A", "B"]
    )
    tasks = list(CsiLabelSelectionTaskset(CsiConfig(tasks_dir=root)))
    assert tasks[0].data.prompt == tasks[1].data.prompt
    assert [task.data.id for task in tasks] == [
        "csi-train-A-seed0",
        "csi-train-B-seed0",
    ]
    assert [task.data.idx for task in tasks] == [0, 1]


def test_eval_config_keeps_native_taskset_fields(bundles):
    from verifiers.v1.configs.cli.eval import EvalConfig

    config = EvalConfig.model_validate(
        {
            "model": "local-policy",
            "push": False,
            "env": {
                "taskset": {
                    "id": "csi-label-selection",
                    "tasks_dir": str(bundles["eval"]),
                    "split": "eval",
                },
                "agent": {
                    "harness": {"id": "bash"},
                    "runtime": {"type": "docker"},
                    "deployment": {"trust_compose": True},
                },
            },
        }
    )
    restored = EvalConfig.model_validate_json(config.model_dump_json())
    assert isinstance(restored.env.taskset, CsiConfig)
    assert restored.env.taskset.split == "eval" and not restored.push
