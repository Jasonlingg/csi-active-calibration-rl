"""Preparation keeps the Harbor contract and split boundaries intact."""

import hashlib
import json
import subprocess

import pytest

from active_calibration.harbor_grader import load_bundle
from active_calibration.verifiers_task import prepare_tasks


@pytest.fixture
def local_image_builder(monkeypatch):
    calls = []

    def build(argv, check):
        assert argv[:3] == ["docker", "build", "--quiet"] and check
        from pathlib import Path

        context = Path(argv[-1])
        image = (
            "sha256:"
            + hashlib.sha256((context / "episode.npz").read_bytes()).hexdigest()
        )
        Path(argv[argv.index("--iidfile") + 1]).write_text(image + "\n")
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr("active_calibration.verifiers_task.subprocess.run", build)
    return calls


def test_prepared_splits_are_disjoint_and_manifests_match(
    tmp_path, local_image_builder
):
    events = {}
    for split in ("train", "eval"):
        root = prepare_tasks(tmp_path / split, split=split, receivers=["A"], seed=13)
        index = json.loads((root / "index.json").read_text())
        assert index["split"] == split and len(index["tasks"]) == 1
        task = root / index["tasks"][0]["path"]
        manifest_path = task / "build_manifest.json"
        assert (
            hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            == index["tasks"][0]["manifest_sha256"]
        )
        manifest = json.loads(manifest_path.read_text())
        for name, digest in manifest["sha256"].items():
            assert hashlib.sha256((task / name).read_bytes()).hexdigest() == digest
        episode, seed, budget, digest = load_bundle(task / "tests/episode.npz")
        assert episode.split == split and seed == 13 and budget == 16
        events[split] = set(episode.pool_event_ids) | set(episode.eval_event_ids)
        assert 'environment_mode = "separate"' in (task / "task.toml").read_text()
        assert (
            manifest["verifiers"]["verifier_image"] in (task / "task.toml").read_text()
        )
        assert not (task / "environment/episode.npz").exists()
        assert "10.4121/48e1331b" in (root / "PROVENANCE.md").read_text()
    assert events["train"].isdisjoint(events["eval"])
    assert len(local_image_builder) == 2


@pytest.mark.parametrize(
    "kwargs",
    [
        {"split": "test"},
        {"split": "train", "receivers": ["F"]},
        {"receivers": []},
        {"receivers": ["A", "A"]},
        {"seed": True},
        {"seed": -1},
        {"seed": 0.5},
    ],
)
def test_bad_preparation_does_not_build_images_or_create_output(
    tmp_path, monkeypatch, kwargs
):
    def unexpected(*args, **kw):
        pytest.fail("invalid configuration reached Docker")

    monkeypatch.setattr("active_calibration.verifiers_task.subprocess.run", unexpected)
    output = tmp_path / "new-parent" / "bundle"
    with pytest.raises(ValueError):
        prepare_tasks(output, **kwargs)
    assert not output.parent.exists()


def test_preparation_preserves_existing_output(tmp_path, monkeypatch):
    output = tmp_path / "existing"
    output.mkdir()
    (output / "keep").write_text("existing result")
    with pytest.raises(FileExistsError):
        prepare_tasks(output)
    assert (output / "keep").read_text() == "existing result"


def test_failed_build_does_not_leave_a_partial_bundle(tmp_path, monkeypatch):
    def fail(argv, check):
        raise subprocess.CalledProcessError(1, argv)

    monkeypatch.setattr("active_calibration.verifiers_task.subprocess.run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        prepare_tasks(tmp_path / "failed", receivers=["A"])
    assert list(tmp_path.iterdir()) == []
