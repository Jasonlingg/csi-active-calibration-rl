"""Exercise the private-service boundary and replay grading without Docker."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pytest

from active_calibration.env import ActiveCalibrationEnv
from active_calibration.harbor_grader import grade, load_bundle
from active_calibration.harbor_service import CalibrationService, ServiceError, create_server


@pytest.fixture
def harbor_bundle(episode, tmp_path):
    """A small episode whose queried labels change accuracy from zero to one."""
    frozen = replace(
        episode,
        source_f=np.full((8, 5), -100.0),
        source_y=np.zeros(8, dtype=np.int64),
        pool_f=np.repeat(np.arange(18)[:, None], 5, axis=1).astype(float),
        pool_y=np.ones(18, dtype=np.int64),
        eval_f=np.zeros((7, 5)),
        eval_y=np.ones(7, dtype=np.int64),
        receiver="F",
        split="eval",
        pool_event_ids=tuple(range(100, 118)),
        eval_event_ids=tuple(range(200, 207)),
    )
    path = tmp_path / "episode.npz"
    values = {
        name: getattr(frozen, name)
        for name in (
            "source_f", "source_y", "pool_f", "pool_y", "eval_f", "eval_y",
            "pool_event_ids", "eval_event_ids", "receiver", "split",
        )
    }
    np.savez_compressed(path, **values, seed=np.int64(13), budget=np.int64(16))
    return {"path": path, "episode": frozen, "seed": 13, "budget": 16}


def journal_for(bundle, actions=None):
    return {
        "schema_version": 1,
        "bundle_sha256": hashlib.sha256(bundle["path"].read_bytes()).hexdigest(),
        "seed": bundle["seed"],
        "budget": bundle["budget"],
        "actions": list(range(16)) if actions is None else list(actions),
    }


def request_json(url, path, body=None):
    raw = None if body is None else json.dumps(body).encode("utf-8")
    request = Request(
        url + path, data=raw,
        headers={} if raw is None else {"Content-Type": "application/json"},
    )
    try:
        with urlopen(request, timeout=5) as response:
            return response.status, json.load(response)
    except HTTPError as error:
        return error.code, json.loads(error.read())


@pytest.fixture
def calibration_http(harbor_bundle, tmp_path):
    history = tmp_path / "state" / "query_history.json"
    server = create_server(harbor_bundle["path"], history, host="127.0.0.1", port=0)
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
    )
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", history, server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()


def test_replay_grader_returns_the_exact_environment_terminal_reward(harbor_bundle, tmp_path):
    bundle = harbor_bundle
    actions = [7, 0, 4, 2, 17, 1, 16, 3, 5, 6, 8, 9, 10, 11, 12, 13]
    history = tmp_path / "query_history.json"
    history.write_text(json.dumps(journal_for(bundle, actions)))
    env = ActiveCalibrationEnv([bundle["episode"]], budget=16)
    env.reset(seed=bundle["seed"], options={"episode_index": 0})
    for number, action in enumerate(actions, start=1):
        _, reward, terminated, truncated, info = env.step(action)
        assert terminated == (number == 16)
        assert not truncated
        if not terminated:
            assert reward == 0.0
    result = grade(bundle["path"], history)
    assert result["valid"] is True
    assert result["queries"] == 16
    assert result["reward"] == reward == 1.0
    assert result["initial_accuracy"] == info["initial_accuracy"] == 0.0
    assert result["final_accuracy"] == info["final_accuracy"] == 1.0
    assert result["accuracy_gain"] == result["reward"]


@pytest.mark.parametrize("attack", [
    "missing", "incomplete", "repeated", "out_of_range", "boolean", "float",
    "claimed_score", "wrong_bundle", "wrong_seed", "duplicate_json_field",
])
def test_invalid_journals_fail_closed_without_accepting_claimed_rewards(harbor_bundle, tmp_path, attack):
    history_path = tmp_path / "query_history.json"
    history = journal_for(harbor_bundle)
    if attack == "incomplete":
        history["actions"].pop()
    elif attack == "repeated":
        history["actions"][-1] = history["actions"][0]
    elif attack == "out_of_range":
        history["actions"][0] = 18
    elif attack == "boolean":
        history["actions"][0] = True
    elif attack == "float":
        history["actions"][0] = 0.0
    elif attack == "claimed_score":
        history["claimed_score"] = 999.0
    elif attack == "wrong_bundle":
        history["bundle_sha256"] = "0" * 64
    elif attack == "wrong_seed":
        history["seed"] += 1
    if attack != "missing":
        raw = json.dumps(history)
        if attack == "duplicate_json_field":
            raw = raw.replace('"schema_version": 1', '"schema_version": 1, "schema_version": 1', 1)
        history_path.write_text(raw)
    result = grade(harbor_bundle["path"], history_path)
    assert result["valid"] is False
    assert result["reward"] == -1.0
    assert result["reason"]


def test_http_service_hides_answers_locks_budget_and_journals_only_accepted_queries(
    harbor_bundle, calibration_http
):
    url, history_path, _ = calibration_http
    status, state = request_json(url, "/state")
    assert status == 200
    assert set(state) == {"observation", "queries_used", "budget", "terminated"}
    assert state["budget"] == 16 and state["queries_used"] == 0
    assert not state["terminated"]
    assert set(state["observation"]) == {
        "features", "scores", "labels", "action_mask", "queried_mask", "budget"
    }
    assert state["observation"]["labels"] == [-1] * 18
    assert journal_for(harbor_bundle, []) == json.loads(history_path.read_text())
    unchanged = history_path.read_bytes()
    assert request_json(url, "/state")[1] == state
    for path, body, expected_status in [
        ("/reset", {}, 404),
        ("/query", {"index": 0, "budget": 999}, 400),
        ("/query", {"index": True}, 400),
        ("/query", {"index": 18}, 409),
    ]:
        assert request_json(url, path, body)[0] == expected_status
        assert request_json(url, "/state")[1] == state
        assert history_path.read_bytes() == unchanged
    assert request_json(url, "/episode.npz")[0] == 404
    assert request_json(url, "/query_history.json")[0] == 404

    for number, action in enumerate(range(16), start=1):
        status, result = request_json(url, "/query", {"index": action})
        assert status == 200 and result["queries_used"] == number
        assert result["acquired_label"] == result["observation"]["labels"][action] == 1
        assert result["terminated"] == (number == 16)
        assert json.loads(history_path.read_text()) == journal_for(harbor_bundle, range(number))
        if number < 16:
            assert result["info"] == {} and result["reward"] == 0.0
        if number == 1:
            before_invalid = history_path.read_bytes()
            before_state = request_json(url, "/state")[1]
            assert request_json(url, "/query", {"index": action})[0] == 409
            assert history_path.read_bytes() == before_invalid
            assert request_json(url, "/state")[1] == before_state
    assert result["reward"] == 1.0
    assert result["observation"]["labels"].count(-1) == 2
    assert not any(result["observation"]["action_mask"])
    final_history = history_path.read_bytes()
    assert request_json(url, "/query", {"index": 16})[0] == 409
    assert history_path.read_bytes() == final_history
    assert grade(harbor_bundle["path"], history_path)["reward"] == result["reward"]


def test_concurrent_duplicate_http_queries_consume_one_label(calibration_http):
    url, history_path, _ = calibration_http
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: request_json(url, "/query", {"index": 3}), range(2)))
    assert sorted(status for status, _ in results) == [200, 409]
    assert json.loads(history_path.read_text())["actions"] == [3]
    state = request_json(url, "/state")[1]
    assert state["queries_used"] == 1
    assert sum(label != -1 for label in state["observation"]["labels"]) == 1


def test_compact_cli_completes_the_same_private_journal_and_reward(
    harbor_bundle, calibration_http, tmp_path
):
    """Exercise the client shipped to agents, isolated from project imports."""
    from active_calibration import harbor_client

    url, history_path, server = calibration_http
    client = tmp_path / "csi-calibration"
    shutil.copyfile(harbor_client.__file__, client)

    def run(*arguments, success=True):
        result = subprocess.run(
            [sys.executable, "-I", "-S", str(client), "--url", url, *arguments],
            cwd=tmp_path, capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == (0 if success else 1), result.stderr
        if success:
            assert len(result.stdout.splitlines()) == 1
            return json.loads(result.stdout)

    state = run("state", "--compact", "--features")
    raw = server.service.state()
    assert state["queries_remaining"] == 16
    assert [row[0] for row in state["available"]] == list(range(18))
    assert state["revealed"] == []
    assert state["features"] == [
        [i, *[round(v, 4) for v in row]]
        for i, row in enumerate(raw["observation"]["features"])
    ]
    assert not {"reward", "info", "receiver", "split"} & state.keys()
    # Use a nonsequential order: dropping rows must never renumber actions.
    actions = [17, 3, 15, 0, 12, 2, 10, 1, 8, 4, 16, 5, 14, 6, 13, 7]
    for number, action in enumerate(actions, start=1):
        state = run("query", str(action), "--compact")
        assert "features" not in state
        assert state["queries_used"] == number
        assert state["queries_remaining"] == 16 - number
        assert state["index"] == action and state["acquired_label"] == 1
        assert state["revealed"] == [[i, 1] for i in sorted(actions[:number])]
        current = server.service.state()["observation"]
        assert [row[0] for row in state["available"]] == [
            i for i, valid in enumerate(current["action_mask"]) if valid
        ]
        for row in state["available"]:
            assert row[1:] == [round(v, 4) for v in current["scores"][row[0]]]
        if number < 16:
            assert state["reward"] == 0.0 and state["info"] == {}
        if number == 1:
            before = history_path.read_bytes()
            run("query", str(action), "--compact", success=False)
            assert history_path.read_bytes() == before
    assert state["terminated"] is True
    assert state["available"] == []
    assert state["reward"] == grade(harbor_bundle["path"], history_path)["reward"] == 1.0
    assert json.loads(history_path.read_text())["actions"] == actions


def test_compact_view_does_not_publish_padding_or_unknown_labels(episode):
    from active_calibration.harbor_client import compact_state

    env = ActiveCalibrationEnv([episode], budget=2, max_pool_size=9)
    obs, _ = env.reset(seed=0)
    raw = {"observation": {k: v.tolist() for k, v in obs.items()},
           "queries_used": 0, "budget": 2, "terminated": False}
    compact = compact_state(raw, include_features=True)
    assert [row[0] for row in compact["available"]] == list(range(6))
    assert [row[0] for row in compact["features"]] == list(range(6))
    assert compact["revealed"] == []
    # Static features can still be inspected after acquiring a row.
    obs, _, _, _, _ = env.step(3)
    raw.update(observation={k: v.tolist() for k, v in obs.items()}, queries_used=1)
    compact = compact_state(raw, include_features=True)
    assert [row[0] for row in compact["available"]] == [0, 1, 2, 4, 5]
    assert [row[0] for row in compact["features"]] == list(range(6))
    assert compact["revealed"] == [[3, int(obs["labels"][3])]]


def test_compact_feature_flag_requires_compact_mode_before_request(monkeypatch):
    from active_calibration import harbor_client

    def unexpected_fetch(*args):
        pytest.fail("invalid CLI arguments must not contact the service")

    monkeypatch.setattr(harbor_client, "fetch_state", unexpected_fetch)
    monkeypatch.setattr(harbor_client, "query", unexpected_fetch)
    for arguments in (["state", "--features"], ["query", "0", "--features"]):
        with pytest.raises(SystemExit) as error:
            harbor_client.main(arguments)
        assert error.value.code == 2


def test_service_restart_replays_spent_budget_instead_of_resetting_it(harbor_bundle, tmp_path):
    history_path = tmp_path / "query_history.json"
    service = CalibrationService(harbor_bundle["path"], history_path)
    for action in range(3):
        service.query(action)
    restored = CalibrationService(harbor_bundle["path"], history_path)
    assert restored.state() == service.state()
    with pytest.raises(ServiceError):
        restored.query(0)
    for action in range(3, 16):
        restored.query(action)
    exhausted = CalibrationService(harbor_bundle["path"], history_path)
    assert exhausted.state()["terminated"]
    with pytest.raises(ServiceError):
        exhausted.query(16)
    assert json.loads(history_path.read_text())["actions"] == list(range(16))


def test_generated_reference_executes_and_completes_a_replayable_episode(
    harbor_bundle, tmp_path, monkeypatch
):
    """Run the generated script so import collisions fail as they would in Harbor."""
    import active_calibration.harbor_task as task_module

    def synthetic_episodes(project_root=None, split="train", receivers=None):
        assert split == "eval" and list(receivers) == ["F"]
        return [harbor_bundle["episode"]]

    monkeypatch.setattr(task_module, "load_episodes", synthetic_episodes)
    task = task_module.build_task(tmp_path / "reference_task", receiver="F", seed=13)
    private_bundle = task / "environment/calibration/episode.npz"
    history_path = tmp_path / "reference_history.json"
    server = create_server(private_bundle, history_path, host="127.0.0.1", port=0)
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
    )
    thread.start()
    try:
        child_env = dict(
            os.environ,
            CSI_CALIBRATION_URL=f"http://127.0.0.1:{server.server_address[1]}",
            PYTHONPATH="",
            PYTHONDONTWRITEBYTECODE="1",
        )
        run = subprocess.run(
            [sys.executable, "-S", str(task / "solution/run_baseline.py")],
            cwd=task / "solution", env=child_env,
            capture_output=True, text=True, timeout=20,
        )
        assert run.returncode == 0, run.stderr
        assert server.service.state()["terminated"] is True
        assert server.service.state()["queries_used"] == 16
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()

    history = json.loads(history_path.read_text())
    actions = history["actions"]
    assert len(actions) == len(set(actions)) == 16
    assert all(type(action) is int and 0 <= action < 18 for action in actions)
    frozen, seed, budget, digest = load_bundle(private_bundle)
    assert history["bundle_sha256"] == digest
    env = ActiveCalibrationEnv([frozen], budget=budget)
    env.reset(seed=seed, options={"episode_index": 0})
    for action in actions:
        _, reward, terminated, truncated, info = env.step(action)
    assert terminated and not truncated
    report = grade(private_bundle, history_path)
    assert report["valid"] is True
    assert report["reward"] == reward == 1.0
    assert report["initial_accuracy"] == info["initial_accuracy"]
    assert report["final_accuracy"] == info["final_accuracy"]


@pytest.mark.parametrize("budget", [16.0, True, 15, 17])
def test_builder_rejects_invalid_budgets_before_loading_or_writing(tmp_path, monkeypatch, budget):
    """A successful build must not produce a budget the verifier cannot load."""
    import active_calibration.harbor_task as task_module

    def unexpected_load(*args, **kwargs):
        pytest.fail("invalid budgets must be rejected before reading recordings")

    monkeypatch.setattr(task_module, "load_episodes", unexpected_load)
    output = tmp_path / "new_parent" / "task"
    with pytest.raises(ValueError, match="integer budget of exactly 16"):
        task_module.build_task(output, budget=budget)
    assert not output.parent.exists()


def test_training_task_bundles_use_events_disjoint_from_evaluation(tmp_path):
    """Preparing LLM training tasks must not silently use evaluation recordings."""
    from active_calibration.data import load_episodes
    from active_calibration.harbor_task import build_task

    generated = {}
    for split in ("train", "eval"):
        task = build_task(tmp_path / split, receiver="A", seed=13, split=split)
        bundle_path = task / "tests/episode.npz"
        episode, seed, budget, digest = load_bundle(bundle_path)
        expected = load_episodes(split=split, receivers=["A"])[0]
        assert episode.split == split
        assert episode.pool_event_ids == expected.pool_event_ids
        assert episode.eval_event_ids == expected.eval_event_ids
        np.testing.assert_array_equal(episode.pool_y, expected.pool_y)
        np.testing.assert_array_equal(episode.eval_y, expected.eval_y)
        assert bundle_path.read_bytes() == (task / "environment/calibration/episode.npz").read_bytes()
        assert json.loads((task / "build_manifest.json").read_text())["split"] == split
        service = CalibrationService(bundle_path, tmp_path / f"{split}_history.json")
        for action in range(16):
            result = service.query(action)
        report = grade(bundle_path, tmp_path / f"{split}_history.json")
        assert report["valid"] is True
        assert report["reward"] == result["reward"]
        generated[split] = set(episode.pool_event_ids) | set(episode.eval_event_ids)
    assert generated["train"].isdisjoint(generated["eval"])


@pytest.mark.parametrize("split,receiver", [("train", "F"), ("train", "G"), ("test", "A")])
def test_builder_rejects_invalid_training_partitions_before_loading(
    tmp_path, monkeypatch, split, receiver
):
    import active_calibration.harbor_task as task_module

    def unexpected_load(*args, **kwargs):
        pytest.fail("invalid partitions must be rejected before reading recordings")

    monkeypatch.setattr(task_module, "load_episodes", unexpected_load)
    output = tmp_path / "new_parent" / "task"
    with pytest.raises(ValueError):
        task_module.build_task(output, split=split, receiver=receiver)
    assert not output.parent.exists()


def test_builder_makes_a_portable_task_with_private_bundle_outside_agent_image(
    harbor_bundle, tmp_path, monkeypatch
):
    import active_calibration.harbor_task as task_module

    def synthetic_episodes(project_root=None, split="train", receivers=None):
        assert split == "eval" and list(receivers) == ["F"]
        return [harbor_bundle["episode"]]

    monkeypatch.setattr(task_module, "load_episodes", synthetic_episodes)
    output = tmp_path / "generated"
    built = task_module.build_task(output, receiver="F", seed=13, budget=16)
    assert built == output
    with pytest.raises(FileExistsError):
        task_module.build_task(output, receiver="F", seed=13, budget=16)

    # Moving the generated directory must retain all private backend/verifier data.
    moved = tmp_path / "elsewhere" / "task"
    shutil.copytree(output, moved)
    assert not any(path.is_symlink() for path in moved.rglob("*"))
    backend_bundle = moved / "environment/calibration/episode.npz"
    verifier_bundle = moved / "tests/episode.npz"
    assert backend_bundle.read_bytes() == verifier_bundle.read_bytes()
    digest = hashlib.sha256(backend_bundle.read_bytes()).hexdigest()
    assert load_bundle(verifier_bundle)[3] == digest
    service = CalibrationService(backend_bundle, tmp_path / "portable_history.json")
    assert service.bundle_sha256 == digest

    dockerfile = moved / "environment/Dockerfile"
    copy_sources = []
    for line in dockerfile.read_text().splitlines():
        parts = shlex.split(line)
        if parts and parts[0].upper() == "COPY":
            assert not any(part.startswith("--from") for part in parts)
            copy_sources.extend(part for part in parts[1:-1] if not part.startswith("--"))
    assert copy_sources == ["client.py"]
    # Exercise exactly the client the agent image copies, without site packages.
    agent_root = tmp_path / "agent_image_contents"
    agent_root.mkdir()
    client = agent_root / "csi-calibration"
    shutil.copyfile(moved / "environment/client.py", client)
    private_server = create_server(
        backend_bundle, tmp_path / "portable_http_history.json", host="127.0.0.1", port=0
    )
    private_thread = threading.Thread(
        target=private_server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
    )
    private_thread.start()
    private_url = f"http://127.0.0.1:{private_server.server_address[1]}"
    try:
        state_run = subprocess.run(
            [sys.executable, "-I", "-S", str(client), "--url", private_url, "state"],
            cwd=agent_root, capture_output=True, text=True, timeout=10,
        )
        assert state_run.returncode == 0, state_run.stderr
        assert json.loads(state_run.stdout) == private_server.service.state()
        query_run = subprocess.run(
            [sys.executable, "-I", "-S", str(client), "--url", private_url, "query", "0"],
            cwd=agent_root, capture_output=True, text=True, timeout=10,
        )
        assert query_run.returncode == 0, query_run.stderr
        assert json.loads(query_run.stdout)["queries_used"] == 1
        assert private_server.service.state()["observation"]["labels"][0] == 1
    finally:
        private_server.shutdown()
        private_server.server_close()
        private_thread.join(timeout=5)
        assert not private_thread.is_alive()
    assert list(agent_root.iterdir()) == [client]

    # Run the copied verifier package, not the installed project, from elsewhere.
    history = tmp_path / "portable_complete_history.json"
    portable_bundle = dict(harbor_bundle, path=verifier_bundle)
    history.write_text(json.dumps(journal_for(portable_bundle)))
    logs = tmp_path / "portable_verifier_logs"
    child_env = dict(os.environ, PYTHONPATH=str(moved / "tests"))
    command = [
        sys.executable, "-c",
        "from pathlib import Path; import active_calibration; "
        "from active_calibration.harbor_grader import main; import sys; "
        "assert Path(active_calibration.__file__).resolve().is_relative_to(Path(sys.argv[1])); "
        "main(sys.argv[2:])",
        str(moved / "tests"), "--bundle", str(verifier_bundle),
        "--history", str(history), "--out-dir", str(logs),
    ]
    run = subprocess.run(
        command, cwd=tmp_path / "elsewhere", env=child_env,
        capture_output=True, text=True, timeout=20,
    )
    assert run.returncode == 0, run.stderr
    report = json.loads((logs / "report.json").read_text())
    assert report["valid"] is True and report["reward"] == 1.0
    assert (logs / "reward.txt").read_text().strip() == "1.0"
