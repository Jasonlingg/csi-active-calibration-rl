"""Replay a trusted sidecar query journal in an isolated Harbor verifier.

No reported score or revealed label is accepted as grading input. The frozen
private episode and its raw-file hash bind the journal to one receiver, seed,
row permutation, and 16-query budget. Harbor collects the journal directly
from the sidecar; an agent-authored journal is not an equivalent trust boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import re
from typing import Any, Sequence

import numpy as np

from .data import Episode
from .env import ActiveCalibrationEnv
from .learner import _feature_matrix, _labels


QUERY_BUDGET = 16
MAX_HISTORY_BYTES = 16 * 1024
BUNDLE_KEYS = frozenset({
    "source_f", "source_y", "pool_f", "pool_y", "eval_f", "eval_y",
    "pool_event_ids", "eval_event_ids", "receiver", "split", "seed", "budget",
})
HISTORY_KEYS = frozenset({"schema_version", "bundle_sha256", "seed", "budget", "actions"})


def _integer_scalar(value: np.ndarray, name: str) -> int:
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind not in "iu":
        raise ValueError(f"{name} must be an integer scalar")
    return int(array.item())


def _string_scalar(value: np.ndarray, name: str) -> str:
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind != "U":
        raise ValueError(f"{name} must be a Unicode string scalar")
    result = str(array.item())
    if not result.strip() or "\x00" in result:
        raise ValueError(f"{name} must be a nonempty string without NUL characters")
    return result


def _event_ids(value: np.ndarray, length: int, name: str) -> tuple[int, ...]:
    array = np.asarray(value)
    if array.shape != (length,) or array.dtype.kind not in "iu":
        raise ValueError(f"{name} must be an integer vector with {length} entries")
    result = tuple(int(item) for item in array)
    if len(set(result)) != length:
        raise ValueError(f"{name} must contain unique recording event IDs")
    return result


def load_bundle(bundle_path: str | Path) -> tuple[Episode, int, int, str]:
    """Load one frozen episode; return ``(episode, seed, budget, SHA256)``.

    The bundle is verifier/sidecar-owned data, not an agent submission. Reading
    one byte snapshot ensures that the parsed arrays and recorded hash refer
    to the same bytes. NumPy object deserialization is always disabled.
    """
    raw = Path(bundle_path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        if len(archive.files) != len(BUNDLE_KEYS) or set(archive.files) != BUNDLE_KEYS:
            raise ValueError("private bundle fields do not match the required schema")
        seed = _integer_scalar(archive["seed"], "seed")
        budget = _integer_scalar(archive["budget"], "budget")
        if seed < 0:
            raise ValueError("seed must be nonnegative")
        if budget != QUERY_BUDGET:
            raise ValueError(f"the Harbor episode budget must be exactly {QUERY_BUDGET}")
        matrices: dict[str, np.ndarray] = {}
        width: int | None = None
        for name in ("source_f", "pool_f", "eval_f"):
            value = archive[name]
            if value.dtype.kind not in "fiu":
                raise ValueError(f"{name} must contain real numeric features")
            matrices[name] = _feature_matrix(value, name, width)
            width = matrices[name].shape[1]
        labels: dict[str, np.ndarray] = {}
        for name, features_name in (("source_y", "source_f"), ("pool_y", "pool_f"),
                                    ("eval_y", "eval_f")):
            value = archive[name]
            if value.dtype.kind not in "iu":
                raise ValueError(f"{name} must contain integer class IDs")
            labels[name] = _labels(value, len(matrices[features_name]), 8, name)
        if len(matrices["pool_f"]) < budget:
            raise ValueError("the query pool is smaller than the fixed query budget")
        pool_ids = _event_ids(archive["pool_event_ids"], len(matrices["pool_f"]), "pool_event_ids")
        eval_ids = _event_ids(archive["eval_event_ids"], len(matrices["eval_f"]), "eval_event_ids")
        if set(pool_ids) & set(eval_ids):
            raise ValueError("query and reward recording events must be disjoint")
        episode = Episode(
            **matrices, **labels, pool_event_ids=pool_ids, eval_event_ids=eval_ids,
            receiver=_string_scalar(archive["receiver"], "receiver"),
            split=_string_scalar(archive["split"], "split"),
        )
    return episode, seed, budget, digest


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _load_history(history_path: str | Path) -> dict[str, Any]:
    with Path(history_path).open("rb") as stream:
        raw = stream.read(MAX_HISTORY_BYTES + 1)
    if len(raw) > MAX_HISTORY_BYTES:
        raise ValueError(f"history exceeds the {MAX_HISTORY_BYTES}-byte limit")
    history = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object,
                         parse_constant=_reject_constant)
    if not isinstance(history, dict):
        raise ValueError("history must be a JSON object")
    if set(history) != HISTORY_KEYS:
        raise ValueError("history fields do not match the required schema")
    return history


def _invalid(reason: str, queries: int = 0) -> dict[str, Any]:
    return {"reward": -1.0, "valid": False, "reason": reason, "queries": queries}


def grade(bundle_path: str | Path, history_path: str | Path) -> dict[str, Any]:
    """Return the exact Gym terminal accuracy gain, or -1 for invalid history."""
    queries = 0
    try:
        episode, seed, budget, digest = load_bundle(bundle_path)
    except Exception as exc:
        return _invalid(f"invalid private bundle: {type(exc).__name__}: {exc}")
    try:
        history = _load_history(history_path)
        if type(history["schema_version"]) is not int or history["schema_version"] != 1:
            raise ValueError("schema_version must be integer 1")
        claimed_hash = history["bundle_sha256"]
        if not isinstance(claimed_hash, str) or re.fullmatch(r"[0-9a-f]{64}", claimed_hash) is None:
            raise ValueError("bundle_sha256 must be a lowercase SHA-256 hex string")
        if claimed_hash != digest:
            raise ValueError("history does not match the frozen bundle hash")
        for name, expected in (("seed", seed), ("budget", budget)):
            if type(history[name]) is not int or history[name] != expected:
                raise ValueError(f"history {name} does not match the frozen bundle")
        actions = history["actions"]
        if not isinstance(actions, list):
            raise ValueError("actions must be a JSON array")
        queries = len(actions)
        if queries != QUERY_BUDGET:
            raise ValueError(f"history must contain exactly {QUERY_BUDGET} actions")
        if any(type(action) is not int for action in actions):
            raise ValueError("every action must be an integer, not a boolean or float")
        if len(set(actions)) != len(actions):
            raise ValueError("history repeats a query action")
        if any(action < 0 or action >= len(episode.pool_f) for action in actions):
            raise ValueError("history contains an out-of-range action")

        env = ActiveCalibrationEnv([episode], budget=budget)
        env.reset(seed=seed, options={"episode_index": 0})
        final_info: dict[str, Any] = {}
        final_reward = 0.0
        for index, action in enumerate(actions):
            _, reward, terminated, truncated, info = env.step(action)
            if truncated or terminated != (index == budget - 1):
                raise ValueError("replayed episode did not terminate at the fixed budget")
            if not terminated and reward != 0.0:
                raise ValueError("replayed episode returned a nonterminal reward")
            if terminated:
                final_reward, final_info = float(reward), info
        initial_accuracy = float(final_info["initial_accuracy"])
        final_accuracy = float(final_info["final_accuracy"])
        gain = final_accuracy - initial_accuracy
        if (not all(math.isfinite(value) for value in (initial_accuracy, final_accuracy, gain, final_reward))
                or not 0.0 <= initial_accuracy <= 1.0 or not 0.0 <= final_accuracy <= 1.0):
            raise ValueError("replayed episode produced invalid accuracy values")
        if final_reward != gain:
            raise ValueError("terminal Gym reward does not equal final minus initial accuracy")
        return {
            "reward": final_reward, "valid": True,
            "reason": f"completed exactly {budget} legal queries", "queries": queries,
            "initial_accuracy": initial_accuracy, "final_accuracy": final_accuracy,
            "accuracy_gain": gain,
        }
    except Exception as exc:
        return _invalid(f"invalid history or replay: {type(exc).__name__}: {exc}", queries)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--history", required=True, type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("/logs/verifier"))
    args = parser.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    reward_path = args.out_dir / "reward.txt"
    # Leave a failing reward behind if an unexpected grading error occurs.
    reward_path.write_text("-1.0\n", encoding="utf-8")
    try:
        result = grade(args.bundle, args.history)
    except Exception as exc:
        result = _invalid(f"unexpected grader failure: {type(exc).__name__}: {exc}")
    (args.out_dir / "report.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    reward_path.write_text(f"{result['reward']}\n", encoding="utf-8")
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
