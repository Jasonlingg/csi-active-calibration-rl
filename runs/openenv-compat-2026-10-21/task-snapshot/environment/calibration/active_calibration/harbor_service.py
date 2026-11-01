"""Private HTTP adapter for a single budgeted calibration episode.

Only this sidecar holds query labels and reward examples. The agent receives
the small standard-library client, not this module or the episode bundle.
"""

from __future__ import annotations

import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import tempfile
import threading
from typing import Any

import numpy as np

from .data import Episode
from .env import ActiveCalibrationEnv


MAX_BODY_BYTES = 4096
_BUNDLE_FIELDS = {
    "source_f", "source_y", "pool_f", "pool_y", "eval_f", "eval_y",
    "pool_event_ids", "eval_event_ids", "receiver", "split", "seed", "budget",
}


class ServiceError(Exception):
    """An invalid client request, with a deliberate HTTP response status."""

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


def _integer_scalar(value: np.ndarray, name: str) -> int:
    if value.shape != () or value.dtype.kind not in "iu":
        raise ValueError(f"bundle {name} must be an integer scalar")
    return int(value.item())


def _string_scalar(value: np.ndarray, name: str) -> str:
    if value.shape != () or value.dtype.kind != "U":
        raise ValueError(f"bundle {name} must be a Unicode string scalar")
    return str(value.item())


def _event_ids(value: np.ndarray, length: int, name: str) -> tuple[int, ...]:
    if value.shape != (length,) or value.dtype.kind not in "iu":
        raise ValueError(f"bundle {name} must contain one integer event ID per example")
    result = tuple(int(item) for item in value)
    if len(set(result)) != len(result):
        raise ValueError(f"bundle {name} must not contain duplicate events")
    return result


def _load_bundle(path: Path) -> tuple[Episode, int, int, str]:
    raw = path.read_bytes()
    # Hash exactly the archive that NumPy reads, even if its path is replaced.
    import io
    with np.load(io.BytesIO(raw), allow_pickle=False) as bundle:
        if set(bundle.files) != _BUNDLE_FIELDS:
            raise ValueError("episode bundle has unexpected or missing fields")
        values = {name: bundle[name] for name in _BUNDLE_FIELDS}
    seed = _integer_scalar(values["seed"], "seed")
    budget = _integer_scalar(values["budget"], "budget")
    if seed < 0:
        raise ValueError("bundle seed must be nonnegative")
    pool_ids = _event_ids(values["pool_event_ids"], len(values["pool_f"]), "pool_event_ids")
    eval_ids = _event_ids(values["eval_event_ids"], len(values["eval_f"]), "eval_event_ids")
    if set(pool_ids) & set(eval_ids):
        raise ValueError("bundle query and reward events must be disjoint")
    episode = Episode(
        source_f=values["source_f"], source_y=values["source_y"],
        pool_f=values["pool_f"], pool_y=values["pool_y"],
        eval_f=values["eval_f"], eval_y=values["eval_y"],
        receiver=_string_scalar(values["receiver"], "receiver"),
        split=_string_scalar(values["split"], "split"),
        pool_event_ids=pool_ids, eval_event_ids=eval_ids,
    )
    return episode, seed, budget, hashlib.sha256(raw).hexdigest()


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate object field")
        result[key] = value
    return result


class CalibrationService:
    """One episode shared by all HTTP clients, with an atomic action journal."""

    def __init__(self, bundle_path: str | Path, history_path: str | Path) -> None:
        episode, self.seed, self.budget, self.bundle_sha256 = _load_bundle(Path(bundle_path))
        self._env = ActiveCalibrationEnv([episode], budget=self.budget)
        self._observation, _ = self._env.reset(
            seed=self.seed, options={"episode_index": 0}
        )
        self._terminated = False
        self._actions: list[int] = []
        self._lock = threading.RLock()
        self._history_path = Path(history_path)
        # A sidecar restart must not grant another label budget. An existing
        # journal is accepted only for this exact bundle and is replayed.
        if self._history_path.exists():
            self._restore_history()
        self._write_history()

    def _journal(self, actions: list[int] | None = None) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "bundle_sha256": self.bundle_sha256,
            "seed": self.seed,
            "budget": self.budget,
            "actions": list(self._actions if actions is None else actions),
        }

    def _restore_history(self) -> None:
        previous = json.loads(self._history_path.read_text(encoding="utf-8"),
                              object_pairs_hook=_strict_object)
        if (not isinstance(previous, dict)
                or set(previous) != set(self._journal())
                or any(previous[key] != self._journal()[key]
                       for key in ("schema_version", "bundle_sha256", "seed", "budget"))
                or not isinstance(previous["actions"], list)):
            raise ValueError("existing query history does not match this episode")
        for action in previous["actions"]:
            if (type(action) is not int or self._terminated
                    or not self._env.action_space.contains(action)
                    or not self._env.action_masks()[action]):
                raise ValueError("existing query history contains an invalid action")
            self._observation, _, self._terminated, _, _ = self._env.step(action)
            self._actions.append(action)

    def _write_history(self, actions: list[int] | None = None) -> None:
        self._history_path.parent.mkdir(parents=True, exist_ok=True)
        temporary: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=self._history_path.parent,
                prefix=f".{self._history_path.name}.", suffix=".tmp", delete=False,
            ) as stream:
                temporary = stream.name
                json.dump(self._journal(actions), stream, indent=2, allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self._history_path)
        finally:
            if temporary is not None:
                try:
                    os.unlink(temporary)
                except FileNotFoundError:
                    pass

    def _state_unlocked(self) -> dict[str, Any]:
        return {
            "observation": {key: value.tolist() for key, value in self._observation.items()},
            "queries_used": len(self._actions),
            "budget": self.budget,
            "terminated": self._terminated,
        }

    def state(self) -> dict[str, Any]:
        with self._lock:
            return self._state_unlocked()

    def query(self, index: int) -> dict[str, Any]:
        with self._lock:
            if type(index) is not int:
                raise ServiceError("index must be an integer")
            if self._terminated:
                raise ServiceError("the query budget is exhausted", status=409)
            if (not self._env.action_space.contains(index)
                    or not self._env.action_masks()[index]):
                raise ServiceError("index must select an available pool row", status=409)
            # Persist the accepted action while holding the same lock as the
            # transition. A process interruption can then be recovered by
            # deterministic replay, without silently replenishing the budget.
            self._write_history(self._actions + [index])
            self._observation, reward, self._terminated, truncated, info = self._env.step(index)
            if truncated:
                raise RuntimeError("calibration episode unexpectedly truncated")
            self._actions.append(index)
            result = self._state_unlocked()
            result.update({
                "acquired_label": info["label"],
                "index": index,
                "reward": float(reward),
                "info": info if self._terminated else {},
            })
            return result


class CalibrationHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], service: CalibrationService) -> None:
        self.service = service
        super().__init__(address, CalibrationRequestHandler)


class CalibrationRequestHandler(BaseHTTPRequestHandler):
    server: CalibrationHTTPServer

    def _respond(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)
        self.close_connection = True

    def do_GET(self) -> None:
        if self.path == "/health":
            self._respond(200, {"ok": True})
        elif self.path == "/state":
            self._respond(200, self.server.service.state())
        else:
            self._respond(404, {"error": "unknown endpoint"})

    def _query_index(self) -> int:
        if self.headers.get("Transfer-Encoding") is not None:
            raise ServiceError("chunked request bodies are unsupported")
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as error:
            raise ServiceError("invalid Content-Length") from error
        if length <= 0 or length > MAX_BODY_BYTES:
            raise ServiceError("request body must contain 1 to 4096 bytes")
        raw = self.rfile.read(length)
        try:
            body = json.loads(raw.decode("utf-8"), object_pairs_hook=_strict_object)
        except (UnicodeDecodeError, ValueError) as error:
            raise ServiceError("request body must be valid JSON") from error
        if not isinstance(body, dict) or set(body) != {"index"}:
            raise ServiceError("request body must contain exactly the index field")
        if type(body["index"]) is not int:
            raise ServiceError("index must be an integer")
        return body["index"]

    def do_POST(self) -> None:
        if self.path != "/query":
            self._respond(404, {"error": "unknown endpoint"})
            return
        try:
            index = self._query_index()
            result = self.server.service.query(index)
        except ServiceError as error:
            self._respond(error.status, {"error": str(error)})
        except Exception:
            # Do not expose bundle paths, data, or internal tracebacks over HTTP.
            self._respond(500, {"error": "internal service error"})
        else:
            self._respond(200, result)

    def log_message(self, format: str, *args: Any) -> None:
        return


def create_server(bundle_path: str | Path, history_path: str | Path,
                  host: str = "0.0.0.0", port: int = 8080) -> CalibrationHTTPServer:
    """Construct a server; callers may use port zero for a loopback smoke test."""
    return CalibrationHTTPServer((host, port), CalibrationService(bundle_path, history_path))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    server = create_server(args.bundle, args.history, args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
