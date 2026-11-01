#!/usr/bin/env python3
"""Public, standard-library-only client for the calibration label service."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_URL = os.environ.get("CSI_CALIBRATION_URL", "http://calibration:8080")


def _request(path: str, url: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
    data = None if body is None else json.dumps(body, allow_nan=False).encode("utf-8")
    headers = {} if data is None else {"Content-Type": "application/json"}
    request = Request(url.rstrip("/") + path, data=data, headers=headers)
    with urlopen(request, timeout=30) as response:
        result = json.load(response)
    if not isinstance(result, dict):
        raise ValueError("calibration service returned a non-object response")
    return result


def fetch_state(url: str = DEFAULT_URL) -> dict[str, Any]:
    """Fetch the current observation without acquiring any labels."""
    return _request("/state", url)


def query(index: int, url: str = DEFAULT_URL) -> dict[str, Any]:
    """Acquire one available row's label; HTTP errors do not consume a query."""
    if type(index) is not int:
        raise ValueError("index must be an integer")
    return _request("/query", url, {"index": index})


def compact_state(state: dict[str, Any], *, include_features: bool = False) -> dict[str, Any]:
    """Render only public observation fields for a small terminal transcript.

    Scores/features are display values rounded to four decimal places. The
    service and classifier keep their original precision. Row indices are
    stable, even when acquired rows disappear from the available-row table.
    """
    observation = state["observation"]

    def rounded(row):
        return [round(float(value), 4) + 0.0 for value in row]

    result = {
        "queries_used": state["queries_used"],
        "queries_remaining": state["budget"] - state["queries_used"],
        "budget": state["budget"],
        "terminated": state["terminated"],
        "score_columns": ["index", *[f"class_{i}" for i in range(8)]],
        "available": [
            [i, *rounded(observation["scores"][i])]
            for i, valid in enumerate(observation["action_mask"]) if valid
        ],
        "label_columns": ["index", "label"],
        "revealed": [
            [i, observation["labels"][i]]
            for i, queried in enumerate(observation["queried_mask"]) if queried
        ],
    }
    if include_features:
        # Static features need only be requested once. Include acquired rows
        # as well, so a later request can still support diversity strategies.
        result["feature_columns"] = "index followed by normalized window features"
        result["features"] = [
            [i, *rounded(observation["features"][i])]
            for i, (valid, queried) in enumerate(zip(
                observation["action_mask"], observation["queried_mask"]
            )) if valid or queried
        ]
    if "acquired_label" in state:
        result.update({key: state[key] for key in ("index", "acquired_label", "reward", "info")})
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL, help="calibration service base URL")
    subparsers = parser.add_subparsers(dest="command", required=True)
    state_parser = subparsers.add_parser("state", help="show the current observation")
    query_parser = subparsers.add_parser("query", help="acquire a row's activity label")
    query_parser.add_argument("index", type=int)
    # Accept the common forms both `--url URL state` and `state --url URL`.
    for child in (state_parser, query_parser):
        child.add_argument("--url", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
        child.add_argument("--compact", action="store_true",
                           help="small JSON view: available scores and revealed labels, rounded to 4 decimals")
        child.add_argument("--features", action="store_true",
                           help="include static features in the compact view (usually needed only once)")
    args = parser.parse_args(argv)
    if args.features and not args.compact:
        parser.error("--features requires --compact; the default view already includes all features")
    try:
        result = fetch_state(args.url) if args.command == "state" else query(args.index, args.url)
        if args.compact:
            result = compact_state(result, include_features=args.features)
    except HTTPError as error:
        try:
            detail = json.loads(error.read().decode("utf-8")).get("error", error.reason)
        except (ValueError, UnicodeDecodeError, AttributeError):
            detail = error.reason
        print(f"calibration service HTTP {error.code}: {detail}", file=sys.stderr)
        return 1
    except (URLError, TimeoutError, ValueError, OSError) as error:
        print(f"calibration request failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=None if args.compact else 2,
                     separators=(",", ":") if args.compact else None, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
