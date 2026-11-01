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
    args = parser.parse_args(argv)
    try:
        result = fetch_state(args.url) if args.command == "state" else query(args.index, args.url)
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
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
