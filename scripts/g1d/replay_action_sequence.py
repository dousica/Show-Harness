#!/usr/bin/env python3
"""Replay a VLM action sequence, one token at a time, through the Gumi web teleop.

Why this exists
---------------
The G1-D demo the user wants is:

    bearing on the table -> camera -> API -> VLM -> a sequence of atomic
    tokens -> MoveIt plans each step -> the arm executes them in order.

The VLM's decision is a JSON object, and the web teleop (Gumi) is the human
surface that shows the arm doing exactly that. This script is the BRIDGE
between the two: it takes the tokens the VLM produced (from a trace file, a
JSON file, or stdin) and POSTs them to the Gumi server's ``/api/step`` **one
token per call**, so the web UI's history strip, the MJPEG streams and the
saved rollout all show the sequence being reproduced step by step.

The Gumi server must already be running (it owns the robot):

    python3 gumi/collect_rollouts_web.py data/rollouts_g1d --robot g1d --port 8600

Input formats (all accepted):

1. A Gumi operator trace (``data/gpt_operator_traces/<session>/events.jsonl``):
   every line is an event; the ``decision`` events carry the model's
   ``left.action`` token. Those are extracted in order.

2. A plain JSON list of tokens: ``["MV_FWD", "MV_LEFT", "GRASP"]``

3. A JSON object with a ``tokens`` key: ``{"tokens": [...]}``

4. A JSON object with a ``left.action`` key (a single VLM decision)

5. Free text: whitespace/comma separated tokens (``MV_FWD MV_LEFT GRASP``)

Examples
--------
    # 1) Replay a saved VLM trace, one token per click-equivalent call.
    python3 scripts/g1d/replay_action_sequence.py \
        --teleop-url http://localhost:8600 \
        --trace data/gpt_operator_traces/20261027_101500_gpt-5.6-sol/events.jsonl

    # 2) Replay an explicit token list.
    python3 scripts/g1d/replay_action_sequence.py \
        --teleop-url http://localhost:8600 \
        --tokens "MV_FWD*3 MV_LEFT MV_DOWN GRASP"

    # 3) Read the tokens from stdin (e.g. the VLM's own stdout).
    echo '{"tokens": ["MV_FWD", "GRASP"]}' | \
        python3 scripts/g1d/replay_action_sequence.py --teleop-url http://localhost:8600

    # 4) Plan only -- resolve and print the sequence, send nothing.
    python3 scripts/g1d/replay_action_sequence.py --tokens "MV_FWD MV_LEFT" --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Optional, Sequence

import requests

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.action_units import GRIPPER_ATOMS, MOVE_ATOMS, ROTATE_ATOMS

#: Every token the Gumi web teleop accepts.
VALID_TOKENS = tuple(MOVE_ATOMS) + tuple(ROTATE_ATOMS) + tuple(GRIPPER_ATOMS)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--teleop-url",
        default="http://localhost:8600",
        help="the Gumi web-teleop base URL (default: http://localhost:8600)",
    )
    parser.add_argument(
        "--trace",
        type=Path,
        default=None,
        help="a Gumi operator trace (events.jsonl) to extract the VLM tokens from",
    )
    parser.add_argument(
        "--tokens",
        default=None,
        help="an explicit token list: 'MV_FWD*3 MV_LEFT GRASP' or a JSON list",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="a JSON file with the token list",
    )
    parser.add_argument(
        "--start-recording",
        action="store_true",
        help="POST /api/start before replaying (so the run is recorded + videoed)",
    )
    parser.add_argument(
        "--stop-recording",
        action="store_true",
        help="POST /api/stop after the last token (saves the rollout + video)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="resolve and print the sequence, but send nothing to the robot",
    )
    parser.add_argument(
        "--delay-s",
        type=float,
        default=0.0,
        help="pause between tokens (default: 0 -- as fast as the arm can go)",
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=60.0,
        help="per-request timeout (default: 60 s)",
    )
    parser.add_argument(
        "--stop-on-failure",
        action="store_true",
        default=True,
        help="abort the replay at the first failed token (default: on)",
    )
    parser.add_argument(
        "--continue-on-failure",
        dest="stop_on_failure",
        action="store_false",
        help="keep going after a failed token",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# token extraction
# ---------------------------------------------------------------------------
def _expand(text: str) -> list[str]:
    """'MV_FWD*3' -> three MV_FWD; 'w' is NOT expanded (aliases are the UI's job)."""
    out: list[str] = []
    for raw in str(text).replace(",", " ").split():
        token = raw.strip().upper()
        if not token:
            continue
        if "*" in token:
            name, _, count = token.partition("*")
            try:
                times = int(count)
            except ValueError:
                raise ValueError(f"bad repeat count in {raw!r}")
            if times < 1:
                raise ValueError(f"bad repeat count in {raw!r}")
            out.extend([name] * times)
        else:
            out.append(token)
    return out


def tokens_from_trace(path: Path) -> list[str]:
    """The VLM's tokens from a Gumi operator trace, in decision order.

    Each ``decision`` event carries the normalized decision, whose ``left.action``
    is the token the single arm should execute. Only EXECUTED decisions are
    replayed: an event whose ``outcome`` is ``paused`` / ``dry_run`` /
    ``oscillation_rejected`` / ``finish_rejected`` / ``stale_observation`` never
    reached the robot, so replaying it would fabricate motion.
    """
    tokens: list[str] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        if event.get("kind") != "decision":
            continue
        outcome = str(event.get("outcome", "")).lower()
        if outcome and outcome != "executed":
            continue
        decision = event.get("decision") or {}
        left = decision.get("left") or {}
        action = str(left.get("action", "")).upper().strip()
        if not action or action == "STILL":
            continue
        repeat = int(left.get("repeat", 1) or 1)
        tokens.extend([action] * max(1, repeat))
    return tokens


def tokens_from_json_text(text: str) -> list[str]:
    """Tokens from a JSON document, or from free text."""
    stripped = text.strip()
    if stripped.startswith("{") or stripped.startswith("["):
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON: {exc}") from exc
        return _tokens_from_payload(payload)
    return _expand(stripped)


def _tokens_from_payload(payload: Any) -> list[str]:
    if isinstance(payload, list):
        out: list[str] = []
        for item in payload:
            if isinstance(item, str):
                out.extend(_expand(item))
            elif isinstance(item, dict):
                action = str(item.get("action", "")).upper().strip()
                repeat = int(item.get("repeat", 1) or 1)
                if action and action != "STILL":
                    out.extend([action] * max(1, repeat))
        return out
    if isinstance(payload, dict):
        for key in ("tokens", "actions", "sequence", "action_sequence"):
            if key in payload:
                return _tokens_from_payload(payload[key])
        if "left" in payload and isinstance(payload["left"], dict):
            return _tokens_from_payload(payload["left"])
        if "action" in payload:
            return _tokens_from_payload([payload])
    raise ValueError("could not find a token list in the JSON payload")


def resolve_tokens(args: argparse.Namespace) -> list[str]:
    """The token sequence to replay, from whichever source was given."""
    tokens: list[str] = []
    if args.trace is not None:
        if not args.trace.is_file():
            raise FileNotFoundError(f"trace not found: {args.trace}")
        tokens.extend(tokens_from_trace(args.trace))
    if args.json is not None:
        if not args.json.is_file():
            raise FileNotFoundError(f"json file not found: {args.json}")
        tokens.extend(tokens_from_json_text(args.json.read_text(encoding="utf-8")))
    if args.tokens:
        tokens.extend(tokens_from_json_text(args.tokens))
    if not tokens:
        data = sys.stdin.read()
        if data.strip():
            tokens.extend(tokens_from_json_text(data))
    if not tokens:
        raise ValueError(
            "no tokens given: use --trace / --tokens / --json, or pipe them on stdin"
        )
    unknown = [t for t in tokens if t not in VALID_TOKENS]
    if unknown:
        raise ValueError(
            f"unsupported token(s): {', '.join(sorted(set(unknown)))}; "
            f"valid: {', '.join(VALID_TOKENS)}"
        )
    return tokens


# ---------------------------------------------------------------------------
# replay
# ---------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        tokens = resolve_tokens(args)
    except (ValueError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print("=" * 70)
    print(f"  G1-D action-sequence replay  ({len(tokens)} token(s))")
    print(f"  teleop : {args.teleop_url}")
    print("=" * 70)
    print(f"  sequence: {' '.join(tokens)}")
    if args.dry_run:
        print("  DRY RUN: nothing is sent to the robot.")
        return 0

    base = str(args.teleop_url).rstrip("/")
    session = requests.Session()

    try:
        state = session.get(base + "/api/state", timeout=args.timeout_s)
        state.raise_for_status()
        info = state.json()
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: cannot reach the Gumi teleop at {base}: {exc}")
        print(
            "Start it first:\n"
            "  python3 gumi/collect_rollouts_web.py data/rollouts_g1d --robot g1d --port 8600",
            file=sys.stderr,
        )
        return 2

    print(f"  robot   : {info.get('robot', '?')}  arm={info.get('arm', '?')}")
    print(f"  ee_pos  : {info.get('ee_pos')}")
    print(f"  step_m  : {info.get('step_m')}   yaw_step_rad: {info.get('yaw_step_rad')}")
    print("")

    if args.start_recording:
        try:
            result = session.post(base + "/api/start", json={}, timeout=args.timeout_s)
            payload = result.json()
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR: could not start the recording: {exc}", file=sys.stderr)
            return 2
        if not payload.get("ok"):
            print(f"ERROR: /api/start refused: {payload.get('message')}", file=sys.stderr)
            return 2
        print(f"  recording: {payload.get('message')}")

    failures = 0
    for index, token in enumerate(tokens, start=1):
        started = time.monotonic()
        try:
            response = session.post(
                base + "/api/step", json={"tokens": [token]}, timeout=args.timeout_s
            )
            payload = response.json()
        except Exception as exc:  # noqa: BLE001
            print(f"  [{index:02d}/{len(tokens):02d}] {token:<11} REQUEST FAILED: {exc}")
            failures += 1
            if args.stop_on_failure:
                break
            continue
        elapsed = time.monotonic() - started
        ok = bool(payload.get("ok"))
        message = str(payload.get("message", ""))
        mark = "OK " if ok else "FAIL"
        print(f"  [{index:02d}/{len(tokens):02d}] {token:<11} {mark} {elapsed:5.2f}s  {message}")
        if not ok:
            failures += 1
            if args.stop_on_failure:
                print("  -> aborting the replay (a failed token stops the sequence)")
                break
        if args.delay_s > 0.0:
            time.sleep(args.delay_s)

    if args.stop_recording and failures == 0:
        try:
            response = session.post(base + "/api/stop", json={}, timeout=args.timeout_s)
            payload = response.json()
        except Exception as exc:  # noqa: BLE001
            print(f"  WARNING: /api/stop failed: {exc}")
        else:
            print(f"  saved: {payload.get('message', payload)}")

    print("")
    print(f"  replayed {len(tokens) - failures}/{len(tokens)} token(s), {failures} failure(s)")
    return 0 if failures == 0 else 3


if __name__ == "__main__":
    raise SystemExit(main())
