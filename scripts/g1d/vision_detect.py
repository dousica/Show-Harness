#!/usr/bin/env python3
"""Send G1-D camera frames to the vision API and print one JSON result per call.

This is the perception-only half of the G1-D pipeline: it grabs a fresh RGB frame
from the camera bridge, asks the configured vision API to find a named object,
and prints a validated JSON detection. It never subscribes to depth, never reads
a TF, and never sends a robot command -- ``action_sent`` is always ``false``.

Examples
--------
OpenAI-compatible endpoint (the harness's standard VLM config)::

    python scripts/g1d/vision_detect.py --target "red steel bearing"

Any other HTTP API (point the ``vision_api`` block at it)::

    python scripts/g1d/vision_detect.py --target bearing --transport custom

The printed line is the record a downstream planner consumes::

    {"found": true, "label": "bearing", "bbox_norm": [0.31, 0.42, 0.55, 0.61], ...}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import load_secrets_env  # noqa: E402
from core.g1d.vision_api import VisionApiError, build_vision_client  # noqa: E402


def _load_config(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def _read_image(path: Path) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--topic",
        default=None,
        help="ROS 2 image topic to read; e.g. /g1d/cameras/head_left/image_raw",
    )
    source_group.add_argument(
        "--image",
        type=Path,
        help="read a still image from disk instead of a camera topic",
    )
    parser.add_argument("--target", required=True, help="object description for the API")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/robot_g1d.yaml",
        help="robot config carrying the vlm / vision_api blocks",
    )
    parser.add_argument("--backend", default=None, help="configured VLM backend name")
    parser.add_argument(
        "--transport",
        default=None,
        choices=["openai_compat", "custom"],
        help="override the transport from the config file",
    )
    parser.add_argument(
        "--rate-hz", type=float, default=0.2,
        help="maximum API request frequency; default 1 call per 5 seconds",
    )
    parser.add_argument(
        "--count", type=int, default=1,
        help="number of calls; use 0 to keep calling until Ctrl+C",
    )
    parser.add_argument(
        "--max-image-age-s", type=float, default=0.5,
        help="reject a camera frame older than this",
    )
    parser.add_argument(
        "--min-confidence", type=float, default=None,
        help="override the configured minimum confidence",
    )
    parser.add_argument(
        "--max-image-side", type=int, default=None,
        help="downscale the frame so its longest side is at most this many pixels",
    )
    parser.add_argument(
        "--save-frames",
        type=Path,
        default=None,
        help="directory to save each sent frame as PNG (for auditing)",
    )
    parser.add_argument(
        "--extra-instruction", default="",
        help="extra instruction appended to the detection prompt",
    )
    args = parser.parse_args(argv)

    if not args.target.strip():
        parser.error("--target must not be empty")
    if not np.isfinite(args.rate_hz) or not 0 < args.rate_hz <= 2.0:
        parser.error("--rate-hz must be in (0, 2]; continuous requests are rate limited")
    if args.count < 0:
        parser.error("--count cannot be negative")
    if not np.isfinite(args.max_image_age_s) or args.max_image_age_s <= 0:
        parser.error("--max-image-age-s must be finite and positive")
    if not args.config.is_file():
        parser.error(f"config file does not exist: {args.config}")

    load_secrets_env()
    try:
        cfg = _load_config(args.config)
        client = build_vision_client(
            cfg, backend=args.backend, transport=args.transport
        )
    except Exception as exc:  # noqa: BLE001 - reported as a CLI error
        parser.error(f"could not initialize the vision client: {exc}")

    if args.min_confidence is not None:
        client.min_confidence = float(args.min_confidence)
    if args.max_image_side is not None:
        client.max_image_side = int(args.max_image_side)

    source = None
    if args.topic:
        from core.g1d.camera_source import G1DCameraSource

        source = G1DCameraSource(args.topic, max_age_s=args.max_image_age_s)
        try:
            source.start()
        except ImportError as exc:
            parser.error(str(exc))

    print(
        f"source: {args.topic or str(args.image)}; target: {args.target!r}; "
        f"transport: {client.transport.name}; model: {client.model or '(unset)'}",
        file=sys.stderr,
    )
    print("perception only: no depth, no TF, no robot command.", file=sys.stderr)

    calls = 0
    interval = 1.0 / args.rate_hz
    next_call = time.monotonic()
    try:
        while args.count == 0 or calls < args.count:
            if source is not None:
                frame = source.read_fresh(timeout_s=5.0)
            else:
                frame = _read_image(args.image)
            if args.save_frames is not None:
                from core.record.images import save_png

                save_png(args.save_frames / f"frame_{calls:04d}.png", frame)
            result = client.detect(
                frame, args.target, extra_instruction=args.extra_instruction
            )
            print(json.dumps(result.to_dict(), ensure_ascii=False))
            calls += 1
            if args.count == 0 or calls < args.count:
                next_call += interval
                delay = next_call - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
        return 0
    except (VisionApiError, RuntimeError, ValueError, KeyboardInterrupt) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    finally:
        if source is not None:
            source.stop()


if __name__ == "__main__":
    raise SystemExit(main())
