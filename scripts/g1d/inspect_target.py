#!/usr/bin/env python3
"""Call the configured Show-Harness VLM to locate a named object in RGB images.

This is perception-only: it captures RGB frames at a fixed maximum rate and
prints a validated JSON detection. With the measured calibration (stereo +
eye-to-hand, see configs/robot_g1d.yaml's ``calibration`` block) it also
rectifies, matches metric depth, and places the detection in the robot's base
frame. It never sends a robot command. The existing camera bridge publishes
BGR8; convert to RGB before sending.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Sequence

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.duration import Duration
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformException, TransformListener

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from core.config import load_secrets_env, make_api_key_refresher, resolve_vlm_config
from core.g1d.calibration import (
    CalibrationBundle,
    CalibrationError,
    load_calibration,
    rectified_base_transform,
)
from core.g1d.vision_geometry import (
    StereoCalibration,
    StereoDepthProcessor,
    VLMDetection,
    estimate_box_depth,
    pixel_to_camera_xyz,
)
SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string"},
        "bbox_norm": {
            "type": "array",
            "items": {"type": "number"},
            "minItems": 4,
            "maxItems": 4,
        },
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
    "required": ["label", "bbox_norm", "confidence"],
    "additionalProperties": False,
}


class LatestRgbSubscriber(Node):
    def __init__(self, topic: str, max_age_s: float) -> None:
        super().__init__("harness_g1d_rgb_observer")
        self.max_age_s = max_age_s
        self.latest: np.ndarray | None = None
        self.received_at: float | None = None
        self.encoding: str | None = None
        self.subscription = self.create_subscription(
            Image, topic, self._on_image, qos_profile_sensor_data
        )

    def _on_image(self, msg: Image) -> None:
        try:
            encoding = msg.encoding.lower()
            if encoding not in ("bgr8", "rgb8"):
                raise ValueError(f"expected bgr8/rgb8, received {msg.encoding!r}")
            raw = np.frombuffer(msg.data, dtype=np.uint8)
            frame = raw.reshape(msg.height, msg.step)[:, :msg.width * 3]
            frame = frame.reshape(msg.height, msg.width, 3)
            if encoding == "bgr8":
                frame = frame[:, :, ::-1]
            self.latest = np.ascontiguousarray(frame)
            self.received_at = time.monotonic()
            self.encoding = encoding
        except (ValueError, IndexError) as exc:
            self.get_logger().error(f"dropping malformed image: {exc}")

    def read_fresh(self, timeout_s: float = 5.0) -> np.ndarray:
        deadline = time.monotonic() + timeout_s
        while rclpy.ok() and time.monotonic() < deadline:
            if (
                self.latest is not None
                and self.received_at is not None
                and time.monotonic() - self.received_at <= self.max_age_s
            ):
                return self.latest.copy()
            rclpy.spin_once(self, timeout_sec=0.05)
        raise RuntimeError("no fresh RGB image received before timeout")


class LatestStereoSubscriber(LatestRgbSubscriber):
    def __init__(
        self,
        left_topic: str,
        right_topic: str,
        left_info_topic: str,
        right_info_topic: str,
        *,
        max_age_s: float,
        max_skew_s: float,
    ) -> None:
        super().__init__(left_topic, max_age_s)
        self.max_skew_s = max_skew_s
        self.right: tuple[np.ndarray, float, float] | None = None
        self.left_stamp_s: float | None = None
        self.right_info: CameraInfo | None = None
        self.left_frame_id: str | None = None
        self.left_stamp_ns: int | None = None
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._right_sub = self.create_subscription(
            Image, right_topic, self._on_right, qos_profile_sensor_data
        )
        self._left_info_sub = self.create_subscription(
            CameraInfo, left_info_topic, self._on_left_info, qos_profile_sensor_data
        )
        self._right_info_sub = self.create_subscription(
            CameraInfo, right_info_topic, self._on_right_info, qos_profile_sensor_data
        )
        self.left_info: CameraInfo | None = None

    @staticmethod
    def _stamp(msg: Image | CameraInfo) -> float:
        return float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9

    @staticmethod
    def _decode(msg: Image) -> np.ndarray:
        encoding = msg.encoding.lower()
        if encoding not in ("bgr8", "rgb8"):
            raise ValueError(f"expected bgr8/rgb8, received {msg.encoding!r}")
        raw = np.frombuffer(msg.data, dtype=np.uint8)
        frame = raw.reshape(msg.height, msg.step)[:, :msg.width * 3]
        frame = frame.reshape(msg.height, msg.width, 3)
        if encoding == "bgr8":
            frame = frame[:, :, ::-1]
        return np.ascontiguousarray(frame)

    def _on_image(self, msg: Image) -> None:
        try:
            self.latest = self._decode(msg)
            self.received_at = time.monotonic()
            self.encoding = msg.encoding.lower()
            self.left_stamp_s = self._stamp(msg)
            self.left_stamp_ns = int(msg.header.stamp.sec) * 1_000_000_000 + int(
                msg.header.stamp.nanosec
            )
            self.left_frame_id = msg.header.frame_id
        except (ValueError, IndexError) as exc:
            self.get_logger().error(f"dropping malformed left image: {exc}")

    def _on_right(self, msg: Image) -> None:
        try:
            self.right = (self._decode(msg), self._stamp(msg), time.monotonic())
        except (ValueError, IndexError) as exc:
            self.get_logger().error(f"dropping malformed right image: {exc}")

    def _on_left_info(self, msg: CameraInfo) -> None:
        self.left_info = msg

    def _on_right_info(self, msg: CameraInfo) -> None:
        self.right_info = msg

    def validate_camera_info(self, calibration: StereoCalibration) -> None:
        for side, info in (("left", self.left_info), ("right", self.right_info)):
            if info is None:
                raise RuntimeError(f"missing {side} CameraInfo")
            if (info.width, info.height) != (
                calibration.image_width,
                calibration.image_height,
            ):
                raise RuntimeError(
                    f"{side} CameraInfo dimensions do not match the stereo calibration"
                )
            if len(info.k) != 9 or info.k[0] <= 0 or info.k[4] <= 0:
                raise RuntimeError(
                    f"{side} CameraInfo has no valid intrinsic matrix; calibrate the camera first"
                )
        if not np.allclose(
            self.left_info.k, calibration.K_left.reshape(-1), rtol=1e-3, atol=1e-3
        ):
            raise RuntimeError("left CameraInfo K differs from the stereo calibration file")
        if not np.allclose(
            self.right_info.k, calibration.K_right.reshape(-1), rtol=1e-3, atol=1e-3
        ):
            raise RuntimeError("right CameraInfo K differs from the stereo calibration file")

    def read_fresh_pair(self, timeout_s: float = 5.0):
        deadline = time.monotonic() + timeout_s
        while rclpy.ok() and time.monotonic() < deadline:
            now = time.monotonic()
            ready = (
                self.latest is not None
                and self.received_at is not None
                and now - self.received_at <= self.max_age_s
                and self.right is not None
                and now - self.right[2] <= self.max_age_s
                and self.left_stamp_s is not None
                and abs(self.left_stamp_s - self.right[1]) <= self.max_skew_s
            )
            if ready:
                return (
                    self.latest.copy(),
                    self.right[0].copy(),
                    self.left_stamp_s,
                )
            rclpy.spin_once(self, timeout_sec=0.05)
        raise RuntimeError(
            "no fresh synchronized stereo pair; check paired timestamps and camera topics"
        )

    def base_transform_for_rectified_left(self, base_frame: str, processor: StereoDepthProcessor):
        if not base_frame:
            return None
        from core.g1d.calibration import rectified_base_transform as _rectified_base

        if not self.left_frame_id or self.left_stamp_ns is None:
            raise RuntimeError("left image frame_id/timestamp is missing")
        try:
            transform = self._tf_buffer.lookup_transform(
                base_frame,
                self.left_frame_id,
                Time(nanoseconds=self.left_stamp_ns, clock_type=self.get_clock().clock_type),
                timeout=Duration(seconds=0.5),
            ).transform
        except TransformException as exc:
            raise RuntimeError(
                f"no time-aligned TF {base_frame} <- {self.left_frame_id}: {exc}"
            ) from exc
        q = transform.rotation
        x, y, z, w = q.x, q.y, q.z, q.w
        rotation = np.asarray([
            [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
            [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
            [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
        ])
        # The TF tree gives the RAW camera's pose in the base frame; the depth
        # map lives in the RECTIFIED frame, so compose with R1^-1 -- the same
        # inversion the hand-eye-file path applies.
        fake = type("_Extrinsic", (), {
            "T_base_camera": np.eye(4),
        })()
        matrix = np.eye(4)
        matrix[:3, :3] = rotation
        matrix[:3, 3] = [
            transform.translation.x,
            transform.translation.y,
            transform.translation.z,
        ]
        fake = type("_Extrinsic", (), {"T_base_camera": matrix})()
        return _rectified_base(fake, processor.rectification_rotation_left)


def _load_client(config_path: Path, backend: str | None):
    from core.vlm.vlm_client import VLMClient

    load_secrets_env()
    # Only VLM values are needed here; reading raw YAML avoids unrelated hardware
    # layers such as a site-specific arm calibration.
    import yaml

    with config_path.open("r", encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    vlm_cfg = resolve_vlm_config(cfg, backend=backend)
    return VLMClient(
        base_url=vlm_cfg["base_url"],
        model=vlm_cfg["model"],
        api_key=vlm_cfg.get("api_key", "EMPTY"),
        timeout_s=float(vlm_cfg.get("timeout_s", 120)),
        max_tokens=int(vlm_cfg.get("max_tokens", 256)),
        temperature=float(vlm_cfg.get("temperature", 0.0)),
        chat_template_kwargs=vlm_cfg.get("chat_template_kwargs", {}),
        provider=vlm_cfg.get("provider", "vllm"),
        api_dialect=vlm_cfg.get("api_dialect"),
        reasoning_effort=vlm_cfg.get("reasoning_effort"),
        max_retries=vlm_cfg.get("max_retries"),
        api_key_refresh=make_api_key_refresher(vlm_cfg),
    )


def _parse_detection(client, frame: np.ndarray, label: str, min_confidence: float):
    prompt = (
        "Find the single visible object matching this target: " + json.dumps(label) + ".\n"
        "Return the tight visible bounding box as normalized [x_min,y_min,x_max,y_max] "
        "coordinates from 0 to 1, measured on the full input image. If the target is "
        "not clearly visible, use an empty label, bbox [0,0,0,0], and confidence 0. "
        "Return only the required JSON object. Do not estimate metric depth or propose "
        "robot motion."
    )
    response = client.complete_json(
        prompt,
        frame,
        schema=SCHEMA,
        max_tokens=128,
        temperature=0.0,
        agentview_label=None,
        wrist_label=None,
    )
    payload = response.payload["json"]
    if isinstance(payload, dict) and payload.get("label") == "":
        raise ValueError("VLM reported that the target was not visible")
    detection = VLMDetection.from_json(payload)
    if not detection.label:
        raise ValueError("VLM reported that the target was not visible")
    if detection.confidence < min_confidence:
        raise ValueError(
            f"target confidence {detection.confidence:.3f} is below "
            f"the configured {min_confidence:.3f} threshold"
        )
    return detection, response.payload.get("latency_s")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topic", default="/g1d/cameras/head_left/image_raw")
    parser.add_argument("--target", required=True, help="object name/description for VLM")
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/robot_g1d.yaml",
        help="robot config carrying the vlm block AND the calibration block",
    )
    parser.add_argument("--backend", default=None, help="configured VLM backend name")
    parser.add_argument(
        "--rate-hz", type=float, default=0.2,
        help="maximum VLM request frequency; default 1 call per 5 seconds",
    )
    parser.add_argument("--max-image-age-s", type=float, default=0.5)
    parser.add_argument("--right-topic", default="/g1d/cameras/head_right/image_raw")
    parser.add_argument("--left-info-topic", default="/g1d/cameras/head_left/camera_info")
    parser.add_argument("--right-info-topic", default="/g1d/cameras/head_right/camera_info")
    parser.add_argument(
        "--stereo-calibration", type=Path, default=None,
        help="measured stereo YAML; overrides the config's calibration.stereo_yaml",
    )
    parser.add_argument(
        "--handeye-calibration", type=Path, default=None,
        help="eye-to-hand result file (or directory); overrides the config's "
        "calibration.handeye_yaml",
    )
    parser.add_argument("--max-stereo-skew-s", type=float, default=None)
    parser.add_argument("--min-depth-m", type=float, default=None)
    parser.add_argument("--max-depth-m", type=float, default=None)
    parser.add_argument(
        "--base-frame", default="",
        help="robot frame for point_base_m; defaults to the hand-eye base frame "
        "(torso_link) with the FILE extrinsic -- no live TF required. Pass "
        "--use-live-tf to read the TF tree instead (must be TF-published).",
    )
    parser.add_argument(
        "--use-live-tf", action="store_true",
        help="read the camera TF from the live TF tree instead of the hand-eye file",
    )
    parser.add_argument("--min-confidence", type=float, default=0.7)
    parser.add_argument(
        "--count", type=int, default=1,
        help="number of calls; use 0 to continue until Ctrl+C",
    )
    args = parser.parse_args(argv)
    if not args.target.strip():
        parser.error("--target must not be empty")
    if not np.isfinite(args.rate_hz) or not 0 < args.rate_hz <= 2.0:
        parser.error("--rate-hz must be in (0, 2]; continuous VLM requests are rate limited")
    if not np.isfinite(args.max_image_age_s) or args.max_image_age_s <= 0:
        parser.error("--max-image-age-s must be finite and positive")
    for numeric, name, bounds in (
        (args.max_stereo_skew_s, "--max-stereo-skew-s", (0, float("inf"))),
        (args.min_depth_m, "--min-depth-m", (0, float("inf"))),
        (args.max_depth_m, "--max-depth-m", (0, float("inf"))),
    ):
        if numeric is not None and (
            not np.isfinite(numeric) or not bounds[0] < numeric < bounds[1]
        ):
            parser.error(f"{name} must be finite and positive when given")
    if not np.isfinite(args.min_confidence) or not 0 <= args.min_confidence <= 1:
        parser.error("--min-confidence must be in [0, 1]")
    if args.count < 0:
        parser.error("--count cannot be negative")
    if not args.config.is_file():
        parser.error(f"VLM config file does not exist: {args.config}")
    if args.stereo_calibration is not None and not args.stereo_calibration.is_file():
        parser.error(f"stereo calibration file does not exist: {args.stereo_calibration}")
    if args.handeye_calibration is not None and not args.handeye_calibration.exists():
        parser.error(f"hand-eye calibration does not exist: {args.handeye_calibration}")

    # Depth comes from the config's calibration block by default; explicit
    # --stereo-calibration / --handeye-calibration override it. No depth at
    # all when the block is absent -- 2D detections still work.
    bundle: CalibrationBundle | None = None
    if args.stereo_calibration is not None or args.handeye_calibration is not None:
        from core.config import load_yaml as _load_yaml

        try:
            cfg = _load_yaml(str(args.config))
        except Exception as exc:
            parser.error(f"could not load robot config {args.config}: {exc}")
        cfg_block = dict(cfg.get("calibration") or {})
        if args.stereo_calibration is not None:
            cfg_block["stereo_yaml"] = str(args.stereo_calibration)
        if args.handeye_calibration is not None:
            cfg_block["handeye_yaml"] = str(args.handeye_calibration)
        if args.max_stereo_skew_s is not None:
            cfg_block["max_stereo_skew_s"] = float(args.max_stereo_skew_s)
        if args.min_depth_m is not None:
            cfg_block["min_depth_m"] = float(args.min_depth_m)
        if args.max_depth_m is not None:
            cfg_block["max_depth_m"] = float(args.max_depth_m)
        cfg = {**cfg, "calibration": cfg_block}
        try:
            bundle = load_calibration(cfg)
        except CalibrationError as exc:
            parser.error(f"camera calibration: {exc}")
    else:
        from core.config import load_yaml as _load_yaml

        try:
            cfg = _load_yaml(str(args.config))
        except Exception as exc:
            parser.error(f"could not load robot config {args.config}: {exc}")
        if isinstance(cfg.get("calibration"), dict) and (
            cfg["calibration"].get("stereo_yaml") or cfg["calibration"].get("handeye_yaml")
        ):
            try:
                bundle_cfg = dict(cfg)
                if args.max_stereo_skew_s is not None:
                    bundle_cfg["calibration"] = {
                        **bundle_cfg.get("calibration", {}),
                        "max_stereo_skew_s": float(args.max_stereo_skew_s),
                    }
                if args.min_depth_m is not None:
                    bundle_cfg["calibration"] = {
                        **bundle_cfg.get("calibration", {}),
                        "min_depth_m": float(args.min_depth_m),
                    }
                if args.max_depth_m is not None:
                    bundle_cfg["calibration"] = {
                        **bundle_cfg.get("calibration", {}),
                        "max_depth_m": float(args.max_depth_m),
                    }
                bundle = load_calibration(bundle_cfg)
            except CalibrationError as exc:
                parser.error(f"camera calibration: {exc}")
        elif args.use_live_tf or args.base_frame:
            parser.error(
                "--base-frame / --use-live-tf needs the config's calibration block "
                "(or --stereo-calibration + --handeye-calibration)"
            )

    try:
        client = _load_client(args.config, args.backend)
    except Exception as exc:
        parser.error(f"could not initialize VLM client from {args.config}: {exc}")
    from core.g1d.ros_environment import require_g1d_ros_environment

    require_g1d_ros_environment()
    rclpy.init(args=None)
    processor = None
    stereo = None
    if bundle is not None:
        stereo = bundle.stereo
        processor = StereoDepthProcessor(
            stereo, min_depth_m=bundle.min_depth_m, max_depth_m=bundle.max_depth_m
        )
        node = LatestStereoSubscriber(
            args.topic,
            args.right_topic,
            args.left_info_topic,
            args.right_info_topic,
            max_age_s=args.max_image_age_s,
            max_skew_s=bundle.max_stereo_skew_s,
        )
    else:
        node = LatestRgbSubscriber(args.topic, args.max_image_age_s)
    calls = 0
    interval = 1.0 / args.rate_hz
    next_call = time.monotonic()
    try:
        base_frame = args.base_frame or (bundle.base_frame if bundle else "")
        if base_frame and bundle is not None:
            T_file_rectified = rectified_base_transform(
                bundle.extrinsic, processor.rectification_rotation_left
            )
        elif base_frame and not args.use_live_tf:
            raise RuntimeError(
                "--base-frame without the file calibration needs --use-live-tf"
            )
        if bundle is not None:
            print(f"calibration: {bundle.describe()}")
        if args.use_live_tf and not base_frame:
            print(
                "warning: --use-live-tf without a --base-frame does nothing; "
                "the 3D point stays in the rectified optical frame",
                file=sys.stderr,
            )
        print(
            f"RGB topic: {args.topic}; target: {args.target!r}; "
            f"VLM max rate: {args.rate_hz:g} Hz"
        )
        print("Perception only: no depth conversion or robot command is performed.")
        while rclpy.ok() and (args.count == 0 or calls < args.count):
            if processor is not None:
                left, right, _stamp_s = node.read_fresh_pair()
                node.validate_camera_info(stereo)
                frame, depth_m, _ = processor.compute(left, right)
            else:
                frame = node.read_fresh()
                depth_m = None
            detection, latency_s = _parse_detection(
                client, frame, args.target, args.min_confidence
            )
            result = {
                "label": detection.label,
                "bbox_norm": detection.bbox_norm,
                "confidence": detection.confidence,
                "image_size": [int(frame.shape[1]), int(frame.shape[0])],
                "vlm_latency_s": latency_s,
                "action_sent": False,
            }
            if depth_m is not None:
                fx, fy, cx, cy = processor.rectified_intrinsics
                estimate = estimate_box_depth(depth_m, detection.bbox_norm)
                point_optical = pixel_to_camera_xyz(
                    *estimate.pixel_xy,
                    estimate.median_m,
                    fx=fx,
                    fy=fy,
                    cx=cx,
                    cy=cy,
                )
                result["depth_m"] = estimate.median_m
                result["depth_mad_m"] = estimate.mad_m
                result["valid_depth_pixels"] = estimate.valid_pixels
                result["point_rectified_left_optical_m"] = point_optical.tolist()
                if base_frame:
                    if args.use_live_tf:
                        T_base_rectified = node.base_transform_for_rectified_left(
                            base_frame, processor
                        )
                        result["tf_source"] = "live_tf"
                    else:
                        T_base_rectified = T_file_rectified
                        result["tf_source"] = "handeye_file"
                        result["handeye_sha256"] = bundle.extrinsic.intrinsics_sha256
                    point_base = T_base_rectified[:3, :3] @ point_optical + T_base_rectified[:3, 3]
                    result["base_frame"] = base_frame
                    result["point_base_m"] = point_base.tolist()
            calls += 1
            print(json.dumps(result, ensure_ascii=False))
            next_call += interval
            delay = next_call - time.monotonic()
            if delay > 0 and (args.count == 0 or calls < args.count):
                deadline = time.monotonic() + delay
                while rclpy.ok() and time.monotonic() < deadline:
                    rclpy.spin_once(node, timeout_sec=min(0.1, deadline - time.monotonic()))
        return 0
    except (RuntimeError, ValueError, KeyboardInterrupt) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
