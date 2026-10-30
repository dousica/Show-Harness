"""Read fresh RGB frames from the LZH G1-D camera bridge.

The bridge (``lzh_camera_bridge``) publishes ``bgr8`` images on
``/g1d/cameras/<name>/image_raw``. This module turns that topic into RGB frames
the vision API can consume, with two properties the raw topic does not give you:

* **Colour correctness.** The bridge's images are BGR; the VLM expects RGB.
  Swapping the channels here means every caller gets RGB and no caller has to
  remember the convention.
* **Freshness.** The bridge republishes at a fixed rate, and a camera that has
  stopped can leave the last frame sitting in the subscription buffer. Every
  read here checks the frame's age, so a stale image is an error rather than a
  silent detection on a frozen picture.

The ROS import is LAZY: importing this module on a machine without ROS 2 (the
offline test suite, a laptop) works fine. Live subscription requires Domain 42
and the ``/home/yushu/g1d_ros2_lzh/setup_env.sh`` workspace overlay.
"""

from __future__ import annotations

import time
from typing import Any, Optional, Sequence

import numpy as np

__all__ = ["G1DCameraSource", "decode_image_message"]


def decode_image_message(msg: Any) -> np.ndarray:
    """Decode a ``sensor_msgs/Image`` into an RGB ``uint8`` HWC array.

    ``bgr8`` (what the G1-D bridge publishes) is converted to RGB; ``rgb8`` is
    taken as is. Any other encoding is rejected rather than silently
    misinterpreted -- a mis-read channel order produces a plausible-looking but
    wrong detection, which is far worse than a loud failure.
    """
    encoding = str(getattr(msg, "encoding", "")).lower()
    if encoding not in ("bgr8", "rgb8"):
        raise ValueError(
            f"expected a bgr8/rgb8 image, received {getattr(msg, 'encoding', None)!r}"
        )
    height = int(msg.height)
    width = int(msg.width)
    step = int(getattr(msg, "step", width * 3))
    raw = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    frame = raw.reshape(height, step)[:, : width * 3].reshape(height, width, 3)
    if encoding == "bgr8":
        frame = frame[:, :, ::-1]
    return np.ascontiguousarray(frame)


class G1DCameraSource:
    """Subscribe to one camera topic and hand out fresh RGB frames.

    Usage::

        with G1DCameraSource("/g1d/cameras/head_left/image_raw") as source:
            frame = source.read_fresh(timeout_s=5.0)

    The node is created lazily on first use so that constructing the object --
    which the config layer does -- never requires ROS to be importable.
    """

    def __init__(
        self,
        topic: str,
        *,
        max_age_s: float = 0.5,
        node_name: str = "harness_g1d_camera_source",
    ) -> None:
        if not str(topic).strip():
            raise ValueError("topic must not be empty")
        age = float(max_age_s)
        if not np.isfinite(age) or age <= 0:
            raise ValueError("max_age_s must be finite and positive")
        self.topic = str(topic)
        self.max_age_s = age
        self.node_name = str(node_name)
        self._node: Any = None
        self._subscription: Any = None
        self._latest: Optional[np.ndarray] = None
        self._received_at: Optional[float] = None
        self._stamp_s: Optional[float] = None
        self._frame_id: Optional[str] = None
        self._dropped = 0

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> "G1DCameraSource":
        if self._node is not None:
            return self
        from core.g1d.ros_environment import require_g1d_ros_environment

        require_g1d_ros_environment()
        try:
            import rclpy
            from rclpy.node import Node
            from rclpy.qos import qos_profile_sensor_data
            from sensor_msgs.msg import Image
        except ImportError as exc:  # pragma: no cover - environment guard
            raise ImportError(
                "rclpy / sensor_msgs are not importable. Source ROS 2 first: "
            "`export ROS_DOMAIN_ID=42` and "
            "`source /home/yushu/g1d_ros2_lzh/setup_env.sh`."
            ) from exc
        if not rclpy.ok():
            rclpy.init(args=None)
        self._node = Node(self.node_name)
        self._subscription = self._node.create_subscription(
            Image, self.topic, self._on_image, qos_profile_sensor_data
        )
        return self

    def stop(self) -> None:
        if self._node is not None:
            self._node.destroy_subscription(self._subscription)
            self._node.destroy_node()
            self._node = None
            self._subscription = None

    def __enter__(self) -> "G1DCameraSource":
        return self.start()

    def __exit__(self, *_exc: object) -> None:
        self.stop()

    # -- subscription callback ---------------------------------------------
    def _on_image(self, msg: Any) -> None:
        try:
            frame = decode_image_message(msg)
        except (ValueError, IndexError):
            self._dropped += 1
            return
        self._latest = frame
        self._received_at = time.monotonic()
        header = getattr(msg, "header", None)
        stamp = getattr(header, "stamp", None)
        if stamp is not None:
            self._stamp_s = float(stamp.sec) + float(stamp.nanosec) * 1e-9
            self._frame_id = getattr(header, "frame_id", None) or None

    # -- reads --------------------------------------------------------------
    def read_fresh(self, timeout_s: float = 5.0) -> np.ndarray:
        """Return the newest frame, or raise if none arrives within the timeout.

        The returned frame is RGB ``uint8`` HWC.
        """
        deadline = time.monotonic() + float(timeout_s)
        while True:
            if (
                self._latest is not None
                and self._received_at is not None
                and time.monotonic() - self._received_at <= self.max_age_s
            ):
                return self._latest.copy()
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"no fresh RGB frame on {self.topic} within {timeout_s:g}s; "
                    "check that lzh_camera_bridge is running and publishing"
                )
            self._spin(0.05)

    def _spin(self, timeout_sec: float) -> None:
        if self._node is None:
            time.sleep(timeout_sec)
            return
        import rclpy

        rclpy.spin_once(self._node, timeout_sec=timeout_sec)

    @property
    def frame_id(self) -> Optional[str]:
        """The ``frame_id`` of the last frame, if the message carried one."""
        return self._frame_id

    @property
    def stamp_s(self) -> Optional[float]:
        """The header timestamp of the last frame (bridge RECEIVE time)."""
        return self._stamp_s

    @property
    def dropped_frames(self) -> int:
        """How many frames were rejected as malformed."""
        return self._dropped

    @property
    def last_age_s(self) -> Optional[float]:
        """Age of the last frame in seconds, or ``None`` if none arrived yet."""
        if self._received_at is None:
            return None
        return time.monotonic() - self._received_at
