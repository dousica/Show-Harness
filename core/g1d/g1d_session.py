"""A Show-Harness session for the Unitree G1-D.

The session owns ONE arm's connection to the G1-D ROS 2 stack plus the camera
topics, and exposes the SAME observation contract the harness's runners already
speak (:class:`core.franka.franka_session.FrankaSession`,
:class:`core.piper.piper_session.PiperSession`):

    ``get_observation()`` -> ``{"agentview", "wrist", "ee_pose", "gripper_width"}``

so ``scripts/run_real.py`` can drive a G1-D arm without the runner knowing
anything G1-D-specific.

Two things are different from the Franka/Piper sessions, and both are deliberate:

* **The cameras are the head cameras, not eye-in-hand cameras.** The G1-D's
  head carries a left and a right camera. The "wrist" slot is filled with the
  wrist camera of the SELECTED side when it exists, so the VLM's wrist-view
  prompt still means something; when that topic is absent the session falls
  back to the head's other eye and says so in the log, because silently
  showing the wrong view is worse than showing a labelled substitute.
* **There is no impedance controller.** The G1-D is driven through the SDK
  bridge's trajectory Action, so there is nothing to "start" on connect.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from core.g1d.camera_source import G1DCameraSource
from core.g1d.g1d_interface import G1DRobotInterface, G1DTransport, MockG1DTransport
from core.g1d.safety import ZFloor
from core.g1d.single_arm import SingleArmProfile

#: The nominal fully-open Dex1 jaw opening (meters) -- PROVISIONAL, see
#: ``core.g1d.geometry.DEX1_NOMINAL_OPENING_M``.
DEFAULT_OPEN_WIDTH_M = 0.09


@dataclass
class G1DSessionConfig:
    """Connection + camera settings for a G1-D single-arm session."""

    side: str = "left"
    namespace: str = "lzh"
    use_mock_robot: bool = False
    use_mock_cameras: bool = False
    open_width_m: float = DEFAULT_OPEN_WIDTH_M

    # The head cameras (both are published by lzh_camera_bridge).
    head_camera_topic: str = "/g1d/cameras/head_left/image_raw"
    head_right_camera_topic: str = "/g1d/cameras/head_right/image_raw"
    # The wrist camera of the SELECTED side (falls back to the other head eye).
    wrist_camera_topic: str = "/g1d/cameras/left_wrist/image_raw"

    camera_max_age_s: float = 0.5
    camera_connect_timeout_s: float = 10.0
    observation_resolution: Optional[int] = 256

    # The minimum end-effector height (torso_link Z, meters). None -> no floor,
    # which a LIVE run must not use (core/g1d/safety.py).
    z_floor: Optional[ZFloor] = None
    # When True the transport plans and validates trajectories but never sends
    # them. The default for bring-up.
    dry_run: bool = True

    # When True a GOAL_TOLERANCE_VIOLATED abort raises (the old behaviour).
    # The default is False: the bridge keeps HOLDING the commanded final
    # position, so the Cartesian goal was reached -- a closed-loop rollout
    # must not die on it. The event is still logged and recorded.
    tolerance_violation_is_fatal: bool = False

    verbose: bool = True


class G1DSession:
    """Owns the arm + camera handles for a G1-D atomic-control session."""

    def __init__(self, config: Optional[G1DSessionConfig] = None) -> None:
        self.config = config or G1DSessionConfig()
        self.profile = SingleArmProfile(self.config.side, self.config.namespace)
        self.robot: Any = None
        self._agentview: Optional[G1DCameraSource] = None
        self._wrist: Optional[G1DCameraSource] = None
        self._wrist_is_fallback = False
        self._connected = False

    # -- lifecycle ---------------------------------------------------------
    def connect(self) -> "G1DSession":
        self._connect_robot()
        self._connect_cameras()
        self._connected = True
        return self

    def _log(self, msg: str) -> None:
        if self.config.verbose:
            print(f"[g1d-session] {msg}")

    def _connect_robot(self) -> None:
        cfg = self.config
        if cfg.use_mock_robot:
            # A deterministic in-memory transport: no ROS, no SDK, no network.
            transport: G1DTransport = MockG1DTransport(
                initial_pose=(0.30, 0.20, 0.40, 0.0, 0.0, 0.0, 1.0),
                open_width_m=cfg.open_width_m,
                closed_width_m=0.005,
            )
            self._log(f"MOCK G1-D arm ({cfg.side})")
        else:
            from core.g1d.ros_transport import G1DRosTransport

            transport = G1DRosTransport(
                self.profile,
                dry_run=cfg.dry_run,
                z_floor=cfg.z_floor,
                tolerance_violation_is_fatal=cfg.tolerance_violation_is_fatal,
                verbose=cfg.verbose,
            )
            self._log(
                f"Connecting to the G1-D {cfg.side} arm over ROS 2 "
                f"(action: {self.profile.trajectory_action})"
            )
        self.robot = G1DRobotInterface(transport).connect()
        pose = np.asarray(self.robot.get_ee_pose(), dtype=float)
        self._log(
            f"ee_pose={np.round(pose, 4).tolist()} "
            f"gripper={float(self.robot.get_gripper_position()[0]) * 1000:.1f}mm"
        )
        if cfg.z_floor is not None:
            self._log(cfg.z_floor.describe())

    def _connect_cameras(self) -> None:
        cfg = self.config
        if cfg.use_mock_cameras:
            self._log("MOCK cameras")
            self._agentview = _MockCamera(256, 256)
            self._wrist = _MockCamera(256, 256)
            return

        self._agentview = G1DCameraSource(
            cfg.head_camera_topic, max_age_s=cfg.camera_max_age_s
        ).start()
        # Prefer the selected side's wrist camera; fall back to the other head
        # eye rather than serving no wrist view at all.
        wrist_topic = cfg.wrist_camera_topic
        try:
            self._wrist = G1DCameraSource(
                wrist_topic, max_age_s=cfg.camera_max_age_s
            ).start()
        except Exception as exc:  # noqa: BLE001 - a missing topic is not fatal
            self._log(f"wrist camera {wrist_topic} unavailable ({exc}); using the head's other eye")
            self._wrist = G1DCameraSource(
                cfg.head_right_camera_topic, max_age_s=cfg.camera_max_age_s
            ).start()
            self._wrist_is_fallback = True

    def close(self) -> None:
        for source in (self._agentview, self._wrist):
            if source is not None:
                try:
                    source.stop()
                except Exception as exc:  # noqa: BLE001 - best-effort cleanup
                    self._log(f"camera stop error: {exc}")
        if self.robot is not None:
            try:
                self.robot.close()
            except Exception as exc:  # noqa: BLE001
                self._log(f"robot close error: {exc}")
        self._connected = False

    def __enter__(self) -> "G1DSession":
        return self.connect()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # -- observations ------------------------------------------------------
    def _read_rgb(self, source: Any, name: str) -> np.ndarray:
        frame = source.read_fresh(timeout_s=self.config.camera_connect_timeout_s)
        array = np.asarray(frame, dtype=np.uint8)
        res = self.config.observation_resolution
        if res:
            from core.franka import camera_utils

            array = camera_utils.resize_with_pad(array, res, res)
        return np.ascontiguousarray(array)

    def get_observation(self) -> dict[str, Any]:
        """Capture a fresh observation (same contract as FrankaSession)."""
        if not self._connected:
            raise RuntimeError("G1DSession.get_observation called before connect()")
        if self._agentview is None or self._wrist is None:
            raise RuntimeError("get_observation requires cameras")
        agentview = self._read_rgb(self._agentview, "agentview")
        wrist = self._read_rgb(self._wrist, "wrist")
        ee_pose = np.asarray(self.robot.get_ee_pose(), dtype=float)
        gripper_width = float(self.robot.get_gripper_position()[0])
        return {
            "agentview": agentview,
            "wrist": wrist,
            "ee_pose": ee_pose,
            "gripper_width": gripper_width,
        }

    def get_camera_frames(self) -> Optional[dict[str, np.ndarray]]:
        """Latest frames for the live-display poller (never raises)."""
        if not self._connected or self._agentview is None or self._wrist is None:
            return None
        try:
            return {
                "agentview": self._read_rgb(self._agentview, "agentview"),
                "wrist": self._read_rgb(self._wrist, "wrist"),
            }
        except Exception:  # noqa: BLE001 - a cosmetic display must never crash a rollout
            return None

    @property
    def wrist_is_fallback(self) -> bool:
        """True when the "wrist" view is actually the head's other eye."""
        return self._wrist_is_fallback


class _MockCamera:
    """A deterministic offline stand-in for a camera source."""

    def __init__(self, width: int, height: int) -> None:
        self.width = int(width)
        self.height = int(height)
        self._frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self._frame[:, :, 0] = 40  # a flat, obviously-fake image

    def read_fresh(self, timeout_s: float = 5.0) -> np.ndarray:
        return self._frame.copy()

    def stop(self) -> None:
        return None
