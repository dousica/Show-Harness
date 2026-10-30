"""Transport boundary between a G1-D bridge and Show-Harness.

This module deliberately has no ROS, SDK, or hardware imports. A future bridge client
can implement :class:`G1DTransport`; :class:`G1DRobotInterface` validates and maps that
transport to the duck-typed API consumed by ``RealAtomicController``.
"""
from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np


class G1DTransport(Protocol):
    """Normalized operations a G1-D transport must provide.

    Pose units and ordering are meters and ``[x, y, z, qx, qy, qz, qw]`` in the robot
    base frame. The transport is responsible for mapping those values to its ROS 2 / SDK
    messages and for reporting measured feedback rather than commanded targets.
    """

    def connect(self) -> None: ...

    def close(self) -> None: ...

    def read_end_effector_pose(self) -> Sequence[float]: ...

    def read_gripper_width_m(self) -> float: ...

    def command_end_effector_pose(self, pose7: Sequence[float]) -> None: ...

    def command_gripper(self, closed: bool) -> None: ...


def _pose7(value: Sequence[float], *, name: str) -> np.ndarray:
    pose = np.array(value, dtype=np.float64, copy=True).reshape(-1)
    if pose.shape != (7,):
        raise ValueError(f"{name} must contain 7 values [x,y,z,qx,qy,qz,qw], got {pose.shape}")
    if not np.all(np.isfinite(pose)):
        raise ValueError(f"{name} must contain only finite values")
    quat_norm = float(np.linalg.norm(pose[3:]))
    if quat_norm < 1e-8:
        raise ValueError(f"{name} quaternion must have non-zero length")
    pose[3:] /= quat_norm
    return pose


class G1DRobotInterface:
    """Adapt a G1-D transport to the interface expected by real-arm controllers."""

    def __init__(self, transport: G1DTransport) -> None:
        self.transport = transport
        self._connected = False

    def connect(self) -> "G1DRobotInterface":
        if not self._connected:
            self.transport.connect()
            self._connected = True
        return self

    def close(self) -> None:
        if self._connected:
            try:
                self.transport.close()
            finally:
                self._connected = False

    def _require_connected(self, operation: str) -> None:
        if not self._connected:
            raise RuntimeError(f"G1DRobotInterface.{operation} called before connect()")

    def get_ee_pose(self) -> np.ndarray:
        self._require_connected("get_ee_pose")
        return _pose7(self.transport.read_end_effector_pose(), name="measured pose")

    def get_gripper_position(self) -> np.ndarray:
        self._require_connected("get_gripper_position")
        width = float(self.transport.read_gripper_width_m())
        if not np.isfinite(width) or width < 0.0:
            raise ValueError(f"measured gripper width must be finite and non-negative, got {width}")
        return np.asarray([width], dtype=np.float64)

    def control_gripper(self, close: bool) -> None:
        self._require_connected("control_gripper")
        self.transport.command_gripper(bool(close))

    def update_desired_ee_pose(self, pose: Sequence[float]) -> None:
        self._require_connected("update_desired_ee_pose")
        normalized = _pose7(pose, name="command pose")
        self.transport.command_end_effector_pose(normalized.tolist())


class MockG1DTransport:
    """Deterministic in-memory transport for offline interface/controller tests.

    This is not a dynamics simulator and sends no ROS, SDK, or network commands. Initial
    pose and gripper widths are required so the mock cannot imply G1-D calibration values.
    """

    def __init__(
        self,
        initial_pose: Sequence[float],
        open_width_m: float,
        closed_width_m: float,
    ) -> None:
        self._pose = _pose7(initial_pose, name="initial pose")
        self.open_width_m = self._validate_width(open_width_m, name="open_width_m")
        self.closed_width_m = self._validate_width(closed_width_m, name="closed_width_m")
        if self.closed_width_m >= self.open_width_m:
            raise ValueError("closed_width_m must be less than open_width_m")
        self._gripper_width_m = self.open_width_m
        self._connected = False
        self.pose_commands: list[np.ndarray] = []
        self.gripper_commands: list[bool] = []

    @staticmethod
    def _validate_width(value: float, *, name: str) -> float:
        width = float(value)
        if not np.isfinite(width) or width < 0.0:
            raise ValueError(f"{name} must be finite and non-negative, got {width}")
        return width

    def connect(self) -> None:
        self._connected = True

    def close(self) -> None:
        self._connected = False

    def _require_connected(self) -> None:
        if not self._connected:
            raise RuntimeError("MockG1DTransport used before connect()")

    def read_end_effector_pose(self) -> np.ndarray:
        self._require_connected()
        return self._pose.copy()

    def read_gripper_width_m(self) -> float:
        self._require_connected()
        return self._gripper_width_m

    def command_end_effector_pose(self, pose7: Sequence[float]) -> None:
        self._require_connected()
        self._pose = _pose7(pose7, name="command pose")
        self.pose_commands.append(self._pose.copy())

    def command_gripper(self, closed: bool) -> None:
        self._require_connected()
        closed = bool(closed)
        self.gripper_commands.append(closed)
        self._gripper_width_m = self.closed_width_m if closed else self.open_width_m
