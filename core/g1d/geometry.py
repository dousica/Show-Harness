"""Pure geometry / limit helpers for the G1-D embodiment.

Everything here is ROS-free and hardware-free on purpose: these are the rules the
harness must not get wrong (which frame a translation lives in, how far the
selected arm may reach, how fast it may move), and they must be checkable on a
laptop without the robot, the SDK bridge, or a running MoveIt.

The two authorities for the numbers are:

* the **official G1-D URDF** (``g1_d_ros2.urdf``) for the joint limits, and
* the **SDK bridge** (``g1d_trajectory_action_server.cpp``) for the commissioning
  guardrails it enforces on every trajectory it accepts.

Both agree on the joint limits (verified: URDF limits == the bridge's
``LEFT_ARM_MIN``/``LEFT_ARM_MAX`` tables), so a trajectory that satisfies the
URDF also satisfies the bridge's limit check.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

#: The G1-D URDF's fixed (non-arm) frames. ``torso_link`` is the base both arms
#: are planned against -- it is the SRDF chain base for ``left_arm``/``right_arm``.
TORSO_FRAME = "torso_link"

#: The MoveIt planning groups in ``lzh_moveit_config/config/G1_D.srdf``.
LEFT_GROUP = "left_arm"
RIGHT_GROUP = "right_arm"

#: Per-joint limits, in the URDF's own joint order (shoulder pitch, roll, yaw,
#: elbow, wrist roll, pitch, yaw). Copied from ``g1_d_ros2.urdf``; the SDK bridge
#: enforces the identical numbers, so a trajectory inside these bounds is inside
#: the bridge's limit check too.
LEFT_ARM_LIMITS_RAD: tuple[tuple[float, float], ...] = (
    (-3.0892, 2.6704),
    (-1.5882, 2.2515),
    (-2.6180, 2.6180),
    (-1.0472, 2.0944),
    (-1.972222054, 1.972222054),
    (-1.614429558, 1.614429558),
    (-1.614429558, 1.614429558),
)
RIGHT_ARM_LIMITS_RAD: tuple[tuple[float, float], ...] = (
    (-3.0892, 2.6704),
    (-2.2515, 1.5882),
    (-2.6180, 2.6180),
    (-1.0472, 2.0944),
    (-1.972222054, 1.972222054),
    (-1.614429558, 1.614429558),
    (-1.614429558, 1.614429558),
)

#: The bridge's commissioning speed guard (``arm_max_trajectory_speed_rad_s``,
#: default 0.5 rad/s). It is a commissioning bound, NOT Unitree's motor rating,
#: and it is checked on EVERY segment of an accepted trajectory -- so the harness
#: must plan within it or the bridge aborts the goal before writing any LowCmd.
BRIDGE_MAX_TRAJECTORY_SPEED_RAD_S = 0.5

#: The bridge's start-state guard (``max_start_error_rad``, default 0.03 rad):
#: the trajectory's first point must match the measured joints.
BRIDGE_MAX_START_ERROR_RAD = 0.03

#: The bridge's goal tolerance (``goal_tolerance_rad``, default 0.03 rad).
BRIDGE_GOAL_TOLERANCE_RAD = 0.03

#: Dex1-1 gripper: the bridge's nominal motor stroke and the opening it maps to.
#: This mapping is PROVISIONAL (the bridge says so in its own source); treat it as
#: a nominal range, never as a calibrated jaw width.
DEX1_MOTOR_STROKE_RAD = 5.4
DEX1_NOMINAL_OPENING_M = 0.09
#: The bridge rejects a gripper goal above ``gripper_max_opening_m`` (0.09).
DEX1_MAX_OPENING_M = 0.09


def arm_limits_rad(side: str) -> tuple[tuple[float, float], ...]:
    """The 7 per-joint (lower, upper) bounds for ``side``, in URDF joint order."""
    if side == "left":
        return LEFT_ARM_LIMITS_RAD
    if side == "right":
        return RIGHT_ARM_LIMITS_RAD
    raise ValueError(f"side must be 'left' or 'right', got {side!r}")


def joint_position_limits_error(
    positions: Sequence[float],
    limits: Sequence[tuple[float, float]],
    *,
    tolerance_rad: float = 0.0,
) -> str | None:
    """Return a human-readable reason if ``positions`` violate ``limits``.

    ``None`` means every joint is inside its bounds (within ``tolerance_rad``).
    The tolerance mirrors the bridge's own screening of *measured* feedback
    (0.03 rad); a *commanded* target must be inside the exact bounds.
    """
    values = np.asarray(positions, dtype=float).reshape(-1)
    if values.size != len(limits):
        return f"expected {len(limits)} joint positions, got {values.size}"
    if not np.all(np.isfinite(values)):
        return "joint positions must be finite"
    for index, (value, (lower, upper)) in enumerate(zip(values, limits)):
        if value < lower - tolerance_rad or value > upper + tolerance_rad:
            return (
                f"joint {index} at {value:.4f} rad is outside its G1-D limit "
                f"[{lower:.4f}, {upper:.4f}] rad"
            )
    return None


def clamp_to_arm_limits(
    positions: Sequence[float],
    limits: Sequence[tuple[float, float]],
) -> np.ndarray:
    """Clamp joint positions into the G1-D bounds (used only for a *published*
    diagnostic, never to silently alter a commanded target)."""
    values = np.asarray(positions, dtype=float).reshape(-1)
    if values.size != len(limits):
        raise ValueError(f"expected {len(limits)} joint positions, got {values.size}")
    clamped = values.copy()
    for index, (lower, upper) in enumerate(limits):
        clamped[index] = min(max(float(clamped[index]), float(lower)), float(upper))
    return clamped


def trajectory_speed_ok(
    times_s: Sequence[float],
    positions: Sequence[Sequence[float]],
    *,
    max_speed_rad_s: float = BRIDGE_MAX_TRAJECTORY_SPEED_RAD_S,
) -> str | None:
    """Check the bridge's per-segment speed guard on a sampled trajectory.

    The bridge samples each segment at ~20 ms and compares consecutive samples
    with ``|dq| / dt > max_speed_rad_s``. This reproduces that check on the
    samples we have, so a plan that passes here will not be aborted by the
    bridge for speed.

    Returns ``None`` when the trajectory is acceptable, else the reason.
    """
    stamps = np.asarray(times_s, dtype=float).reshape(-1)
    points = [np.asarray(p, dtype=float).reshape(-1) for p in positions]
    if stamps.size != len(points):
        return "times and positions must have the same length"
    if stamps.size < 2:
        return None
    if not np.all(np.isfinite(stamps)) or np.any(np.diff(stamps) <= 0):
        return "trajectory timestamps must be finite and strictly increasing"
    for index in range(1, len(points)):
        dt = float(stamps[index] - stamps[index - 1])
        if dt <= 0:
            return "trajectory timestamps must be strictly increasing"
        speeds = np.abs(points[index] - points[index - 1]) / dt
        if speeds.size and float(np.max(speeds)) > max_speed_rad_s + 1e-6:
            worst = int(np.argmax(speeds))
            return (
                f"trajectory exceeds the {max_speed_rad_s:g} rad/s command-speed guard "
                f"(joint {worst} at {float(speeds[worst]):.3f} rad/s)"
            )
    return None


def dex1_opening_to_motor(opening_m: float) -> float:
    """Nominal opening (m) -> Dex1 motor position (rad), mirroring the bridge."""
    return float(opening_m) * (DEX1_MOTOR_STROKE_RAD / DEX1_NOMINAL_OPENING_M)


def dex1_motor_to_opening(motor_rad: float) -> float:
    """Dex1 motor position (rad) -> nominal opening (m), mirroring the bridge."""
    return float(motor_rad) * (DEX1_NOMINAL_OPENING_M / DEX1_MOTOR_STROKE_RAD)


@dataclass(frozen=True)
class CartesianTarget:
    """An end-effector target in the ``torso_link`` frame.

    ``pose7`` is ``[x, y, z, qx, qy, qz, qw]`` (meters, xyzw) -- the same
    convention the rest of the harness uses for ``get_ee_pose`` /
    ``update_desired_ee_pose``.
    """

    pose7: tuple[float, ...]

    def __post_init__(self) -> None:
        pose = np.asarray(self.pose7, dtype=float).reshape(-1)
        if pose.size != 7:
            raise ValueError(f"pose7 must hold 7 values, got {pose.size}")
        if not np.all(np.isfinite(pose)):
            raise ValueError("pose7 must be finite")
        if float(np.linalg.norm(pose[3:])) < 1e-8:
            raise ValueError("pose7 quaternion must be non-zero")
        object.__setattr__(self, "pose7", tuple(float(v) for v in pose))

    @property
    def position(self) -> np.ndarray:
        return np.asarray(self.pose7[:3], dtype=float)

    @property
    def quaternion(self) -> np.ndarray:
        return np.asarray(self.pose7[3:], dtype=float)

    def translated(self, delta_xyz: Sequence[float]) -> "CartesianTarget":
        """A copy translated by ``delta_xyz`` (meters, torso_link frame)."""
        delta = np.asarray(delta_xyz, dtype=float).reshape(-1)
        if delta.size != 3 or not np.all(np.isfinite(delta)):
            raise ValueError("delta_xyz must be three finite values")
        moved = self.position + delta
        return CartesianTarget(tuple(moved.tolist()) + tuple(self.pose7[3:]))


def translation_distance(a: Sequence[float], b: Sequence[float]) -> float:
    """Euclidean distance between two 3-vectors."""
    va = np.asarray(a, dtype=float).reshape(-1)
    vb = np.asarray(b, dtype=float).reshape(-1)
    if va.size != 3 or vb.size != 3:
        raise ValueError("both points must be 3-vectors")
    return float(np.linalg.norm(va - vb))


def rotation_distance_rad(a_xyzw: Sequence[float], b_xyzw: Sequence[float]) -> float:
    """Angle (rad) between two unit quaternions (xyzw)."""
    qa = np.asarray(a_xyzw, dtype=float).reshape(-1)
    qb = np.asarray(b_xyzw, dtype=float).reshape(-1)
    if qa.size != 4 or qb.size != 4:
        raise ValueError("both quaternions must be 4-vectors")
    na = float(np.linalg.norm(qa))
    nb = float(np.linalg.norm(qb))
    if na < 1e-8 or nb < 1e-8:
        raise ValueError("quaternions must be non-zero")
    dot = abs(float(np.dot(qa / na, qb / nb)))
    return 2.0 * math.acos(max(-1.0, min(1.0, dot)))
