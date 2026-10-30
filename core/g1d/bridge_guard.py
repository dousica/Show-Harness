"""A faithful Python mirror of the G1-D SDK bridge's trajectory acceptance rules.

Why this module exists
----------------------
The bridge (``g1d_trajectory_action_server.cpp``) is the ONLY thing that can put a
command on the G1-D motors, and it **rejects** any trajectory that fails its guards
-- BEFORE it writes a single ``LowCmd``. If the harness sends a trajectory the bridge
will reject, the motion silently does not happen (the goal is rejected/aborted), and
an autonomous rollout would record a phantom move.

So the harness must know the bridge's rules EXACTLY, not approximately. This module
re-implements them in pure Python/NumPy, with no ROS and no hardware, so the rules
can be unit-tested on a laptop and reused by the ROS transport:

* ``validate_message``      -> :func:`validate_trajectory_structure`
* ``validate_real_state``   -> :func:`validate_trajectory_against_bridge`

Both are transcribed from the bridge's C++ (``g1d_trajectory_action_server.cpp``),
including the details that are easy to get wrong:

* The bridge samples each segment at ~20 ms and compares **consecutive samples**,
  not consecutive waypoints -- so a fast move inside one long segment is still caught.
* The bridge interpolates with **cubic Hermite** when the trajectory carries
  velocities (MoveIt usually does), and linearly otherwise. The Hermite curve can
  OVERSHOOT the waypoints, so checking only the waypoints would miss a limit
  violation. :func:`sample_trajectory` reproduces that overshoot.
* The joint-limit check uses the EXACT bounds (no tolerance) for commanded values,
  while the measured-state check allows 0.03 rad of measurement tolerance.
* The speed guard is a separate, conservative commissioning bound
  (``arm_max_trajectory_speed_rad_s``, default 0.5 rad/s) -- explicitly NOT the
  motor rating.

The numbers in :mod:`core.g1d.geometry` (the URDF joint limits) and here (the bridge's
guard bounds) were verified to be identical, so a trajectory that passes this module
also passes the bridge's own limit check.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from core.g1d.geometry import (
    BRIDGE_GOAL_TOLERANCE_RAD,
    BRIDGE_MAX_START_ERROR_RAD,
    BRIDGE_MAX_TRAJECTORY_SPEED_RAD_S,
    joint_position_limits_error,
)

#: The bridge's sampling step for the per-segment guards (seconds). The bridge uses
#: ``max(10, ceil(dt / 0.02))`` samples per segment.
BRIDGE_SAMPLE_STEP_S = 0.02
#: The minimum number of samples per segment (the bridge's floor).
BRIDGE_MIN_SAMPLES_PER_SEGMENT = 10
#: The bridge's maximum samples per segment (it aborts a "excessively long" segment).
BRIDGE_MAX_SAMPLES_PER_SEGMENT = 100000


@dataclass(frozen=True)
class TrajectoryData:
    """A plain-Python trajectory: the bridge's view of a ``FollowJointTrajectory``.

    ``joint_names`` are the 7 arm joints (in the order the bridge expects for this
    side); ``times_s`` are the ``time_from_start`` stamps in seconds; ``positions``
    and ``velocities`` are per-waypoint vectors of 7 values each. ``velocities`` may
    be empty (the bridge then interpolates linearly).
    """

    joint_names: tuple[str, ...]
    times_s: tuple[float, ...]
    positions: tuple[tuple[float, ...], ...]
    velocities: tuple[tuple[float, ...], ...] = ()

    def __post_init__(self) -> None:
        if len(self.joint_names) != len(set(self.joint_names)):
            raise ValueError("trajectory has duplicate joint names")
        if len(self.positions) != len(self.times_s):
            raise ValueError("times_s and positions must have the same length")
        if self.velocities and len(self.velocities) != len(self.times_s):
            raise ValueError("velocities must have the same length as positions")

    @property
    def duration_s(self) -> float:
        """The trajectory's total duration (the last waypoint's stamp)."""
        return float(self.times_s[-1]) if self.times_s else 0.0


def sample_trajectory(data: TrajectoryData, t: float) -> np.ndarray:
    """The bridge's trajectory sampler at time ``t`` (seconds).

    Reproduces ``g1d_trajectory_action_server.cpp``'s ``sample()`` exactly:

    * before the first waypoint -> the first waypoint's positions,
    * at/after the last waypoint -> the last waypoint's positions,
    * in between -> cubic Hermite if BOTH bracketing waypoints carry a full set of
      velocities, else linear interpolation.

    The Hermite branch is the important one: it can overshoot the waypoints, which
    is exactly why the bridge samples instead of checking the waypoints directly.
    """
    points = data.positions
    count = len(points)
    if count == 0:
        raise ValueError("cannot sample an empty trajectory")
    if count == 1 or t <= float(data.times_s[0]):
        return np.asarray(points[0], dtype=float)
    if t >= float(data.times_s[-1]):
        return np.asarray(points[-1], dtype=float)

    high = 1
    while high < count and float(data.times_s[high]) < t:
        high += 1
    low = high - 1
    t0 = float(data.times_s[low])
    t1 = float(data.times_s[high])
    dt = t1 - t0
    if dt <= 0.0:
        raise ValueError("trajectory timestamps must be strictly increasing")
    s = min(max((t - t0) / dt, 0.0), 1.0)

    q0 = np.asarray(points[low], dtype=float)
    q1 = np.asarray(points[high], dtype=float)
    have_v = bool(data.velocities) and all(
        len(v) == len(data.joint_names) for v in (data.velocities[low], data.velocities[high])
    )
    if not have_v:
        return q0 + s * (q1 - q0)

    v0 = np.asarray(data.velocities[low], dtype=float)
    v1 = np.asarray(data.velocities[high], dtype=float)
    s2 = s * s
    s3 = s2 * s
    h00 = 2 * s3 - 3 * s2 + 1
    h10 = s3 - 2 * s2 + s
    h01 = -2 * s3 + 3 * s2
    h11 = s3 - s2
    return h00 * q0 + h10 * dt * v0 + h01 * q1 + h11 * dt * v1


def validate_trajectory_structure(
    data: TrajectoryData,
    expected_joint_names: Sequence[str],
) -> str | None:
    """Mirror of the bridge's ``validate_message``.

    Returns ``None`` when the trajectory is structurally acceptable, else the reason
    the bridge would reject it.
    """
    expected = list(expected_joint_names)
    if not data.positions:
        return "trajectory has no points"
    if len(data.joint_names) != len(expected):
        return "joint_names count does not match configured controller"
    received = set(data.joint_names)
    if len(received) != len(data.joint_names):
        return "duplicate joint in joint_names"
    for name in data.joint_names:
        if name not in set(expected):
            return f"unexpected joint: {name}"
    for name in expected:
        if name not in received:
            return f"missing joint: {name}"
    for index, positions in enumerate(data.positions):
        if len(positions) != len(data.joint_names):
            return f"invalid positions length at point {index}"
        if not np.all(np.isfinite(np.asarray(positions, dtype=float))):
            return f"NaN/Inf at point {index}"
        if data.velocities:
            velocities = data.velocities[index]
            if len(velocities) != len(data.joint_names):
                return f"invalid velocities length at point {index}"
            if not np.all(np.isfinite(np.asarray(velocities, dtype=float))):
                return f"NaN/Inf at point {index}"
    previous = -1.0
    for index, stamp in enumerate(data.times_s):
        nanoseconds = int(round(float(stamp) * 1e9))
        if nanoseconds < 0 or (index > 0 and nanoseconds <= previous):
            return "time_from_start is invalid/non-increasing"
        previous = nanoseconds
    return None


def validate_trajectory_against_bridge(
    data: TrajectoryData,
    joint_limits: Sequence[tuple[float, float]],
    measured_joints: Sequence[float],
    *,
    max_speed_rad_s: float = BRIDGE_MAX_TRAJECTORY_SPEED_RAD_S,
    max_start_error_rad: float = BRIDGE_MAX_START_ERROR_RAD,
    sample_step_s: float = BRIDGE_SAMPLE_STEP_S,
) -> str | None:
    """Mirror of the bridge's ``validate_real_state``.

    ``joint_limits`` are the 7 ``(lower, upper)`` bounds for this side (the URDF's
    official G1-D bounds, which the bridge uses verbatim); ``measured_joints`` are
    the 7 measured joint positions the bridge compares the first waypoint against.

    Returns ``None`` when the bridge would ACCEPT the trajectory, else the reason it
    would abort it. The checks, in the bridge's own order:

    1. the first waypoint must match the measured joints (within
       ``max_start_error_rad``) -- otherwise the robot would JUMP at goal start;
    2. every SAMPLED position must be inside the EXACT joint bounds (no tolerance:
       a commanded value must be legal, not merely close);
    3. every SAMPLED segment must respect the command-speed guard.
    """
    if not data.positions:
        return "trajectory has no points"
    measured = np.asarray(measured_joints, dtype=float).reshape(-1)
    if measured.size != len(data.joint_names):
        return f"expected {len(data.joint_names)} measured joints, got {measured.size}"

    first = np.asarray(data.positions[0], dtype=float)
    start_error = float(np.max(np.abs(first - measured)))
    if start_error > max_start_error_rad:
        return (
            f"start-state mismatch: {start_error:.6f} rad > {max_start_error_rad} rad"
        )

    if not (math.isfinite(max_speed_rad_s) and max_speed_rad_s > 0.0):
        return "arm_max_trajectory_speed_rad_s must be finite and > 0"

    previous = sample_trajectory(data, 0.0)
    previous_t = 0.0
    for segment in range(len(data.positions)):
        t1 = float(data.times_s[segment])
        t0 = 0.0 if segment == 0 else float(data.times_s[segment - 1])
        dt = t1 - t0
        if dt < 0.0:
            return "trajectory timestamps invalid"
        samples = max(
            BRIDGE_MIN_SAMPLES_PER_SEGMENT, int(math.ceil(dt / sample_step_s))
        )
        if samples > BRIDGE_MAX_SAMPLES_PER_SEGMENT:
            return "trajectory segment excessively long"
        for step in range(samples + 1):
            t = t0 + dt * (float(step) / float(samples))
            positions = sample_trajectory(data, t)
            reason = joint_position_limits_error(positions, joint_limits)
            if reason is not None:
                return (
                    f"official G1-D URDF position limit violated at t={t:.3f}s: {reason}"
                )
            if t > previous_t + 1e-9:
                speed = np.abs(positions - previous) / (t - previous_t)
                if speed.size and float(np.max(speed)) > max_speed_rad_s + 1e-6:
                    worst = int(np.argmax(speed))
                    return (
                        f"arm command-speed guard exceeded at "
                        f"{data.joint_names[worst]}: limit {max_speed_rad_s} rad/s "
                        "(separate conservative setting, not URDF motor rating)"
                    )
                previous = positions
                previous_t = t
    return None


@dataclass
class BridgeGuardReport:
    """The outcome of checking a trajectory against the bridge's rules."""

    ok: bool
    reason: str = ""
    duration_s: float = 0.0
    points: int = 0
    max_speed_rad_s: float = 0.0
    max_start_error_rad: float = 0.0
    checks: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """A one-line, human-readable summary for the console/log."""
        if self.ok:
            return (
                f"bridge guard OK: {self.points} points, {self.duration_s:.2f}s, "
                f"peak {self.max_speed_rad_s:.3f} rad/s "
                f"(limit {BRIDGE_MAX_TRAJECTORY_SPEED_RAD_S} rad/s), "
                f"start error {self.max_start_error_rad:.4f} rad"
            )
        return f"bridge guard REJECTED: {self.reason}"


def check_trajectory(
    data: TrajectoryData,
    joint_limits: Sequence[tuple[float, float]],
    measured_joints: Sequence[float],
    expected_joint_names: Sequence[str],
    *,
    max_speed_rad_s: float = BRIDGE_MAX_TRAJECTORY_SPEED_RAD_S,
    max_start_error_rad: float = BRIDGE_MAX_START_ERROR_RAD,
) -> BridgeGuardReport:
    """Run BOTH bridge checks and report the peak speed / start error as well.

    The peak speed is measured the same way the bridge measures it (over the
    sampled points), so the printed number is the number the bridge would compare
    against its limit -- not a waypoint-to-waypoint approximation.
    """
    checks: list[str] = []
    structure = validate_trajectory_structure(data, expected_joint_names)
    checks.append("structure")
    if structure is not None:
        return BridgeGuardReport(False, structure, checks=checks)

    measured = np.asarray(measured_joints, dtype=float).reshape(-1)
    start_error = 0.0
    if measured.size == len(data.joint_names):
        start_error = float(
            np.max(np.abs(np.asarray(data.positions[0], dtype=float) - measured))
        )
    checks.append("start-state")

    peak = 0.0
    if len(data.positions) >= 2:
        previous = sample_trajectory(data, 0.0)
        previous_t = 0.0
        for segment in range(len(data.positions)):
            t1 = float(data.times_s[segment])
            t0 = 0.0 if segment == 0 else float(data.times_s[segment - 1])
            dt = t1 - t0
            samples = max(
                BRIDGE_MIN_SAMPLES_PER_SEGMENT, int(math.ceil(dt / BRIDGE_SAMPLE_STEP_S))
            )
            for step in range(samples + 1):
                t = t0 + dt * (float(step) / float(samples))
                positions = sample_trajectory(data, t)
                if t > previous_t + 1e-9:
                    peak = max(peak, float(np.max(np.abs(positions - previous) / (t - previous_t))))
                    previous = positions
                    previous_t = t

    reason = validate_trajectory_against_bridge(
        data,
        joint_limits,
        measured,
        max_speed_rad_s=max_speed_rad_s,
        max_start_error_rad=max_start_error_rad,
    )
    checks.append("limits+speed")
    return BridgeGuardReport(
        ok=reason is None,
        reason=reason or "",
        duration_s=data.duration_s,
        points=len(data.positions),
        max_speed_rad_s=peak,
        max_start_error_rad=start_error,
        checks=checks,
    )


def goal_tolerance_note(error_rad: float) -> str:
    """A human-readable note about how the measured error compares to the tolerance."""
    return (
        f"{error_rad:.4f} rad (the bridge's goal tolerance is "
        f"{BRIDGE_GOAL_TOLERANCE_RAD} rad)"
    )


def optional_float(value: Optional[float]) -> Optional[float]:
    """``None``-safe float conversion (a small convenience for the transport)."""
    return None if value is None else float(value)
