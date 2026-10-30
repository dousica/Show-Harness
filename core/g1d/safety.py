"""A hard minimum-height (Z floor) guard for the G1-D arm.

Why this exists
---------------
The harness's other rigs (Franka, Piper) already carry a Z floor: the commanded
end-effector setpoint is never allowed below a calibrated height, so ``MV_DOWN``
cannot press the arm into the table. The G1-D path had no such guard -- a
Cartesian goal could be planned straight through a table.

This module is the G1-D version, and it is deliberately a **separate, explicit
layer** rather than a plugin: the G1-D transport is the last thing that runs
before a trajectory reaches the SDK bridge, so the guard lives HERE, where it
cannot be bypassed by a different script.

What it does
------------
* **Clamp** a commanded Cartesian target so its Z never goes below the floor.
* **Verify** a whole planned trajectory: every sampled joint configuration is
  forward-kinematics'd to the TCP (fingertip midpoint) and its height is
  compared against the floor. Checking only the endpoints would be useless --
  a trajectory can dip below the floor in the middle and come back up.

Frame
-----
The floor is a height in the ``torso_link`` frame (the SRDF chain base of
``left_arm``/``right_arm``), measured in metres, same convention as the
``torso_link``-based end-effector poses the transport reports.

The floor protects against the table. It is NOT a full collision model: the
arm can still collide with a shelf, a rack or the robot's own body if those
obstacles are absent from MoveIt's planning scene. The floor is the minimum
bar, not the whole safety story; see ``docs/g1d.md``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np

from core.g1d.bridge_guard import TrajectoryData, sample_trajectory
from core.g1d.geometry import TORSO_FRAME

#: Extra clearance (meters) added on top of the floor for the GRIPPER BODY,
#: not just the TCP. The Dex1 fingers extend below/behind the fingertip
#: midpoint, so a floor exactly at the TCP height still lets the fingers dip
#: into the table. Default 3 mm.
DEFAULT_HAND_CLEARANCE_M = 0.003


@dataclass(frozen=True)
class ZFloor:
    """A minimum end-effector height in the robot base frame.

    ``z_m`` is the floor itself (the TCP may not go below it). ``margin_m`` is
    the extra clearance required for the hand geometry, so the effective limit
    is ``z_m + margin_m`` -- but the floor is *reported* as ``z_m`` because
    that is the number the operator calibrates.
    """

    z_m: float
    frame: str = TORSO_FRAME
    margin_m: float = DEFAULT_HAND_CLEARANCE_M
    enabled: bool = True
    #: True when this height is a MEASURED (calibrated) value rather than a
    #: shipped placeholder. A calibrated floor may legitimately be 0.0 -- on
    #: the G1-D the tabletop coincides with the ``torso_link`` Z origin, so the
    #: measured table height is ~0.0 m. An UNCALIBRATED 0.0 is still refused:
    #: it would let the TCP descend to the torso origin, straight through the
    #: table.
    calibrated: bool = True

    def __post_init__(self) -> None:
        z = float(self.z_m)
        if not np.isfinite(z):
            raise ValueError("z_floor_m must be a finite number")
        if z < 0.0:
            raise ValueError(f"z_floor_m must be non-negative, got {z}")
        margin = float(self.margin_m)
        if not np.isfinite(margin) or margin < 0.0:
            raise ValueError("margin_m must be finite and non-negative")
        object.__setattr__(self, "z_m", z)
        object.__setattr__(self, "margin_m", margin)
        object.__setattr__(self, "frame", str(self.frame))
        object.__setattr__(self, "enabled", bool(self.enabled))
        object.__setattr__(self, "calibrated", bool(self.calibrated))

    # -- effective limit ---------------------------------------------------
    @property
    def effective_z_m(self) -> float:
        """The TCP height the arm is actually held at (floor + hand clearance)."""
        return self.z_m + self.margin_m

    # -- clamping ----------------------------------------------------------
    def clamp_xyz(self, xyz: Sequence[float]) -> tuple[np.ndarray, bool]:
        """Clamp a 3-vector's Z to the floor.

        Returns ``(clamped_xyz, was_clamped)``.
        """
        point = np.asarray(xyz, dtype=float).reshape(-1).copy()
        if point.size != 3:
            raise ValueError(f"expected 3 values, got {point.size}")
        if not np.all(np.isfinite(point)):
            raise ValueError("position values must be finite")
        if not self.enabled:
            return point, False
        limit = self.effective_z_m
        if point[2] < limit:
            point[2] = limit
            return point, True
        return point, False

    def clamp_pose7(self, pose7: Sequence[float]) -> tuple[np.ndarray, bool]:
        """Clamp a ``[x,y,z,qx,qy,qz,qw]`` pose's Z to the floor."""
        pose = np.asarray(pose7, dtype=float).reshape(-1).copy()
        if pose.size != 7:
            raise ValueError(f"expected 7 values [x,y,z,qx,qy,qz,qw], got {pose.size}")
        if not np.all(np.isfinite(pose)):
            raise ValueError("pose values must be finite")
        clamped, was_clamped = self.clamp_xyz(pose[:3])
        pose[:3] = clamped
        return pose, was_clamped

    # -- trajectory verification -------------------------------------------
    def check_trajectory(
        self,
        positions: Sequence[Sequence[float]],
        kinematics: Any,
        *,
        times_s: Optional[Sequence[float]] = None,
        velocities: Sequence[Sequence[float]] = (),
        joint_names: Optional[Sequence[str]] = None,
        sample_step_s: float = 0.02,
        min_samples: int = 10,
    ) -> str | None:
        """Verify that a joint trajectory never dips below the floor.

        ``positions`` are the per-waypoint joint vectors of ONE arm (the URDF's
        7-joint order). ``kinematics`` is an
        :class:`core.g1d.kinematics.ArmKinematics` used to forward-kinematics
        each sample to the TCP.

        The trajectory is sampled with the SAME sampler the SDK bridge uses
        (:func:`core.g1d.bridge_guard.sample_trajectory`, including its cubic
        Hermite branch), at least ``min_samples`` samples per segment stepping
        by ``sample_step_s`` seconds. That matters: a Hermite segment can
        OVERSHOOT its waypoints, so checking only the waypoints would miss a
        dip in the middle of a segment -- exactly where the arm would hit the
        table.

        Returns ``None`` when the trajectory is acceptable, else the reason.
        """
        if not self.enabled:
            return None
        points = [np.asarray(p, dtype=float).reshape(-1) for p in positions]
        if not points:
            return "trajectory has no points"
        for index, point in enumerate(points):
            if point.size != 7:
                return f"expected 7 joint values at point {index}, got {point.size}"
            if not np.all(np.isfinite(point)):
                return f"non-finite joint values at point {index}"

        if times_s is None:
            stamps = tuple(float(i) for i in range(len(points)))
        else:
            stamps = tuple(float(t) for t in times_s)
            if len(stamps) != len(points):
                return "times and positions must have the same length"
        if len(stamps) > 1 and any(
            stamps[i] <= stamps[i - 1] for i in range(1, len(stamps))
        ):
            return "trajectory timestamps must be strictly increasing"

        vel = tuple(tuple(float(v) for v in row) for row in velocities)
        if vel and len(vel) != len(points):
            return "velocities and positions must have the same length"
        names = tuple(joint_names) if joint_names else tuple(f"j{i}" for i in range(7))

        data = TrajectoryData(
            joint_names=names,
            times_s=stamps,
            positions=tuple(tuple(float(v) for v in p) for p in points),
            velocities=vel,
        )

        limit = self.effective_z_m
        # Walk the trajectory segment by segment, sampling exactly like the
        # bridge does, so a mid-segment dip cannot slip through.
        for segment in range(len(points)):
            t1 = float(stamps[segment])
            t0 = 0.0 if segment == 0 else float(stamps[segment - 1])
            dt = t1 - t0
            if dt < 0.0:
                return "trajectory timestamps invalid"
            samples = max(min_samples, int(np.ceil(dt / sample_step_s)))
            for step in range(samples + 1):
                t = t0 + dt * (float(step) / float(samples))
                joint = sample_trajectory(data, t)
                tcp = kinematics.tcp_matrix(joint)
                z = float(tcp[2, 3])
                if not np.isfinite(z):
                    return "forward kinematics produced a non-finite TCP height"
                if z < limit - 1e-6:
                    return (
                        f"trajectory dips to z={z:.4f} m at t={t:.3f}s, below the "
                        f"z-floor limit of {limit:.4f} m "
                        f"(floor {self.z_m:.4f} m + {self.margin_m:.4f} m hand clearance)"
                    )
        return None

    # -- reporting ---------------------------------------------------------
    def describe(self) -> str:
        """One-line, human-readable description for the console/log."""
        if not self.enabled:
            return "z-floor DISABLED (no software descent limit)"
        note = ""
        if self.z_m <= 0.0 and not self.calibrated:
            note = " (UNCALIBRATED placeholder)"
        return (
            f"z-floor {self.z_m:.4f} m in {self.frame} "
            f"(TCP limited to {self.effective_z_m:.4f} m){note}"
        )


def resolve_z_floor(
    robot_cfg: dict[str, Any],
    *,
    override_m: Optional[float] = None,
    enabled_default: bool = True,
) -> ZFloor:
    """Build a :class:`ZFloor` from a robot config.

    Precedence (highest first):

    1. ``override_m`` (a CLI ``--z-floor-m``) -- an explicit height wins;
    2. ``enable_z_floor: false`` in the config -- the floor is OFF (this is
       what makes the guard opt-out-able for a dry-run, and it must be a
       deliberate act, not a missing key);
    3. ``z_floor_m`` in the config -- the calibrated height;
    4. otherwise the floor is DISABLED and the caller is expected to refuse to
       run on hardware (there is no safe default height: it depends on where
       the table is).

    ``z_floors.<name>`` / ``z_floor_name`` (the Franka/Piper convention) are
    honoured too, so one config can carry a floor per table setting.

    ``z_floor_calibrated: true`` in the config marks the height as MEASURED.
    This matters for the G1-D, where the tabletop coincides with the
    ``torso_link`` Z origin and the measured floor is therefore ~0.0 m: without
    the marker a 0.0 floor would be indistinguishable from the shipped
    placeholder and would be refused. With the marker, 0.0 is a valid,
    calibrated height.
    """
    cfg = dict(robot_cfg or {})
    margin = float(cfg.get("z_floor_margin_m", DEFAULT_HAND_CLEARANCE_M))
    calibrated = bool(cfg.get("z_floor_calibrated", False))

    if override_m is not None:
        value = float(override_m)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"--z-floor-m must be a finite non-negative number, got {value!r}")
        return ZFloor(z_m=value, margin_m=margin, enabled=True, calibrated=True)

    enabled = bool(cfg.get("enable_z_floor", enabled_default))

    floors = cfg.get("z_floors") or {}
    name = cfg.get("z_floor_name")
    if name is not None and isinstance(floors, dict):
        if name not in floors:
            raise ValueError(
                f"robot config z_floor_name {name!r} is missing from z_floors"
            )
        value = float(floors[name])
        return ZFloor(z_m=value, margin_m=margin, enabled=enabled, calibrated=calibrated)

    value = cfg.get("z_floor_m")
    if value is None:
        return ZFloor(z_m=0.0, margin_m=margin, enabled=False, calibrated=False)
    return ZFloor(z_m=float(value), margin_m=margin, enabled=enabled, calibrated=calibrated)
