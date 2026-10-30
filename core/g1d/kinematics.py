"""A minimal forward-kinematics model of the G1-D arms, straight from the URDF.

Why this exists: the harness's atomic controller must report the **measured** end-
effector pose and the **measured** gripper state, and it must be able to say where
the tool is relative to the wrist. The G1-D arm is driven through the SDK bridge,
which exposes joint feedback but no Cartesian pose feedback -- so the pose has to
come from FK on the measured joints.

The chain constants below are transcribed from
``lzh_description/urdf/g1_d_ros2.urdf`` (the G1-D URDF vendored in the LZH
workspace): the 7 arm
joints per side, each with its origin (xyz, rpy) and axis, in the URDF's own
order. The Dex1 gripper mount and finger chains are included so the tool point
(the midpoint between the two fingertip links) can be resolved.

This is a **kinematic** model only: no dynamics, no collision geometry, no
actuation. It is cross-checked against the URDF in
``tests/test_g1d_kinematics.py`` -- if the URDF changes, that test fails loudly
instead of the robot silently moving to the wrong place.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

#: The G1-D arm joints, in the URDF's order (shoulder pitch, roll, yaw, elbow,
#: wrist roll, pitch, yaw). This is the order the harness uses everywhere: the
#: MoveIt group, the bridge's ``left_joints_``/``right_joints_`` tables, and the
#: joint limits in :mod:`core.g1d.geometry` all use it.
ARM_JOINT_SUFFIXES = (
    "shoulder_pitch_joint",
    "shoulder_roll_joint",
    "shoulder_yaw_joint",
    "elbow_joint",
    "wrist_roll_joint",
    "wrist_pitch_joint",
    "wrist_yaw_joint",
)

#: (joint_name, origin_xyz, origin_rpy, axis) for the LEFT arm, transcribed from
#: ``g1_d_ros2.urdf``.
LEFT_ARM_CHAIN: tuple[tuple[str, tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]], ...] = (
    ("left_shoulder_pitch_joint", (0.0039563, 0.10022, 0.24778), (0.27931, 5.4949e-05, -0.00019159), (0.0, 1.0, 0.0)),
    ("left_shoulder_roll_joint", (0.0, 0.038, -0.013831), (-0.27925, 0.0, 0.0), (1.0, 0.0, 0.0)),
    ("left_shoulder_yaw_joint", (0.0, 0.00624, -0.1032), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    ("left_elbow_joint", (0.015783, 0.0, -0.080518), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    ("left_wrist_roll_joint", (0.1, 0.00188791, -0.01), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
    ("left_wrist_pitch_joint", (0.038, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    ("left_wrist_yaw_joint", (0.046, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
)

#: The RIGHT arm chain (the URDF mirrors the left arm in Y).
RIGHT_ARM_CHAIN: tuple[tuple[str, tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]], ...] = (
    ("right_shoulder_pitch_joint", (0.0039563, -0.10021, 0.24778), (-0.27931, 5.4949e-05, 0.00019159), (0.0, 1.0, 0.0)),
    ("right_shoulder_roll_joint", (0.0, -0.038, -0.013831), (0.27925, 0.0, 0.0), (1.0, 0.0, 0.0)),
    ("right_shoulder_yaw_joint", (0.0, -0.00624, -0.1032), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    ("right_elbow_joint", (0.015783, 0.0, -0.080518), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    ("right_wrist_roll_joint", (0.1, -0.00188791, -0.01), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
    ("right_wrist_pitch_joint", (0.038, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    ("right_wrist_yaw_joint", (0.046, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
)

#: The Dex1 gripper mount on the wrist yaw link (fixed rotation of -90 deg about Z).
DEX1_MOUNT_ORIGIN_XYZ = (0.0, 0.0, 0.0)
DEX1_MOUNT_ORIGIN_RPY = (0.0, 0.0, -1.57079633)

#: The two Dex1 finger chains, each ending at a fingertip link.
#: ``finger1`` -> ``left_dex1_1_Link1_3``, ``finger2`` -> ``left_dex1_1_Link2_2``.
DEX1_FINGER1_CHAIN: tuple[tuple[tuple[float, float, float], tuple[float, float, float]], ...] = (
    ((0.04055, 0.0593, 0.0152), (0.0, 0.0, 0.0)),
    ((-0.013, 0.013, 0.0), (0.0, 0.0, 0.0)),
    ((-0.0025165, 0.025043, -0.001), (0.0, 0.0, 0.0)),
)
DEX1_FINGER2_CHAIN: tuple[tuple[tuple[float, float, float], tuple[float, float, float]], ...] = (
    ((-0.04055, 0.0593, -0.0152), (0.0, 0.0, 0.0)),
    ((0.013, 0.013, 0.0), (0.0, 0.0, 0.0)),
)

#: The tool point (TCP) as a fixed offset in the WRIST-YAW link frame: the
#: midpoint between the two fingertip links, computed from the URDF chain above
#: at the zero pose. The approach direction (wrist -> TCP) is essentially the
#: wrist link's +X axis (the component perpendicular to it is < 1 cm).
#:
#: This offset is CONSTANT in the wrist frame for every joint configuration
#: (verified to <1e-9 m), and the TCP's orientation EQUALS the wrist's (the Dex1
#: mount only translates along the tool axis). So the wrist->TCP transform is a
#: pure fixed translation -- which is what lets a MoveIt goal on the wrist link
#: be converted to a TCP target and back exactly.
TCP_OFFSET_IN_WRIST_M = (0.0848215, 0.00125825, -0.0005)

#: The unit approach direction (wrist -> TCP) in the wrist-yaw link frame.
TCP_APPROACH_DIRECTION_WRIST = (0.9999, 0.0148, -0.0059)

#: The Dex1 gripper's opening axis in the wrist-yaw link frame (the Dex1 base
#: link's X axis): the fingers slide along it.
DEX1_OPENING_AXIS_WRIST = (0.0, 1.0, 0.0)


def rpy_to_matrix(rpy: Sequence[float]) -> np.ndarray:
    """URDF fixed-axis roll-pitch-yaw (extrinsic xyz) -> rotation matrix."""
    r, p, y = (float(v) for v in np.asarray(rpy, dtype=float).reshape(-1))
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    rot_x = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    rot_y = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rot_z = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return rot_z @ rot_y @ rot_x


def axis_angle_to_matrix(axis: Sequence[float], angle: float) -> np.ndarray:
    """Rodrigues rotation about a (not-necessarily-unit) axis by ``angle`` radians."""
    unit = np.asarray(axis, dtype=float).reshape(-1)
    norm = float(np.linalg.norm(unit))
    if norm < 1e-12:
        return np.eye(3)
    unit = unit / norm
    x, y, z = unit
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return (
        np.eye(3)
        + math.sin(angle) * skew
        + (1.0 - math.cos(angle)) * (skew @ skew)
    )


def _homogeneous(rotation: np.ndarray, translation: Sequence[float]) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = np.asarray(translation, dtype=float).reshape(-1)
    return matrix


def arm_chain(side: str) -> tuple[tuple[str, tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]], ...]:
    """The 7-joint chain for ``side``."""
    if side == "left":
        return LEFT_ARM_CHAIN
    if side == "right":
        return RIGHT_ARM_CHAIN
    raise ValueError(f"side must be 'left' or 'right', got {side!r}")


def arm_joint_names(side: str) -> list[str]:
    """The 7 joint names for ``side``, in the URDF's order."""
    return [name for name, _, _, _ in arm_chain(side)]


@dataclass(frozen=True)
class ArmKinematics:
    """Forward kinematics for ONE G1-D arm plus its Dex1 gripper.

    ``joint_positions`` is the 7 arm joints in the URDF order (radians).
    """

    side: str

    def __post_init__(self) -> None:
        if self.side not in ("left", "right"):
            raise ValueError(f"side must be 'left' or 'right', got {self.side!r}")

    # -- forward kinematics -------------------------------------------------
    def wrist_matrix(self, joint_positions: Sequence[float]) -> np.ndarray:
        """The wrist-yaw link's 4x4 pose in the ``torso_link`` frame."""
        q = np.asarray(joint_positions, dtype=float).reshape(-1)
        if q.size != 7:
            raise ValueError(f"expected 7 joint positions, got {q.size}")
        if not np.all(np.isfinite(q)):
            raise ValueError("joint positions must be finite")
        matrix = np.eye(4)
        for (_, origin_xyz, origin_rpy, axis), angle in zip(arm_chain(self.side), q):
            joint = _homogeneous(rpy_to_matrix(origin_rpy), origin_xyz)
            joint[:3, :3] = axis_angle_to_matrix(axis, float(angle)) @ joint[:3, :3]
            matrix = matrix @ joint
        return matrix

    def tcp_matrix(self, joint_positions: Sequence[float]) -> np.ndarray:
        """The tool point (TCP) pose in the ``torso_link`` frame.

        The TCP is the midpoint between the two Dex1 fingertip links, so it
        tracks the fingers as they open and close (the fingertip chain's own
        prismatic travel is included).
        """
        wrist = self.wrist_matrix(joint_positions)
        mount = wrist @ _homogeneous(
            rpy_to_matrix(DEX1_MOUNT_ORIGIN_RPY), DEX1_MOUNT_ORIGIN_XYZ
        )
        tip1 = mount
        for origin_xyz, origin_rpy in DEX1_FINGER1_CHAIN:
            tip1 = tip1 @ _homogeneous(rpy_to_matrix(origin_rpy), origin_xyz)
        tip2 = mount
        for origin_xyz, origin_rpy in DEX1_FINGER2_CHAIN:
            tip2 = tip2 @ _homogeneous(rpy_to_matrix(origin_rpy), origin_xyz)
        midpoint = 0.5 * (tip1[:3, 3] + tip2[:3, 3])
        tcp = np.eye(4)
        tcp[:3, :3] = wrist[:3, :3]
        tcp[:3, 3] = midpoint
        return tcp

    def wrist_pose7(self, joint_positions: Sequence[float]) -> np.ndarray:
        """The wrist-yaw link's pose as ``[x, y, z, qx, qy, qz, qw]``."""
        return matrix_to_pose7(self.wrist_matrix(joint_positions))

    def tcp_pose7(self, joint_positions: Sequence[float]) -> np.ndarray:
        """The TCP's pose as ``[x, y, z, qx, qy, qz, qw]``."""
        return matrix_to_pose7(self.tcp_matrix(joint_positions))

    def opening_width_m(
        self, joint_positions: Sequence[float], *, motor_rad: float
    ) -> float:
        """The nominal jaw opening (m) implied by the measured Dex1 motor angle.

        Uses the bridge's own provisional mapping (motor radians -> meters). It is
        NOMINAL: the mapping is not a calibrated jaw width, so it must not be read
        as a grasp force or a real object width.
        """
        from core.g1d.geometry import dex1_motor_to_opening

        return dex1_motor_to_opening(float(motor_rad))


def matrix_to_pose7(matrix: np.ndarray) -> np.ndarray:
    """A 4x4 homogeneous matrix -> ``[x, y, z, qx, qy, qz, qw]``."""
    rotation = np.asarray(matrix, dtype=float)[:3, :3]
    quaternion = _matrix_to_quaternion(rotation)
    translation = np.asarray(matrix, dtype=float)[:3, 3]
    return np.concatenate([translation, quaternion])


def wrist_to_tcp_pose7(wrist_pose7: Sequence[float]) -> np.ndarray:
    """Convert a wrist-yaw link pose into the corresponding TCP pose.

    The TCP sits a FIXED distance along the tool axis from the wrist, with the
    SAME orientation (the Dex1 mount only translates). The tool axis is the
    wrist link's local +X, so the offset in the torso frame is
    ``R_wrist @ TCP_OFFSET_IN_WRIST_M``.

    This is the inverse of :func:`tcp_to_wrist_pose7`, and both are exact
    (no iteration) because the offset is constant in the wrist frame.
    """
    pose = np.asarray(wrist_pose7, dtype=float).reshape(-1)
    if pose.size != 7:
        raise ValueError(f"expected 7 values [x,y,z,qx,qy,qz,qw], got {pose.size}")
    rotation = _quaternion_to_matrix(pose[3:])
    offset = rotation @ np.asarray(TCP_OFFSET_IN_WRIST_M, dtype=float)
    return np.concatenate([pose[:3] + offset, pose[3:]])


def tcp_to_wrist_pose7(tcp_pose7: Sequence[float]) -> np.ndarray:
    """Convert a TCP pose into the corresponding wrist-yaw link pose.

    The inverse of :func:`wrist_to_tcp_pose7`: the TCP origin is the wrist
    origin plus the fixed tool-axis offset, so the wrist origin is the TCP
    origin MINUS that offset (same orientation).

    Needed because MoveIt's ``left_arm`` group is planned to the
    ``left_wrist_yaw_link`` tip, while the harness reports and commands the TCP
    (the fingertip midpoint) -- so every MoveIt goal needs this conversion.
    """
    pose = np.asarray(tcp_pose7, dtype=float).reshape(-1)
    if pose.size != 7:
        raise ValueError(f"expected 7 values [x,y,z,qx,qy,qz,qw], got {pose.size}")
    rotation = _quaternion_to_matrix(pose[3:])
    offset = rotation @ np.asarray(TCP_OFFSET_IN_WRIST_M, dtype=float)
    return np.concatenate([pose[:3] - offset, pose[3:]])


def _quaternion_to_matrix(quaternion: Sequence[float]) -> np.ndarray:
    """Unit quaternion (x, y, z, w) -> rotation matrix."""
    x, y, z, w = (float(v) for v in np.asarray(quaternion, dtype=float).reshape(-1))
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1e-12:
        raise ValueError("quaternion must be non-zero")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _matrix_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    """Rotation matrix -> unit quaternion (x, y, z, w), numerically stable."""
    m = np.asarray(rotation, dtype=float)
    trace = float(m[0, 0] + m[1, 1] + m[2, 2])
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    quaternion = np.array([x, y, z, w], dtype=float)
    norm = float(np.linalg.norm(quaternion))
    if norm < 1e-12:
        raise ValueError("degenerate rotation matrix")
    quaternion /= norm
    if quaternion[3] < 0.0:  # canonical sign (w >= 0)
        quaternion = -quaternion
    return quaternion
