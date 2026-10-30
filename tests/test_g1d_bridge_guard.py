"""Tests for the G1-D bridge-guard mirror and the kinematics model.

These are pure-Python tests: no ROS, no robot. They lock down the two things the
harness must not get wrong about the G1-D embodiment:

* the forward-kinematics model matches the official URDF (so a reported end-effector
  pose is the real one), and
* the bridge-guard mirror rejects exactly what the SDK bridge rejects (so a
  trajectory that passes here is not silently refused on the robot).
"""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.g1d import bridge_guard as bg
from core.g1d.geometry import (
    BRIDGE_MAX_TRAJECTORY_SPEED_RAD_S,
    LEFT_ARM_LIMITS_RAD,
    RIGHT_ARM_LIMITS_RAD,
    joint_position_limits_error,
)
from core.g1d.kinematics import (
    ArmKinematics,
    arm_joint_names,
    matrix_to_pose7,
    tcp_to_wrist_pose7,
    wrist_to_tcp_pose7,
)

URDF_PATH = Path.home() / "g1d_ros2_lzh/src/lzh_description/urdf/g1_d_ros2.urdf"


def _parse_urdf_chain(side: str) -> list[tuple]:
    """Parse the arm chain straight out of the URDF (the authority for the numbers)."""
    import xml.etree.ElementTree as ET

    tree = ET.parse(URDF_PATH)
    robot = tree.getroot()
    prefix = "left" if side == "left" else "right"
    chain: list[tuple] = []
    joint_names = [
        f"{prefix}_shoulder_pitch_joint",
        f"{prefix}_shoulder_roll_joint",
        f"{prefix}_shoulder_yaw_joint",
        f"{prefix}_elbow_joint",
        f"{prefix}_wrist_roll_joint",
        f"{prefix}_wrist_pitch_joint",
        f"{prefix}_wrist_yaw_joint",
    ]
    for joint_name in joint_names:
        joint = robot.find(f"joint[@name='{joint_name}']")
        if joint is None:
            raise AssertionError(f"URDF has no joint {joint_name}")
        origin = joint.find("origin")
        axis = joint.find("axis")
        xyz = [float(v) for v in origin.get("xyz", "0 0 0").split()]
        rpy = [float(v) for v in origin.get("rpy", "0 0 0").split()]
        ax = [float(v) for v in axis.get("xyz", "0 0 1").split()]
        chain.append((joint_name, tuple(xyz), tuple(rpy), tuple(ax)))
    return chain


class TestKinematicsMatchesUrdf(unittest.TestCase):
    """The FK model must reproduce the URDF exactly, for both arms."""

    def test_chain_constants_match_the_urdf(self) -> None:
        for side in ("left", "right"):
            expected = _parse_urdf_chain(side)
            from core.g1d.kinematics import arm_chain

            actual = arm_chain(side)
            self.assertEqual(len(actual), len(expected))
            for (name, xyz, rpy, axis), (exp_name, exp_xyz, exp_rpy, exp_axis) in zip(
                actual, expected
            ):
                self.assertEqual(name, exp_name)
                for got, want in zip(xyz, exp_xyz):
                    self.assertAlmostEqual(got, want, places=9)
                for got, want in zip(rpy, exp_rpy):
                    self.assertAlmostEqual(got, want, places=9)
                for got, want in zip(axis, exp_axis):
                    self.assertAlmostEqual(got, want, places=9)

    def test_fk_matches_an_independent_urdf_forward_pass(self) -> None:
        """Recompute the FK from the URDF XML and compare to the module."""
        from core.g1d.kinematics import axis_angle_to_matrix, rpy_to_matrix

        for side in ("left", "right"):
            chain = _parse_urdf_chain(side)
            kin = ArmKinematics(side)
            rng = np.random.default_rng(7)
            for _ in range(25):
                q = rng.uniform(-1.2, 1.2, 7)
                matrix = np.eye(4)
                for (_, xyz, rpy, axis), angle in zip(chain, q):
                    joint = np.eye(4)
                    joint[:3, :3] = rpy_to_matrix(rpy)
                    joint[:3, 3] = xyz
                    joint[:3, :3] = axis_angle_to_matrix(axis, float(angle)) @ joint[:3, :3]
                    matrix = matrix @ joint
                expected = matrix_to_pose7(matrix)
                got = kin.wrist_pose7(q)
                np.testing.assert_allclose(got, expected, atol=1e-9)

    def test_rotation_is_a_proper_rotation(self) -> None:
        kin = ArmKinematics("left")
        rng = np.random.default_rng(11)
        for _ in range(20):
            q = rng.uniform(-1.2, 1.2, 7)
            rotation = kin.wrist_matrix(q)[:3, :3]
            np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
            self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.0, places=12)

    def test_right_arm_mirrors_the_left_arm_in_y(self) -> None:
        """The URDF mirrors the left arm in Y, so the FK must mirror too.

        The mirror is a Y-flip of the pose combined with a sign flip of the four
        joints whose axis is mirrored (shoulder roll/yaw, wrist roll/yaw). The
        positions agree to the URDF's own origin asymmetry: the left shoulder
        origin is at y=+0.10022 and the right at y=-0.10021, a 1e-5 m difference
        in the source URDF itself.
        """
        mirror = np.diag([1.0, -1.0, 1.0])
        negate = (False, True, True, False, True, False, True)
        left = ArmKinematics("left")
        right = ArmKinematics("right")
        rng = np.random.default_rng(13)
        for _ in range(20):
            q = rng.uniform(-1.0, 1.0, 7)
            qr = np.array(
                [(-q[i] if negate[i] else q[i]) for i in range(7)], dtype=float
            )
            lp = left.wrist_matrix(q)
            rp = right.wrist_matrix(qr)
            np.testing.assert_allclose(
                rp[:3, :3], mirror @ lp[:3, :3] @ mirror, atol=1e-9
            )
            np.testing.assert_allclose(rp[:3, 3], mirror @ lp[:3, 3], atol=2e-5)

    def test_tcp_is_a_fixed_offset_from_the_wrist(self) -> None:
        """The wrist->TCP offset must be constant in the wrist frame."""
        for side in ("left", "right"):
            kin = ArmKinematics(side)
            rng = np.random.default_rng(17)
            for _ in range(25):
                q = rng.uniform(-1.2, 1.2, 7)
                wrist = kin.wrist_matrix(q)
                tcp = kin.tcp_matrix(q)
                offset_in_wrist = wrist[:3, :3].T @ (tcp[:3, 3] - wrist[:3, 3])
                np.testing.assert_allclose(
                    offset_in_wrist,
                    np.asarray(
                        __import__(
                            "core.g1d.kinematics", fromlist=["TCP_OFFSET_IN_WRIST_M"]
                        ).TCP_OFFSET_IN_WRIST_M
                    ),
                    atol=1e-9,
                )

    def test_wrist_tcp_conversion_round_trips(self) -> None:
        for side in ("left", "right"):
            kin = ArmKinematics(side)
            rng = np.random.default_rng(19)
            for _ in range(25):
                q = rng.uniform(-1.2, 1.2, 7)
                tcp = kin.tcp_pose7(q)
                wrist = kin.wrist_pose7(q)
                np.testing.assert_allclose(tcp_to_wrist_pose7(tcp), wrist, atol=1e-9)
                np.testing.assert_allclose(wrist_to_tcp_pose7(wrist), tcp, atol=1e-9)


class TestBridgeGuardStructure(unittest.TestCase):
    """The mirror of the bridge's ``validate_message``."""

    def _good(self) -> bg.TrajectoryData:
        names = tuple(arm_joint_names("left"))
        return bg.TrajectoryData(
            joint_names=names,
            times_s=(0.0, 1.0, 2.0),
            positions=(
                (0.0,) * 7,
                (0.05,) * 7,
                (0.10,) * 7,
            ),
            velocities=((0.0,) * 7, (0.05,) * 7, (0.05,) * 7),
        )

    def test_accepts_a_well_formed_trajectory(self) -> None:
        data = self._good()
        self.assertIsNone(
            bg.validate_trajectory_structure(data, arm_joint_names("left"))
        )

    def test_rejects_an_empty_trajectory(self) -> None:
        data = bg.TrajectoryData(
            joint_names=tuple(arm_joint_names("left")), times_s=(), positions=()
        )
        reason = bg.validate_trajectory_structure(data, arm_joint_names("left"))
        self.assertIsNotNone(reason)
        self.assertIn("no points", reason)

    def test_rejects_a_wrong_joint_count(self) -> None:
        names = tuple(arm_joint_names("left"))[:6]
        data = bg.TrajectoryData(
            joint_names=names,
            times_s=(0.0,),
            positions=((0.0,) * 6,),
        )
        reason = bg.validate_trajectory_structure(data, arm_joint_names("left"))
        self.assertIsNotNone(reason)

    def test_rejects_the_opposite_arm(self) -> None:
        """A right-arm trajectory must not be sent to the left arm's Action."""
        names = tuple(arm_joint_names("right"))
        data = bg.TrajectoryData(
            joint_names=names,
            times_s=(0.0,),
            positions=((0.0,) * 7,),
        )
        reason = bg.validate_trajectory_structure(data, arm_joint_names("left"))
        self.assertIsNotNone(reason)
        self.assertIn("unexpected joint", reason)

    def test_rejects_non_increasing_timestamps(self) -> None:
        data = bg.TrajectoryData(
            joint_names=tuple(arm_joint_names("left")),
            times_s=(0.0, 1.0, 1.0),
            positions=((0.0,) * 7, (0.1,) * 7, (0.2,) * 7),
        )
        reason = bg.validate_trajectory_structure(data, arm_joint_names("left"))
        self.assertIsNotNone(reason)
        self.assertIn("non-increasing", reason)

    def test_rejects_non_finite_positions(self) -> None:
        data = bg.TrajectoryData(
            joint_names=tuple(arm_joint_names("left")),
            times_s=(0.0,),
            positions=((0.0,) * 6 + (float("nan"),),),
        )
        reason = bg.validate_trajectory_structure(data, arm_joint_names("left"))
        self.assertIsNotNone(reason)
        self.assertIn("NaN/Inf", reason)


class TestBridgeGuardRealState(unittest.TestCase):
    """The mirror of the bridge's ``validate_real_state``."""

    def _names(self) -> list[str]:
        return arm_joint_names("left")

    def test_accepts_a_slow_trajectory_starting_at_the_measured_state(self) -> None:
        names = tuple(self._names())
        data = bg.TrajectoryData(
            joint_names=names,
            times_s=(0.0, 2.0),
            positions=((0.0,) * 7, (0.2,) * 7),
            velocities=((0.0,) * 7, (0.1,) * 7),
        )
        reason = bg.validate_trajectory_against_bridge(
            data, LEFT_ARM_LIMITS_RAD, [0.0] * 7
        )
        self.assertIsNone(reason)

    def test_rejects_a_start_state_mismatch(self) -> None:
        """The first waypoint must match the measured joints (no jump at goal start)."""
        names = tuple(self._names())
        data = bg.TrajectoryData(
            joint_names=names,
            times_s=(0.0, 2.0),
            positions=((0.5,) * 7, (0.6,) * 7),
            velocities=((0.0,) * 7, (0.05,) * 7),
        )
        reason = bg.validate_trajectory_against_bridge(
            data, LEFT_ARM_LIMITS_RAD, [0.0] * 7
        )
        self.assertIsNotNone(reason)
        self.assertIn("start-state mismatch", reason)

    def test_rejects_a_trajectory_that_exceeds_the_speed_guard(self) -> None:
        """A fast move must be caught even when the waypoints look slow enough.

        The waypoint-to-waypoint speed here is 0.2 rad/s (under the 0.5 limit), but
        the Hermite interpolation between the two points (with the endpoint
        velocities) overshoots past the limit -- which is exactly why the bridge
        samples instead of checking the waypoints.
        """
        names = tuple(self._names())
        data = bg.TrajectoryData(
            joint_names=names,
            times_s=(0.0, 0.5),
            positions=((0.0,) * 7, (0.1,) * 7),
            velocities=((2.0,) * 7, (2.0,) * 7),
        )
        reason = bg.validate_trajectory_against_bridge(
            data, LEFT_ARM_LIMITS_RAD, [0.0] * 7
        )
        self.assertIsNotNone(reason)
        self.assertIn("speed guard", reason)

    def test_rejects_a_trajectory_that_violates_a_joint_limit(self) -> None:
        """A commanded position outside the exact URDF bound must be rejected.

        The motion is slow enough to satisfy the speed guard, so the LIMIT check is
        what rejects it -- and it is checked on the SAMPLED positions, not only on
        the waypoints.
        """
        names = tuple(self._names())
        lower = LEFT_ARM_LIMITS_RAD[0][0]
        data = bg.TrajectoryData(
            joint_names=names,
            times_s=(0.0, 20.0),
            positions=((0.0,) * 7, (lower - 0.5,) * 7),
            velocities=((0.0,) * 7, (-0.05,) * 7),
        )
        reason = bg.validate_trajectory_against_bridge(
            data, LEFT_ARM_LIMITS_RAD, [0.0] * 7
        )
        self.assertIsNotNone(reason)
        self.assertIn("position limit violated", reason)

    def test_right_arm_limits_differ_from_the_left(self) -> None:
        """The shoulder-roll limit is mirrored, so the wrong table would be wrong.

        The trajectory starts AT the measured state (so the start-state check
        passes) and moves to 1.8 rad on the shoulder-roll joint -- inside the LEFT
        limit, outside the RIGHT one.
        """
        self.assertNotEqual(LEFT_ARM_LIMITS_RAD[1], RIGHT_ARM_LIMITS_RAD[1])
        names = tuple(self._names())
        start = [0.0] * 7
        end = [0.0] * 7
        end[1] = 1.8
        data = bg.TrajectoryData(
            joint_names=names,
            times_s=(0.0, 8.0),
            positions=(tuple(start), tuple(end)),
            velocities=((0.0,) * 7, (0.0,) * 7),
        )
        self.assertIsNone(
            bg.validate_trajectory_against_bridge(data, LEFT_ARM_LIMITS_RAD, start)
        )
        reason = bg.validate_trajectory_against_bridge(data, RIGHT_ARM_LIMITS_RAD, start)
        self.assertIsNotNone(reason)
        self.assertIn("position limit violated", reason)


class TestBridgeGuardSampling(unittest.TestCase):
    """The sampler must reproduce the bridge's interpolation exactly."""

    def test_linear_interpolation_without_velocities(self) -> None:
        names = tuple(arm_joint_names("left"))
        data = bg.TrajectoryData(
            joint_names=names,
            times_s=(0.0, 1.0),
            positions=((0.0,) * 7, (1.0,) * 7),
        )
        np.testing.assert_allclose(bg.sample_trajectory(data, 0.5), [0.5] * 7, atol=1e-12)
        np.testing.assert_allclose(bg.sample_trajectory(data, 0.25), [0.25] * 7, atol=1e-12)

    def test_hermite_interpolation_with_velocities(self) -> None:
        names = tuple(arm_joint_names("left"))
        data = bg.TrajectoryData(
            joint_names=names,
            times_s=(0.0, 1.0),
            positions=((0.0,) * 7, (1.0,) * 7),
            velocities=((0.0,) * 7, (0.0,) * 7),
        )
        # Hermite with zero endpoint velocities over dt=1: h00*q0 + h01*q1.
        value = bg.sample_trajectory(data, 0.5)[0]
        expected = 0.5  # 2*0.125 - 3*0.25 + 1 = 0.5
        self.assertAlmostEqual(float(value), expected, places=12)

    def test_clamping_outside_the_trajectory(self) -> None:
        names = tuple(arm_joint_names("left"))
        data = bg.TrajectoryData(
            joint_names=names,
            times_s=(1.0, 2.0),
            positions=((0.0,) * 7, (1.0,) * 7),
        )
        np.testing.assert_allclose(bg.sample_trajectory(data, 0.0), [0.0] * 7, atol=1e-12)
        np.testing.assert_allclose(bg.sample_trajectory(data, 5.0), [1.0] * 7, atol=1e-12)


class TestBridgeGuardReport(unittest.TestCase):
    """The combined report."""

    def test_report_on_an_acceptable_trajectory(self) -> None:
        names = tuple(arm_joint_names("left"))
        data = bg.TrajectoryData(
            joint_names=names,
            times_s=(0.0, 2.0),
            positions=((0.0,) * 7, (0.2,) * 7),
            velocities=((0.0,) * 7, (0.1,) * 7),
        )
        report = bg.check_trajectory(
            data, LEFT_ARM_LIMITS_RAD, [0.0] * 7, list(names)
        )
        self.assertTrue(report.ok, report.reason)
        self.assertIn("bridge guard OK", report.summary())
        self.assertAlmostEqual(report.duration_s, 2.0)

    def test_report_on_a_rejected_trajectory(self) -> None:
        names = tuple(arm_joint_names("left"))
        data = bg.TrajectoryData(
            joint_names=names,
            times_s=(0.0, 2.0),
            positions=((1.0,) * 7, (1.2,) * 7),
            velocities=((0.0,) * 7, (0.1,) * 7),
        )
        report = bg.check_trajectory(
            data, LEFT_ARM_LIMITS_RAD, [0.0] * 7, list(names)
        )
        self.assertFalse(report.ok)
        self.assertIn("REJECTED", report.summary())

    def test_peak_speed_is_reported(self) -> None:
        names = tuple(arm_joint_names("left"))
        data = bg.TrajectoryData(
            joint_names=names,
            times_s=(0.0, 2.0),
            positions=((0.0,) * 7, (0.4,) * 7),
            velocities=((0.0,) * 7, (0.2,) * 7),
        )
        report = bg.check_trajectory(
            data, LEFT_ARM_LIMITS_RAD, [0.0] * 7, list(names)
        )
        self.assertTrue(report.ok, report.reason)
        self.assertGreater(report.max_speed_rad_s, 0.0)
        self.assertLessEqual(
            report.max_speed_rad_s, BRIDGE_MAX_TRAJECTORY_SPEED_RAD_S + 1e-6
        )


class TestGeometryHelpers(unittest.TestCase):
    """The shared limit helpers."""

    def test_limits_error_reports_the_offending_joint(self) -> None:
        positions = [0.0] * 7
        positions[3] = LEFT_ARM_LIMITS_RAD[3][1] + 0.01
        reason = joint_position_limits_error(positions, LEFT_ARM_LIMITS_RAD)
        self.assertIsNotNone(reason)
        self.assertIn("joint 3", reason)

    def test_limits_error_accepts_the_exact_bounds(self) -> None:
        positions = [0.0] * 7
        for index, (lower, upper) in enumerate(LEFT_ARM_LIMITS_RAD):
            positions[index] = upper
        self.assertIsNone(joint_position_limits_error(positions, LEFT_ARM_LIMITS_RAD))

    def test_limits_error_needs_the_right_count(self) -> None:
        self.assertIsNotNone(joint_position_limits_error([0.0] * 3, LEFT_ARM_LIMITS_RAD))


if __name__ == "__main__":
    unittest.main(verbosity=2)
