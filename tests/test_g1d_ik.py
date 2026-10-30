"""Tests for the G1-D inverse kinematics, the arm profile and the transport logic.

No ROS, no robot: these are pure-Python tests that lock down the math and the
safety rules the harness depends on.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.g1d.bridge_guard import TrajectoryData, check_trajectory, validate_trajectory_structure
from core.g1d.geometry import LEFT_ARM_LIMITS_RAD, RIGHT_ARM_LIMITS_RAD
from core.g1d.inverse_kinematics import ArmInverseKinematics
from core.g1d.kinematics import ArmKinematics, arm_joint_names, tcp_to_wrist_pose7
from core.g1d.single_arm import (
    BRIDGE_NAMESPACES,
    DEFAULT_BRIDGE_NAMESPACE,
    SingleArmProfile,
)


class TestSingleArmProfile(unittest.TestCase):
    """The profile is the single place that knows which arm is selected."""

    def test_left_and_right_select_different_endpoints(self) -> None:
        left = SingleArmProfile("left")
        right = SingleArmProfile("right")
        self.assertNotEqual(left.trajectory_action, right.trajectory_action)
        self.assertNotEqual(left.gripper_action, right.gripper_action)
        self.assertEqual(left.group_name, "left_arm")
        self.assertEqual(right.group_name, "right_arm")
        self.assertTrue(all(name.startswith("left_") for name in left.joint_names))
        self.assertTrue(all(name.startswith("right_") for name in right.joint_names))

    def test_the_lzh_namespace_selects_the_gripper_endpoint(self) -> None:
        lzh = SingleArmProfile("left", namespace="lzh")
        self.assertEqual(lzh.gripper_action, "/lzh_left_dex1/gripper_command")
        self.assertEqual(set(BRIDGE_NAMESPACES), {"lzh"})

    def test_senior_bridge_namespace_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "namespace must be one of"):
            SingleArmProfile("left", namespace="g1d")

    def test_the_default_namespace_is_the_running_build(self) -> None:
        self.assertEqual(DEFAULT_BRIDGE_NAMESPACE, "lzh")
        self.assertIn(DEFAULT_BRIDGE_NAMESPACE, BRIDGE_NAMESPACES)

    def test_an_unknown_namespace_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "namespace must be one of"):
            SingleArmProfile("left", namespace="nope")

    def test_an_unknown_side_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "side must be exactly"):
            SingleArmProfile("middle")

    def test_trajectory_validation_rejects_the_wrong_joint_set(self) -> None:
        profile = SingleArmProfile("left")
        profile.validate_trajectory_joint_names(arm_joint_names("left"))
        with self.assertRaises(ValueError):
            profile.validate_trajectory_joint_names(arm_joint_names("right"))
        with self.assertRaises(ValueError):
            profile.validate_trajectory_joint_names(
                list(arm_joint_names("left")) + list(arm_joint_names("right"))
            )
        with self.assertRaises(ValueError):
            profile.validate_trajectory_joint_names([])


class TestInverseKinematics(unittest.TestCase):
    """The local IK must actually reach the poses it claims to reach."""

    def _target_from(self, kin, joints, *, d_translation=0.0, d_rotation=0.0, axis=None):
        """A TCP target a small step away from the given joint configuration."""
        tcp = kin.tcp_pose7(joints)
        rotation = R.from_quat(tcp[3:])
        if d_rotation:
            if axis is None:
                axis = np.array([0.0, 0.0, 1.0])
            rotation = rotation * R.from_rotvec(axis * d_rotation)
        position = tcp[:3] + np.asarray(d_translation, dtype=float)
        return np.concatenate([position, rotation.as_quat()])

    def test_it_reproduces_a_known_pose(self) -> None:
        """A pose that IS a valid joint configuration must be recovered."""
        kin = ArmKinematics("left")
        ik = ArmInverseKinematics("left")
        joints = np.array([0.2, 0.3, 0.0, -0.6, 0.0, 0.0, 0.0])
        target = kin.tcp_pose7(joints)
        solution = ik.solve(target, seed=joints)
        self.assertIsNotNone(solution)
        got = kin.tcp_pose7(solution)
        np.testing.assert_allclose(got[:3], target[:3], atol=1e-4)
        error = float(
            (R.from_quat(target[3:]).inv() * R.from_quat(got[3:])).magnitude()
        )
        self.assertLess(error, 3e-3)

    def test_it_reaches_a_small_translation_step(self) -> None:
        kin = ArmKinematics("left")
        ik = ArmInverseKinematics("left")
        joints = np.array([0.2, 0.3, 0.0, -0.6, 0.0, 0.0, 0.0])
        target = self._target_from(kin, joints, d_translation=(0.01, 0.0, 0.0))
        solution = ik.solve(target, seed=joints)
        self.assertIsNotNone(solution)
        got = kin.tcp_pose7(solution)
        np.testing.assert_allclose(got[:3], target[:3], atol=1e-4)

    def test_it_reaches_a_small_rotation_step(self) -> None:
        kin = ArmKinematics("left")
        ik = ArmInverseKinematics("left")
        joints = np.array([0.2, 0.3, 0.0, -0.6, 0.0, 0.0, 0.0])
        target = self._target_from(kin, joints, d_rotation=0.05)
        solution = ik.solve(target, seed=joints)
        self.assertIsNotNone(solution)
        got = kin.tcp_pose7(solution)
        error = float(
            (R.from_quat(target[3:]).inv() * R.from_quat(got[3:])).magnitude()
        )
        self.assertLess(error, 3e-3)

    def test_it_reaches_a_translation_step_from_the_live_pose(self) -> None:
        """A 1 cm step from the measured pose must actually converge.

        This is the pose the live rig reported, and the case that used to fail:
        the Jacobian's rotational rows were in a different frame from the
        orientation error, so the step stalled with ~270 mm / ~59 deg of
        residual instead of converging.
        """
        kin = ArmKinematics("left")
        ik = ArmInverseKinematics("left")
        joints = np.array([0.0063, 0.0416, -0.7163, 1.335, 0.0313, 0.182, 0.0171])
        tcp = kin.tcp_pose7(joints)
        for axis in ("x", "y", "z"):
            target = tcp.copy()
            target["xyz".index(axis)] += 0.01
            solution = ik.solve_with_error(target, seed=joints)
            self.assertIsNotNone(solution)
            self.assertTrue(solution.ok, f"{axis}: {solution}")
            self.assertLess(solution.position_error_m, 1e-4)
            self.assertLess(solution.orientation_error_rad, 3e-3)

    def test_it_converges_to_the_least_squares_solution_not_just_inside_tolerance(
        self,
    ) -> None:
        """The reported residual must be tight, not merely under the tolerance.

        The solver used to stop the moment the residual fell under the
        tolerance, so it returned a solution that only just squeaked past it.
        """
        kin = ArmKinematics("left")
        ik = ArmInverseKinematics("left")
        joints = np.array([0.0063, 0.0416, -0.7163, 1.335, 0.0313, 0.182, 0.0171])
        target = kin.tcp_pose7(joints)
        target[0] += 0.01
        solution = ik.solve_with_error(target, seed=joints)
        self.assertIsNotNone(solution)
        self.assertLess(solution.position_error_m, 1e-5)
        self.assertLess(solution.orientation_error_rad, 1e-4)

    def test_it_recovers_a_pose_from_a_random_configuration(self) -> None:
        """A wide sample of poses must be recoverable, not just the easy ones."""
        kin = ArmKinematics("left")
        ik = ArmInverseKinematics("left")
        rng = np.random.default_rng(0)
        recovered = 0
        trials = 50
        for _ in range(trials):
            joints = rng.uniform(-1.0, 1.0, 7)
            target = kin.tcp_pose7(joints)
            target[:3] += rng.normal(0.0, 0.01, 3)
            solution = ik.solve_with_error(target, seed=joints)
            if solution is not None and solution.ok:
                recovered += 1
        self.assertGreaterEqual(recovered, int(0.9 * trials))

    def test_the_solution_respects_the_joint_limits(self) -> None:
        """The solver must never return a joint outside the official G1-D bounds."""
        kin = ArmKinematics("left")
        ik = ArmInverseKinematics("left")
        rng = np.random.default_rng(5)
        joints = rng.uniform(-0.8, 0.8, 7)
        for _ in range(20):
            target = self._target_from(
                kin, joints, d_translation=(0.01, 0.0, 0.0), d_rotation=0.03
            )
            solution = ik.solve(target, seed=joints)
            if solution is None:
                continue
            for value, (lower, upper) in zip(solution, LEFT_ARM_LIMITS_RAD):
                self.assertGreaterEqual(value, lower - 1e-9)
                self.assertLessEqual(value, upper + 1e-9)
            joints = solution

    def test_an_unreachable_target_returns_none(self) -> None:
        """A target far outside the workspace must fail, not silently miss."""
        ik = ArmInverseKinematics("left")
        far = np.array([5.0, 5.0, 5.0, 0.0, 0.0, 0.0, 1.0])
        self.assertIsNone(ik.solve(far, seed=np.zeros(7)))

    def test_a_malformed_target_is_rejected(self) -> None:
        ik = ArmInverseKinematics("left")
        with self.assertRaises(ValueError):
            ik.solve([0.0, 0.0, 0.0], seed=np.zeros(7))
        with self.assertRaises(ValueError):
            ik.solve([float("nan")] * 7, seed=np.zeros(7))

    def test_it_solves_the_right_arm_too(self) -> None:
        kin = ArmKinematics("right")
        ik = ArmInverseKinematics("right")
        joints = np.array([0.2, -0.3, 0.0, -0.6, 0.0, 0.0, 0.0])
        target = self._target_from(kin, joints, d_translation=(0.01, 0.0, 0.0))
        solution = ik.solve(target, seed=joints)
        self.assertIsNotNone(solution)
        for value, (lower, upper) in zip(solution, RIGHT_ARM_LIMITS_RAD):
            self.assertGreaterEqual(value, lower - 1e-9)
            self.assertLessEqual(value, upper + 1e-9)

    def test_the_joint_trajectory_it_implies_passes_the_bridge_guard(self) -> None:
        """The whole point: an IK solution must become an ACCEPTABLE trajectory."""
        kin = ArmKinematics("left")
        ik = ArmInverseKinematics("left")
        joints = np.array([0.2, 0.3, 0.0, -0.6, 0.0, 0.0, 0.0])
        target = self._target_from(kin, joints, d_translation=(0.015, 0.0, 0.0))
        solution = ik.solve(target, seed=joints)
        self.assertIsNotNone(solution)

        # The same construction the transport uses.
        delta = np.abs(solution - joints)
        duration_s = max(0.2, float(np.max(delta)) / (0.5 * 0.9))
        steps = max(2, int(np.ceil(duration_s / 0.05)))
        times, positions, velocities = [], [], []
        for index in range(steps + 1):
            alpha = index / steps
            times.append(duration_s * alpha)
            positions.append(tuple(float(v) for v in (joints + alpha * (solution - joints))))
            velocities.append(tuple(float(v) for v in (solution - joints) / duration_s))
        data = TrajectoryData(
            joint_names=tuple(arm_joint_names("left")),
            times_s=tuple(times),
            positions=tuple(positions),
            velocities=tuple(velocities),
        )
        self.assertIsNone(
            validate_trajectory_structure(data, arm_joint_names("left"))
        )
        report = check_trajectory(
            data, LEFT_ARM_LIMITS_RAD, list(joints), list(arm_joint_names("left"))
        )
        self.assertTrue(report.ok, report.reason)


class TestTcpWristConversion(unittest.TestCase):
    """The TCP <-> wrist conversion must be exact and reversible."""

    def test_it_round_trips(self) -> None:
        for side in ("left", "right"):
            kin = ArmKinematics(side)
            rng = np.random.default_rng(21)
            for _ in range(20):
                joints = rng.uniform(-1.2, 1.2, 7)
                tcp = kin.tcp_pose7(joints)
                wrist = kin.wrist_pose7(joints)
                np.testing.assert_allclose(tcp_to_wrist_pose7(tcp), wrist, atol=1e-9)

    def test_the_offset_is_the_documented_one(self) -> None:
        from core.g1d.kinematics import TCP_OFFSET_IN_WRIST_M

        kin = ArmKinematics("left")
        wrist = kin.wrist_matrix(np.zeros(7))
        tcp = kin.tcp_matrix(np.zeros(7))
        offset = wrist[:3, :3].T @ (tcp[:3, 3] - wrist[:3, 3])
        np.testing.assert_allclose(offset, np.asarray(TCP_OFFSET_IN_WRIST_M), atol=1e-9)


if __name__ == "__main__":
    unittest.main(verbosity=2)
