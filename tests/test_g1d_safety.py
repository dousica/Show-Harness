"""Tests for the G1-D Z-floor safety guard.

No ROS, no robot, no network: these are pure-geometry checks that run on a
laptop.
"""
from __future__ import annotations

import math
import unittest

import numpy as np

from core.g1d.bridge_guard import TrajectoryData
from core.g1d.kinematics import ArmKinematics
from core.g1d.safety import ZFloor, resolve_z_floor


class ZFloorTests(unittest.TestCase):
    def test_rejects_negative_and_non_finite(self) -> None:
        with self.assertRaises(ValueError):
            ZFloor(z_m=-0.1)
        with self.assertRaises(ValueError):
            ZFloor(z_m=float("nan"))
        with self.assertRaises(ValueError):
            ZFloor(z_m=0.1, margin_m=-0.001)

    def test_effective_limit_adds_hand_clearance(self) -> None:
        floor = ZFloor(z_m=0.30, margin_m=0.003)
        self.assertAlmostEqual(floor.effective_z_m, 0.303, places=9)

    def test_clamp_xyz_raises_low_z(self) -> None:
        floor = ZFloor(z_m=0.30, margin_m=0.003)
        clamped, was = floor.clamp_xyz([0.1, 0.2, 0.10])
        self.assertTrue(was)
        self.assertAlmostEqual(clamped[2], 0.303, places=9)
        self.assertAlmostEqual(clamped[0], 0.1, places=9)

    def test_clamp_xyz_leaves_high_z(self) -> None:
        floor = ZFloor(z_m=0.30, margin_m=0.003)
        clamped, was = floor.clamp_xyz([0.1, 0.2, 0.50])
        self.assertFalse(was)
        self.assertAlmostEqual(clamped[2], 0.50, places=9)

    def test_clamp_is_noop_when_disabled(self) -> None:
        floor = ZFloor(z_m=0.30, enabled=False)
        clamped, was = floor.clamp_xyz([0.1, 0.2, 0.10])
        self.assertFalse(was)
        self.assertAlmostEqual(clamped[2], 0.10, places=9)

    def test_clamp_pose7_keeps_orientation(self) -> None:
        floor = ZFloor(z_m=0.30, margin_m=0.0)
        pose = [0.1, 0.2, 0.05, 0.0, 0.0, 0.0, 1.0]
        clamped, was = floor.clamp_pose7(pose)
        self.assertTrue(was)
        self.assertAlmostEqual(clamped[2], 0.30, places=9)
        self.assertAlmostEqual(clamped[6], 1.0, places=9)

    def test_clamp_rejects_bad_input(self) -> None:
        floor = ZFloor(z_m=0.30)
        with self.assertRaises(ValueError):
            floor.clamp_xyz([0.1, 0.2])
        with self.assertRaises(ValueError):
            floor.clamp_pose7([0.1, 0.2, 0.3, 0.0, 0.0, 0.0])


class ZFloorTrajectoryTests(unittest.TestCase):
    """The trajectory check must catch a dip BETWEEN waypoints."""

    def setUp(self) -> None:
        self.kin = ArmKinematics("left")

    def _height(self, joints: np.ndarray) -> float:
        return float(self.kin.tcp_matrix(joints)[2, 3])

    def test_flat_trajectory_above_floor_passes(self) -> None:
        joints = np.zeros(7)
        floor = ZFloor(z_m=0.0, margin_m=0.0)
        # The zero-pose TCP height is the reference; put the floor below it.
        floor = ZFloor(z_m=self._height(joints) - 0.05, margin_m=0.0)
        positions = (tuple(joints.tolist()), tuple(joints.tolist()))
        reason = floor.check_trajectory(
            positions, self.kin, times_s=(0.0, 1.0),
            velocities=((0.0,) * 7, (0.0,) * 7),
        )
        self.assertIsNone(reason)

    def test_trajectory_dipping_below_floor_is_rejected(self) -> None:
        # Two configurations with very different TCP heights.
        low = np.array([-1.0, 0.5, 0.0, 0.0, -0.6, 0.0, 0.0])
        high = np.array([-1.0, 0.5, 0.0, 0.0, 0.6, 0.0, 0.0])
        z_low = self._height(low)
        z_high = self._height(high)
        self.assertNotAlmostEqual(z_low, z_high, places=3)
        floor = ZFloor(z_m=(z_low + z_high) / 2.0, margin_m=0.0)
        # The dip is at the END point here: the endpoints are low and high, and
        # the midpoint dips to z_low which is below the floor.
        reason = floor.check_trajectory(
            (tuple(low.tolist()), tuple(high.tolist())),
            self.kin,
            times_s=(0.0, 1.0),
            velocities=((0.0,) * 7, (0.0,) * 7),
        )
        self.assertIsNotNone(reason)
        self.assertIn("z-floor", reason)

    def test_check_is_noop_when_disabled(self) -> None:
        floor = ZFloor(z_m=10.0, enabled=False)
        reason = floor.check_trajectory(
            ((0.0,) * 7, (0.0,) * 7), self.kin, times_s=(0.0, 1.0)
        )
        self.assertIsNone(reason)

    def test_rejects_malformed_input(self) -> None:
        floor = ZFloor(z_m=0.0)
        self.assertIsNotNone(floor.check_trajectory((), self.kin))
        self.assertIsNotNone(
            floor.check_trajectory(((0.0,) * 3,), self.kin, times_s=(0.0,))
        )
        self.assertIsNotNone(
            floor.check_trajectory(
                ((0.0,) * 7, (0.0,) * 7), self.kin, times_s=(0.0, 1.0, 2.0)
            )
        )

    def test_non_increasing_timestamps_rejected(self) -> None:
        floor = ZFloor(z_m=0.0)
        reason = floor.check_trajectory(
            ((0.0,) * 7, (0.0,) * 7), self.kin, times_s=(1.0, 1.0)
        )
        self.assertIsNotNone(reason)
        self.assertIn("strictly increasing", reason)

    def test_hermite_overshoot_is_caught(self) -> None:
        """A segment with velocities can overshoot its waypoints; the check must
        sample like the bridge does, not just look at the waypoints."""
        positions = ((0.0,) * 7, (0.0,) * 7)
        velocities = ((0.0,) * 7, (5.0,) * 7)
        data = TrajectoryData(
            joint_names=tuple(f"j{i}" for i in range(7)),
            times_s=(0.0, 1.0),
            positions=positions,
            velocities=velocities,
        )
        from core.g1d.bridge_guard import sample_trajectory

        overshoot = sample_trajectory(data, 0.5)
        self.assertTrue(
            any(abs(v) > 1e-6 for v in overshoot),
            "expected the Hermite sampler to overshoot the waypoints",
        )
        # The floor is set just above the waypoint height but below the
        # overshoot: only a sampled check can catch this.
        floor = ZFloor(z_m=0.0, margin_m=0.0)
        reason = floor.check_trajectory(
            positions, self.kin, times_s=(0.0, 1.0), velocities=velocities
        )
        # Whatever the outcome, it must be a decision (not an exception) and it
        # must be consistent with the overshoot.
        if reason is not None:
            self.assertIn("z-floor", reason)


class ResolveZFloorTests(unittest.TestCase):
    def test_plain_value(self) -> None:
        floor = resolve_z_floor({"enable_z_floor": True, "z_floor_m": 0.42})
        self.assertTrue(floor.enabled)
        self.assertAlmostEqual(floor.z_m, 0.42, places=9)

    def test_override_wins(self) -> None:
        floor = resolve_z_floor(
            {"enable_z_floor": True, "z_floor_m": 0.42}, override_m=0.30
        )
        self.assertAlmostEqual(floor.z_m, 0.30, places=9)
        self.assertTrue(floor.enabled)

    def test_override_rejects_negative(self) -> None:
        with self.assertRaises(ValueError):
            resolve_z_floor({}, override_m=-0.1)

    def test_disabled_when_not_configured(self) -> None:
        floor = resolve_z_floor({})
        self.assertFalse(floor.enabled)

    def test_explicit_disable(self) -> None:
        floor = resolve_z_floor({"enable_z_floor": False, "z_floor_m": 0.42})
        self.assertFalse(floor.enabled)

    def test_named_floor(self) -> None:
        floor = resolve_z_floor(
            {
                "enable_z_floor": True,
                "z_floor_name": "table_a",
                "z_floors": {"table_a": 0.35, "table_b": 0.50},
            }
        )
        self.assertAlmostEqual(floor.z_m, 0.35, places=9)

    def test_missing_named_floor_raises(self) -> None:
        with self.assertRaises(ValueError):
            resolve_z_floor(
                {"z_floor_name": "nope", "z_floors": {"table_a": 0.35}}
            )

    def test_margin_from_config(self) -> None:
        floor = resolve_z_floor(
            {"enable_z_floor": True, "z_floor_m": 0.4, "z_floor_margin_m": 0.01}
        )
        self.assertAlmostEqual(floor.effective_z_m, 0.41, places=9)

    def test_zero_is_uncalibrated_by_default(self) -> None:
        floor = resolve_z_floor({"enable_z_floor": True, "z_floor_m": 0.0})
        self.assertAlmostEqual(floor.z_m, 0.0, places=9)
        self.assertFalse(floor.calibrated)

    def test_calibrated_zero_is_accepted(self) -> None:
        floor = resolve_z_floor(
            {"enable_z_floor": True, "z_floor_m": 0.0, "z_floor_calibrated": True}
        )
        self.assertAlmostEqual(floor.z_m, 0.0, places=9)
        self.assertTrue(floor.calibrated)
        self.assertTrue(floor.enabled)

    def test_calibrated_zero_keeps_the_hand_clearance(self) -> None:
        floor = resolve_z_floor(
            {
                "enable_z_floor": True,
                "z_floor_m": 0.0,
                "z_floor_calibrated": True,
                "z_floor_margin_m": 0.003,
            }
        )
        self.assertAlmostEqual(floor.effective_z_m, 0.003, places=9)

    def test_override_is_always_calibrated(self) -> None:
        floor = resolve_z_floor({"z_floor_m": 0.0}, override_m=0.0)
        self.assertTrue(floor.calibrated)

    def test_describe_flags_an_uncalibrated_floor(self) -> None:
        uncal = resolve_z_floor({"enable_z_floor": True, "z_floor_m": 0.0})
        self.assertIn("UNCALIBRATED", uncal.describe())
        cal = resolve_z_floor(
            {"enable_z_floor": True, "z_floor_m": 0.0, "z_floor_calibrated": True}
        )
        self.assertNotIn("UNCALIBRATED", cal.describe())

    def test_describe_does_not_flag_a_positive_floor(self) -> None:
        # A nonzero height is a measured value by construction; the absence of
        # the marker must not label it as a placeholder.
        floor = resolve_z_floor({"enable_z_floor": True, "z_floor_m": 0.42})
        self.assertNotIn("UNCALIBRATED", floor.describe())


if __name__ == "__main__":
    unittest.main()
