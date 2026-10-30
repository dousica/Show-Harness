"""Offline tests for the G1-D adapter contract; no ROS or hardware is used."""
from __future__ import annotations

import unittest

import numpy as np

from core.g1d import G1DRobotInterface, MockG1DTransport
from interpreters.real_atomic_controller import RealAtomicController


INITIAL_POSE = [0.3, 0.0, 0.4, 0.0, 0.0, 0.0, 1.0]


def _robot() -> tuple[G1DRobotInterface, MockG1DTransport]:
    transport = MockG1DTransport(
        initial_pose=INITIAL_POSE,
        open_width_m=0.08,
        closed_width_m=0.01,
    )
    return G1DRobotInterface(transport), transport


class G1DInterfaceTests(unittest.TestCase):
    def test_transport_maps_to_generic_robot_api(self) -> None:
        robot, transport = _robot()
        with self.assertRaisesRegex(RuntimeError, "before connect"):
            robot.get_ee_pose()

        robot.connect()
        np.testing.assert_allclose(robot.get_ee_pose(), INITIAL_POSE)
        self.assertAlmostEqual(robot.get_gripper_position()[0], 0.08)

        target = np.asarray(INITIAL_POSE, dtype=float)
        target[0] += 0.02
        target[3:] *= 2.0  # adapter normalizes command quaternions
        robot.update_desired_ee_pose(target)
        np.testing.assert_allclose(robot.get_ee_pose()[:3], [0.32, 0.0, 0.4])
        np.testing.assert_allclose(np.linalg.norm(robot.get_ee_pose()[3:]), 1.0)

        robot.control_gripper(True)
        self.assertAlmostEqual(robot.get_gripper_position()[0], 0.01)
        self.assertEqual(transport.gripper_commands, [True])
        robot.close()
        with self.assertRaisesRegex(RuntimeError, "before connect"):
            robot.get_ee_pose()

    def test_pose_and_gripper_feedback_are_validated(self) -> None:
        class BadTransport(MockG1DTransport):
            def read_end_effector_pose(self):
                return [0.0] * 6

            def read_gripper_width_m(self):
                return float("nan")

        transport = BadTransport(INITIAL_POSE, open_width_m=0.08, closed_width_m=0.01)
        robot = G1DRobotInterface(transport).connect()
        with self.assertRaisesRegex(ValueError, "7 values"):
            robot.get_ee_pose()

        class BadWidthTransport(MockG1DTransport):
            def read_gripper_width_m(self):
                return float("nan")

        robot = G1DRobotInterface(
            BadWidthTransport(INITIAL_POSE, open_width_m=0.08, closed_width_m=0.01)
        ).connect()
        with self.assertRaisesRegex(ValueError, "gripper width"):
            robot.get_gripper_position()

    def test_shared_atomic_controller_can_drive_mock_adapter(self) -> None:
        robot, transport = _robot()
        robot.connect()
        primitives = {
            "MV_FWD": [1.0, 0.0, 0.0],
            "MV_BACK": [-1.0, 0.0, 0.0],
            "MV_LEFT": [0.0, 1.0, 0.0],
            "MV_RIGHT": [0.0, -1.0, 0.0],
            "MV_UP": [0.0, 0.0, 1.0],
            "MV_DOWN": [0.0, 0.0, -1.0],
            "ROTATE_CW": -1.0,
            "ROTATE_CCW": 1.0,
        }
        controller = RealAtomicController(
            robot,
            primitives,
            step_m=0.02,
            yaw_step_rad=0.1,
            settle_steps=1,
            settle_dt_s=0.0,
            gripper_settle_s=0.0,
            gripper_min_settle_s=0.0,
            verbose=False,
        )

        controller.sync_from_robot()
        controller.step("MV_FWD")
        self.assertAlmostEqual(robot.get_ee_pose()[0], INITIAL_POSE[0] + 0.02)
        self.assertEqual(len(transport.pose_commands), 1)
        controller.step("GRASP")
        self.assertEqual(transport.gripper_commands, [True])
        self.assertAlmostEqual(robot.get_gripper_position()[0], 0.01)


if __name__ == "__main__":
    unittest.main()
