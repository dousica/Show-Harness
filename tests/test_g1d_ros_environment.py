from __future__ import annotations

import unittest

from core.g1d.ros_environment import g1d_ros_environment_errors


class G1DRosEnvironmentTests(unittest.TestCase):
    def _env(self, **updates: str) -> dict[str, str]:
        values = {
            "ROS_DOMAIN_ID": "42",
            "LZH_ROS_WS": "third_party/g1d_ros2_lzh",
            "AMENT_PREFIX_PATH": "third_party/g1d_ros2_lzh/install/lzh_bringup:/opt/ros/humble",
        }
        values.update(updates)
        return values

    def test_requires_domain_42(self) -> None:
        self.assertTrue(g1d_ros_environment_errors(self._env(ROS_DOMAIN_ID="")))
        self.assertTrue(g1d_ros_environment_errors(self._env(ROS_DOMAIN_ID="0")))
        self.assertEqual(g1d_ros_environment_errors(self._env()), [])

    def test_rejects_senior_overlay_in_ros_paths(self) -> None:
        for variable in (
            "AMENT_PREFIX_PATH", "CMAKE_PREFIX_PATH", "COLCON_PREFIX_PATH",
            "PYTHONPATH", "LD_LIBRARY_PATH",
        ):
            with self.subTest(variable=variable):
                errors = g1d_ros_environment_errors(
                    self._env(**{variable: "/home/yushu/g1d_ros2_ws/install/g1d_bringup"})
                )
                self.assertTrue(any("forbidden overlay" in error for error in errors))

    def test_requires_the_lzh_install_overlay(self) -> None:
        errors = g1d_ros_environment_errors(
            self._env(AMENT_PREFIX_PATH="/opt/ros/humble")
        )
        self.assertTrue(any("does not contain" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
