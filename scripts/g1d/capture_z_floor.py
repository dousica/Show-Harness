#!/usr/bin/env python3
"""Read (and optionally calibrate) the G1-D arm's Z safety floor.

The Z floor is the minimum height (in ``torso_link``) the end-effector may be
driven to. It is the guard that stops a Cartesian goal from pressing the arm
into the table -- the G1-D equivalent of the Franka/Piper ``z_floor_m``.

Two ways to use it:

1. **Capture at the tabletop.** Rest the gripper ON the table (the fingers
   touching), then run this script: it forward-kinematics the measured joints
   and reports the TCP height. That height is the floor. The arm never moves.

2. **Read a configured floor.** Pass ``--config`` to resolve the floor from a
   robot config and print it, plus the effective TCP limit (floor + hand
   clearance).

Nothing here sends a robot command. The arm is only ever READ.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.g1d.kinematics import ArmKinematics
from core.g1d.safety import resolve_z_floor


def _capture_height(side: str) -> float:
    """Read the measured joints and return the TCP height (no motion)."""
    import rclpy
    from sensor_msgs.msg import JointState
    from rclpy.qos import qos_profile_sensor_data
    from rclpy.node import Node

    from core.g1d.kinematics import arm_joint_names

    class _JointReader(Node):
        def __init__(self) -> None:
            super().__init__("harness_g1d_z_floor_reader")
            self.values: dict[str, float] = {}
            self.create_subscription(
                JointState, "/joint_states", self._cb, qos_profile_sensor_data
            )

        def _cb(self, msg: JointState) -> None:
            self.values = dict(zip(msg.name, msg.position))

    from core.g1d.ros_environment import require_g1d_ros_environment

    require_g1d_ros_environment()
    rclpy.init(args=None)
    node = _JointReader()
    wanted = arm_joint_names(side)
    try:
        deadline = None
        import time

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
            if all(name in node.values for name in wanted):
                break
        missing = [name for name in wanted if name not in node.values]
        if missing:
            raise RuntimeError(
                f"/joint_states is missing {missing}; is the SDK bridge's lowstate "
                "node running?"
            )
        joints = [float(node.values[name]) for name in wanted]
        tcp = ArmKinematics(side).tcp_matrix(joints)
        return float(tcp[2, 3])
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", choices=("left", "right"), default="left")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[2] / "configs" / "robot_g1d.yaml"),
        help="robot config to read the z-floor from",
    )
    parser.add_argument(
        "--capture",
        action="store_true",
        help="measure the current TCP height (rest the gripper on the table first)",
    )
    args = parser.parse_args(argv)

    if args.capture:
        try:
            height = _capture_height(args.side)
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        print(f"measured TCP height (torso_link): {height:.4f} m")
        print(
            f"use this as the floor: z_floor_m: {height:.4f}  "
            f"(the effective TCP limit adds the hand clearance on top)"
        )
        return 0

    cfg_path = Path(args.config)
    if not cfg_path.is_file():
        print(f"ERROR: {cfg_path} not found", file=sys.stderr)
        return 2
    import yaml

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    try:
        floor = resolve_z_floor(cfg)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"config: {cfg_path}")
    print(f"  {floor.describe()}")
    if not floor.enabled:
        print(
            "  WARNING: no floor is configured. A live G1-D run REFUSES to start "
            "without one -- capture it with --capture and set z_floor_m."
        )
        return 1
    if not math.isfinite(floor.z_m) or (floor.z_m <= 0.0 and not floor.calibrated):
        print(
            f"  WARNING: z_floor_m = {floor.z_m} looks like a placeholder. "
            "Recapture it on YOUR table before a live run."
        )
        return 1
    if floor.z_m <= 0.0:
        print(
            "  NOTE: z_floor_m = 0.0 is marked as CALIBRATED, so it is accepted as "
            "the measured table height (the tabletop is level with the torso_link "
            "Z origin)."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
