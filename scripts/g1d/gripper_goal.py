#!/usr/bin/env python3
"""Send one explicitly selected G1-D Dex1 gripper action through the ROS action server.

This client does not publish Unitree LowCmd. It sends no action unless --send is
provided and the operator types YES. Opening values use meters; the current
bridge's motor-to-width calibration remains provisional and must be checked.
"""
from __future__ import annotations

import argparse
import math
import sys
from typing import Sequence
from pathlib import Path

import rclpy
from action_msgs.msg import GoalStatus
from control_msgs.action import GripperCommand
from rclpy.action import ActionClient
from rclpy.node import Node

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.g1d.single_arm import SingleArmProfile


class GripperGoalNode(Node):
    def __init__(self, profile: SingleArmProfile) -> None:
        super().__init__("harness_g1d_gripper_goal")
        self._client = ActionClient(self, GripperCommand, profile.gripper_action)

    def send(self, opening_m: float) -> None:
        if not self._client.wait_for_server(timeout_sec=5.0):
            raise RuntimeError("selected gripper ActionServer is unavailable")
        goal = GripperCommand.Goal()
        goal.command.position = opening_m
        goal.command.max_effort = 0.0
        future = self._client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future, timeout_sec=8.0)
        if not future.done() or future.result() is None:
            raise RuntimeError("gripper goal was not acknowledged")
        handle = future.result()
        if not handle.accepted:
            raise RuntimeError("gripper ActionServer rejected the goal")
        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=30.0)
        if not result_future.done() or result_future.result() is None:
            handle.cancel_goal_async()
            raise RuntimeError("gripper result timed out; inspect the robot before another action")
        reply = result_future.result()
        if reply.status != GoalStatus.STATUS_SUCCEEDED:
            raise RuntimeError(f"gripper action ended with status {reply.status}")
        if reply.result.stalled or not reply.result.reached_goal:
            raise RuntimeError(
                f"gripper did not reach goal: stalled={reply.result.stalled}, "
                f"reached_goal={reply.result.reached_goal}"
            )
        print(f"Gripper action reports reached_goal at {reply.result.position:.4f} m.")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", required=True, choices=("left", "right"))
    parser.add_argument(
        "--opening-m", type=float, required=True,
        help="requested nominal opening in meters",
    )
    parser.add_argument(
        "--send", action="store_true",
        help="send after an interactive YES confirmation",
    )
    args = parser.parse_args(argv)
    if not math.isfinite(args.opening_m) or not 0.0 <= args.opening_m <= 0.09:
        parser.error("--opening-m must be finite and in the bridge's nominal [0, 0.09] m range")

    profile = SingleArmProfile(args.side)
    print(f"Selected gripper action: {profile.gripper_action}")
    print(f"Requested nominal opening: {args.opening_m:.4f} m")
    print(
        "The existing motor-to-width mapping is provisional; "
        "use a clear, unloaded gripper first."
    )
    if not args.send:
        print("NO ACTION SENT. Add --send to request an operator-confirmed action.")
        return 0
    if input("输入 YES 才发送夹爪目标：").strip() != "YES":
        print("已取消；没有发送夹爪目标。")
        return 0

    from core.g1d.ros_environment import require_g1d_ros_environment

    require_g1d_ros_environment()
    rclpy.init(args=None)
    node = GripperGoalNode(profile)
    try:
        node.send(args.opening_m)
        return 0
    except (RuntimeError, KeyboardInterrupt) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
