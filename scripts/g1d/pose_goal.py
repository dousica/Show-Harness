#!/usr/bin/env python3
"""Plan one single-arm G1-D end-effector pose through the existing MoveIt 2 stack.

The script only calls /plan_kinematic_path and /execute_trajectory. It never
publishes Unitree LowCmd. Planning is the default; real execution requires both
--execute and an interactive YES confirmation.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from typing import Sequence
from pathlib import Path

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import Pose, Quaternion, Vector3
from moveit_msgs.action import ExecuteTrajectory
from moveit_msgs.msg import (
    Constraints,
    DisplayTrajectory,
    MoveItErrorCodes,
    OrientationConstraint,
    PositionConstraint,
)
from moveit_msgs.srv import GetMotionPlan
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from tf2_ros import Buffer, TransformException, TransformListener

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.g1d.single_arm import SingleArmProfile
from core.g1d.vision_geometry import translate_pose_in_tool_axis


STATE_MAX_AGE_S = 0.5
START_DRIFT_LIMIT_RAD = 0.03
MAX_ARM_SPEED_RAD_S = 0.5
ALL_ARM_JOINT_NAMES = (
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)


def _pose_from_args(values: Sequence[float]) -> Pose:
    x, y, z, qx, qy, qz, qw = (float(value) for value in values)
    values = (x, y, z, qx, qy, qz, qw)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("target pose values must be finite")
    q_norm = math.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    if q_norm < 1e-8:
        raise ValueError("target quaternion must be non-zero")
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = x, y, z
    pose.orientation = Quaternion(x=qx/q_norm, y=qy/q_norm, z=qz/q_norm, w=qw/q_norm)
    return pose


def _rotation_distance(a: Quaternion, b: Quaternion) -> float:
    dot = abs(a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w)
    return 2.0 * math.acos(max(-1.0, min(1.0, dot)))


class PoseGoalNode(Node):
    def __init__(self, profile: SingleArmProfile, base_frame: str, tip_link: str) -> None:
        super().__init__("harness_g1d_pose_goal")
        self.profile = profile
        self.base_frame = base_frame
        self.tip_link = tip_link
        self._joint_positions: dict[str, float] = {}
        self._joint_rx_time: float | None = None
        self.create_subscription(
            JointState, "/joint_states", self._joint_state_cb, qos_profile_sensor_data
        )
        self._planner = self.create_client(GetMotionPlan, "/plan_kinematic_path")
        self._execute = ActionClient(self, ExecuteTrajectory, "/execute_trajectory")
        self._display = self.create_publisher(DisplayTrajectory, "/display_planned_path", 10)
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

    def _joint_state_cb(self, msg: JointState) -> None:
        values = dict(zip(msg.name, msg.position))
        full_state = {
            name: float(values[name])
            for name in ALL_ARM_JOINT_NAMES
            if name in values
        }
        if (len(full_state) == len(ALL_ARM_JOINT_NAMES) and
                all(math.isfinite(value) for value in full_state.values())):
            self._joint_positions = full_state
            self._joint_rx_time = time.monotonic()

    def fresh_arm_state(self, timeout_s: float = 3.0) -> dict[str, float]:
        deadline = time.monotonic() + timeout_s
        while rclpy.ok() and time.monotonic() < deadline:
            if (
                self._joint_rx_time is not None
                and time.monotonic() - self._joint_rx_time <= STATE_MAX_AGE_S
                and len(self._joint_positions) == len(ALL_ARM_JOINT_NAMES)
            ):
                return {name: self._joint_positions[name] for name in self.profile.joint_names}
            rclpy.spin_once(self, timeout_sec=0.05)
        raise RuntimeError("no fresh /joint_states for the selected arm's 7 joints")

    def current_tip_pose(self, timeout_s: float = 3.0) -> Pose:
        deadline = time.monotonic() + timeout_s
        while rclpy.ok() and time.monotonic() < deadline:
            try:
                tf = self._tf_buffer.lookup_transform(
                    self.base_frame,
                    self.tip_link,
                    rclpy.time.Time(),
                    timeout=Duration(seconds=0.1),
                )
                pose = Pose()
                pose.position.x = tf.transform.translation.x
                pose.position.y = tf.transform.translation.y
                pose.position.z = tf.transform.translation.z
                pose.orientation = tf.transform.rotation
                return pose
            except TransformException:
                rclpy.spin_once(self, timeout_sec=0.05)
        raise RuntimeError(
            f"TF {self.base_frame} -> {self.tip_link} unavailable; check URDF and robot state"
        )

    def make_request(
        self,
        start: dict[str, float],
        target: Pose,
        *,
        position_tolerance_m: float,
        orientation_tolerance_rad: float,
        velocity_scaling: float,
    ) -> GetMotionPlan.Request:
        request = GetMotionPlan.Request()
        motion = request.motion_plan_request
        motion.group_name = self.profile.group_name
        motion.num_planning_attempts = 5
        motion.allowed_planning_time = 8.0
        motion.max_velocity_scaling_factor = velocity_scaling
        motion.max_acceleration_scaling_factor = velocity_scaling
        motion.start_state.is_diff = True
        motion.start_state.joint_state.name = list(self.profile.joint_names)
        motion.start_state.joint_state.position = [start[name] for name in self.profile.joint_names]

        goal = Constraints()
        goal.name = f"{self.profile.side}_arm_pose_goal"
        position = PositionConstraint()
        position.header.frame_id = self.base_frame
        position.link_name = self.tip_link
        position.target_point_offset = Vector3()
        sphere = SolidPrimitive()
        sphere.type = SolidPrimitive.SPHERE
        sphere.dimensions = [position_tolerance_m]
        position.constraint_region.primitives.append(sphere)
        region_pose = Pose()
        region_pose.position = target.position
        region_pose.orientation.w = 1.0
        position.constraint_region.primitive_poses.append(region_pose)
        position.weight = 1.0
        goal.position_constraints.append(position)

        orientation = OrientationConstraint()
        orientation.header.frame_id = self.base_frame
        orientation.link_name = self.tip_link
        orientation.orientation = target.orientation
        orientation.absolute_x_axis_tolerance = orientation_tolerance_rad
        orientation.absolute_y_axis_tolerance = orientation_tolerance_rad
        orientation.absolute_z_axis_tolerance = orientation_tolerance_rad
        orientation.weight = 1.0
        goal.orientation_constraints.append(orientation)
        motion.goal_constraints.append(goal)
        return request

    def plan(self, request: GetMotionPlan.Request, start: dict[str, float]):
        if not self._planner.wait_for_service(timeout_sec=5.0):
            raise RuntimeError("MoveIt service /plan_kinematic_path is unavailable")
        future = self._planner.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=18.0)
        if not future.done() or future.result() is None:
            raise RuntimeError("MoveIt planning service timed out")
        response = future.result().motion_plan_response
        if response.error_code.val != MoveItErrorCodes.SUCCESS:
            raise RuntimeError(f"MoveIt failed to plan (error_code={response.error_code.val})")
        trajectory = response.trajectory
        self.profile.validate_trajectory_joint_names(trajectory.joint_trajectory.joint_names)
        if not trajectory.joint_trajectory.points:
            raise RuntimeError("MoveIt returned an empty trajectory")
        self._validate_trajectory(trajectory.joint_trajectory, start)
        display = DisplayTrajectory()
        display.trajectory_start = response.trajectory_start
        display.trajectory.append(trajectory)
        self._display.publish(display)
        return trajectory

    def _validate_trajectory(self, path, start: dict[str, float]) -> None:
        names = list(path.joint_names)
        self.profile.validate_trajectory_joint_names(names)
        first = path.points[0]
        index = {name: names.index(name) for name in self.profile.joint_names}
        for name in self.profile.joint_names:
            if abs(first.positions[index[name]] - start[name]) > START_DRIFT_LIMIT_RAD:
                raise RuntimeError(f"trajectory start for {name} differs from measured state")
        previous_time = -1.0
        for point in path.points:
            t = float(point.time_from_start.sec) + 1e-9 * float(point.time_from_start.nanosec)
            if not math.isfinite(t) or t <= previous_time:
                raise RuntimeError("trajectory timestamps must be finite and strictly increasing")
            if len(point.positions) != 7 or not all(math.isfinite(q) for q in point.positions):
                raise RuntimeError("trajectory has invalid joint positions")
            if point.velocities:
                if len(point.velocities) != 7 or not all(
                    math.isfinite(v) for v in point.velocities
                ):
                    raise RuntimeError("trajectory has invalid joint velocities")
                if any(abs(v) > MAX_ARM_SPEED_RAD_S for v in point.velocities):
                    raise RuntimeError("trajectory exceeds the configured 0.5 rad/s arm guard")
            previous_time = t

    def execute_after_confirmation(self, trajectory, start: dict[str, float]) -> None:
        print("即将执行 MoveIt 已规划的单臂轨迹。确认周围无人、路径无障碍、急停可用。")
        if input("输入 YES 才发送执行目标：").strip() != "YES":
            print("已取消；没有发送 ExecuteTrajectory。")
            return
        fresh = self.fresh_arm_state()
        drift = max(abs(fresh[name] - start[name]) for name in self.profile.joint_names)
        if drift > START_DRIFT_LIMIT_RAD:
            raise RuntimeError(f"arm moved {drift:.4f} rad after planning; replan before execution")
        if not self._execute.wait_for_server(timeout_sec=5.0):
            raise RuntimeError("MoveIt action /execute_trajectory is unavailable")
        goal = ExecuteTrajectory.Goal()
        goal.trajectory = trajectory
        send_future = self._execute.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future, timeout_sec=8.0)
        if not send_future.done() or send_future.result() is None:
            raise RuntimeError("MoveIt did not acknowledge the execution goal")
        handle = send_future.result()
        if not handle.accepted:
            raise RuntimeError("MoveIt rejected the execution goal")
        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=90.0)
        if not result_future.done() or result_future.result() is None:
            handle.cancel_goal_async()
            raise RuntimeError(
                "execution result timed out; inspect the robot before any next action"
            )
        result = result_future.result()
        if result.status != GoalStatus.STATUS_SUCCEEDED:
            raise RuntimeError(f"MoveIt execution ended with action status {result.status}")
        if result.result.error_code.val != MoveItErrorCodes.SUCCESS:
            raise RuntimeError(
                f"MoveIt execution failed (error_code={result.result.error_code.val})"
            )
        print("MoveIt reports the selected-arm trajectory completed successfully.")


def _parse_pose(text: str) -> list[float]:
    try:
        values = [float(value.strip()) for value in text.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("pose must be comma-separated numeric values") from exc
    if len(values) != 7:
        raise argparse.ArgumentTypeError("pose needs x,y,z,qx,qy,qz,qw")
    return values


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", required=True, choices=("left", "right"))
    parser.add_argument("--base-frame", default="torso_link")
    parser.add_argument(
        "--tip-link", required=True,
        help="MoveIt robot link to target; use a verified TCP link if one exists",
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--pose", type=_parse_pose, help="x,y,z,qx,qy,qz,qw in meters/xyzw")
    target.add_argument(
        "--step-tool-x", type=float,
        help="small translation along current tool +X, meters",
    )
    target.add_argument(
        "--step-base-z", type=float,
        help="small translation along base-frame +Z, meters",
    )
    parser.add_argument("--max-translation-m", type=float, default=0.01)
    parser.add_argument("--position-tolerance-m", type=float, default=0.01)
    parser.add_argument("--orientation-tolerance-rad", type=float, default=0.08)
    parser.add_argument("--max-rotation-rad", type=float, default=0.10)
    parser.add_argument("--velocity-scaling", type=float, default=0.01)
    parser.add_argument(
        "--execute", action="store_true",
        help="ask for YES, then call MoveIt ExecuteTrajectory",
    )
    args = parser.parse_args(argv)

    if not math.isfinite(args.max_translation_m) or not 0 < args.max_translation_m <= 0.05:
        parser.error("--max-translation-m must be in (0, 0.05] meters")
    if not 0 < args.position_tolerance_m <= 0.05:
        parser.error("--position-tolerance-m must be in (0, 0.05] meters")
    if not 0 < args.orientation_tolerance_rad <= 0.5:
        parser.error("--orientation-tolerance-rad must be in (0, 0.5] radians")
    if not 0 < args.max_rotation_rad <= 0.10:
        parser.error("--max-rotation-rad must be in (0, 0.10] radians")
    if not 0 < args.velocity_scaling <= 0.1:
        parser.error("--velocity-scaling must be in (0, 0.1]")
    for name, value in (("--step-tool-x", args.step_tool_x), ("--step-base-z", args.step_base_z)):
        if value is not None and abs(value) > args.max_translation_m:
            parser.error(f"{name} exceeds --max-translation-m")

    profile = SingleArmProfile(args.side)
    from core.g1d.ros_environment import require_g1d_ros_environment

    require_g1d_ros_environment()
    rclpy.init(args=None)
    node = PoseGoalNode(profile, args.base_frame, args.tip_link)
    try:
        start = node.fresh_arm_state()
        current_pose = node.current_tip_pose()
        if args.pose is not None:
            target_pose = _pose_from_args(args.pose)
        else:
            current = [
                current_pose.position.x, current_pose.position.y, current_pose.position.z,
                current_pose.orientation.x, current_pose.orientation.y,
                current_pose.orientation.z, current_pose.orientation.w,
            ]
            if args.step_base_z is not None:
                values = list(current)
                values[2] += args.step_base_z
            else:
                values = translate_pose_in_tool_axis(
                    current, args.step_tool_x, axis="x", max_step_m=args.max_translation_m
                )
            target_pose = _pose_from_args(values)
        translation = math.dist(
            (current_pose.position.x, current_pose.position.y, current_pose.position.z),
            (target_pose.position.x, target_pose.position.y, target_pose.position.z),
        )
        if translation > args.max_translation_m + 1e-9:
            raise ValueError(
                f"requested translation {translation:.4f} m exceeds limit "
                f"{args.max_translation_m:.4f} m"
            )
        rotation = _rotation_distance(current_pose.orientation, target_pose.orientation)
        if rotation > args.max_rotation_rad + 1e-9:
            raise ValueError(
                f"requested rotation {rotation:.4f} rad exceeds limit "
                f"{args.max_rotation_rad:.4f} rad"
            )
        print(f"Selected group: {profile.group_name}; joints: {', '.join(profile.joint_names)}")
        print(f"Gripper action (not called by this script): {profile.gripper_action}")
        print(
            f"Target link: {args.tip_link}; translation={translation:.4f} m, "
            f"rotation={rotation:.4f} rad"
        )
        print(f"Target pose in {args.base_frame}: {target_pose}")
        request = node.make_request(
            start,
            target_pose,
            position_tolerance_m=args.position_tolerance_m,
            orientation_tolerance_rad=args.orientation_tolerance_rad,
            velocity_scaling=args.velocity_scaling,
        )
        trajectory = node.plan(request, start)
        print(
            f"Plan accepted: {len(trajectory.joint_trajectory.points)} points; "
            f"duration={trajectory.joint_trajectory.points[-1].time_from_start.sec}"
            f".{trajectory.joint_trajectory.points[-1].time_from_start.nanosec:09d}s"
        )
        if args.execute:
            node.execute_after_confirmation(trajectory, start)
        else:
            print(
                "PLAN ONLY: no ExecuteTrajectory request was sent. "
                "Inspect /display_planned_path in RViz."
            )
        return 0
    except (RuntimeError, ValueError, KeyboardInterrupt) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
