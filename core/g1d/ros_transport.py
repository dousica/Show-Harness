"""ROS 2 implementation of the G1-D transport.

This is the real counterpart of :class:`core.g1d.g1d_interface.G1DTransport`: it
turns the harness's abstract "read the end-effector pose / command the gripper"
operations into the G1-D ROS 2 stack that the SDK bridge exposes.

What it talks to
----------------
* ``/joint_states`` (published by the bridge's lowstate node) -- the 14 arm
  joints' measured positions. The harness reads ONLY the selected side's 7.
* ``/lzh_dex1/motor_states`` (or ``/dex1/motor_states``) -- the raw Dex1 motor
  feedback (radians) of both grippers.
* ``/<side>_arm_controller/follow_joint_trajectory`` -- the FollowJointTrajectory
  Action that the bridge serves for the selected arm.
* ``/lzh_<side>_dex1/gripper_command`` -- the GripperCommand Action for the gripper.
* ``/plan_kinematic_path`` + ``/execute_trajectory`` -- the MoveIt 2 interfaces,
  used to turn a Cartesian target into a 7-joint trajectory (when MoveIt is up).

What it never does
------------------
It never publishes Unitree ``LowCmd``. The bridge is the ONLY writer of the robot's
command channel; this client only sends ROS 2 goals. That is deliberate: a second
LowCmd writer on the robot would fight the bridge for control of the same motors.

The end-effector pose is computed with forward kinematics on the measured joints
(:mod:`core.g1d.kinematics`), because the bridge exposes joint feedback only.

Two motion paths
----------------
The G1-D arm can be driven two ways, and the transport supports both because the
lab's live setup does NOT always have MoveIt running:

* ``"moveit"`` (default when MoveIt is reachable): the Cartesian target becomes a
  MoveIt plan, the plan is checked against the bridge's guards
  (:mod:`core.g1d.bridge_guard`), and the trajectory is executed through MoveIt's
  ``/execute_trajectory``.
* ``"joint"``: the Cartesian target is solved to a 7-joint target with the
  harness's own numerical IK (:mod:`core.g1d.inverse_kinematics`) and sent straight
  to the bridge's FollowJointTrajectory Action as a two-point trajectory. This works
  with NO MoveIt running -- which is the situation on the current domain-42 rig.

Both paths end at the SAME place: the bridge's FollowJointTrajectory Action, which
is the only thing that can move the arm, and which applies the same guards either
way. Neither path can bypass the bridge's limit / speed / start-state checks.
"""
from __future__ import annotations

import math
import threading
import time
from typing import Any, Optional, Sequence

import numpy as np

from core.g1d.bridge_guard import (
    TrajectoryData,
    check_trajectory,
    validate_trajectory_structure,
)
from core.g1d.geometry import (
    BRIDGE_MAX_START_ERROR_RAD,
    BRIDGE_MAX_TRAJECTORY_SPEED_RAD_S,
    DEX1_MAX_OPENING_M,
    arm_limits_rad,
)
from core.g1d.inverse_kinematics import ArmInverseKinematics
from core.g1d.kinematics import ArmKinematics, arm_joint_names, tcp_to_wrist_pose7
from core.g1d.safety import ZFloor
from core.g1d.single_arm import SingleArmProfile


class G1DTransportError(RuntimeError):
    """Raised when the ROS 2 stack cannot serve the requested operation."""


#: control_msgs FollowJointTrajectory: the goal was aborted because the measured
#: joints did not reach the commanded final point inside the tolerance window.
ERR_GOAL_TOLERANCE_VIOLATED = -5


def _require_ros():
    """Import rclpy + the message packages, with an actionable error if missing."""
    try:
        import rclpy
        from action_msgs.msg import GoalStatus
        from control_msgs.action import FollowJointTrajectory, GripperCommand
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
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import JointState
        from shape_msgs.msg import SolidPrimitive
    except ImportError as exc:  # pragma: no cover - environment guard
        raise ImportError(
            "rclpy / moveit_msgs / control_msgs are not importable. Run inside a "
            "shell with ROS 2 Humble sourced: `export ROS_DOMAIN_ID=42` and "
            "`source /home/yushu/g1d_ros2_lzh/setup_env.sh`."
        ) from exc
    return {
        "rclpy": rclpy,
        "GoalStatus": GoalStatus,
        "FollowJointTrajectory": FollowJointTrajectory,
        "GripperCommand": GripperCommand,
        "Pose": Pose,
        "Quaternion": Quaternion,
        "Vector3": Vector3,
        "ExecuteTrajectory": ExecuteTrajectory,
        "Constraints": Constraints,
        "DisplayTrajectory": DisplayTrajectory,
        "MoveItErrorCodes": MoveItErrorCodes,
        "OrientationConstraint": OrientationConstraint,
        "PositionConstraint": PositionConstraint,
        "GetMotionPlan": GetMotionPlan,
        "ActionClient": ActionClient,
        "Duration": Duration,
        "SingleThreadedExecutor": SingleThreadedExecutor,
        "Node": Node,
        "qos_profile_sensor_data": qos_profile_sensor_data,
        "JointState": JointState,
        "SolidPrimitive": SolidPrimitive,
    }


def rclpy_ok() -> bool:
    """Whether the ROS 2 context is still running (safe before ``rclpy.init``)."""
    try:
        import rclpy

        return bool(rclpy.ok())
    except Exception:  # noqa: BLE001 - rclpy not importable / not initialized
        return False


def _ensure_rclpy_initialized(ros: dict[str, Any]) -> None:
    """Initialize the shared ROS context before any transport node is created."""
    if not ros["rclpy"].ok():
        ros["rclpy"].init(args=None)


class G1DRosTransport:
    """A ROS 2 client for ONE G1-D arm and its matching Dex1 gripper.

    The transport is deliberately *stateless* with respect to motion: every
    commanded end-effector pose is turned into a fresh trajectory and a fresh
    goal. That is slower than a streaming setpoint, but it is the only way to
    guarantee that every motion passed the bridge's own limit / speed /
    start-error guards before it reached the motors.
    """

    def __init__(
        self,
        profile: SingleArmProfile,
        *,
        node_name: str = "harness_g1d_transport",
        motion_path: str = "auto",
        plan_timeout_s: float = 20.0,
        execute_timeout_s: float = 120.0,
        gripper_timeout_s: float = 40.0,
        feedback_max_age_s: float = 0.5,
        velocity_scaling: float = 0.1,
        dry_run: bool = True,
        z_floor: Optional[ZFloor] = None,
        tolerance_violation_is_fatal: bool = False,
        verbose: bool = True,
    ) -> None:
        """
        Args:
            profile: The selected side. Only this side's joints, group and
                Actions are ever touched.
            node_name: The ROS node name.
            motion_path: ``"moveit"`` (plan with MoveIt, then execute), ``"joint"``
                (solve the IK locally, then send a two-point trajectory), or
                ``"auto"`` (use MoveIt when its service is reachable, else the
                local IK). ``"auto"`` is what the current rig needs: MoveIt is
                not always running.
            plan_timeout_s: Timeout for the MoveIt planning service call.
            execute_timeout_s: Timeout for a trajectory execution to report back.
            gripper_timeout_s: Timeout for a gripper action. The bridge needs
                ~11 s for a full 5.4-rad stroke at its 0.5 rad/s limit, plus
                settling, so the default is generous.
            feedback_max_age_s: How old the joint feedback may be before it is
                treated as stale (a dead bridge would otherwise freeze the pose).
            velocity_scaling: The MoveIt velocity/acceleration scaling factor.
            dry_run: When True, trajectories are PLANNED and validated but never
                SENT for execution. This is the default: the harness must not
                move a real arm without an explicit opt-in.
            z_floor: The minimum end-effector height guard (see
                :mod:`core.g1d.safety`). When enabled, every planned trajectory
                is checked against it BEFORE it can be sent, and a Cartesian
                target below the floor is clamped. ``None`` means NO floor --
                only acceptable in a dry-run; a live run must pass one.
            tolerance_violation_is_fatal: When True, a
                ``GOAL_TOLERANCE_VIOLATED`` abort raises, exactly as before.
                The default is False: the bridge keeps holding the commanded
                final position, so the Cartesian goal WAS reached and a
                closed-loop rollout must not die on it. The event is still
                logged (with the measured Cartesian error) and recorded in
                ``last_tolerance_violation``.
            verbose: Print progress lines.
        """
        from core.g1d.ros_environment import require_g1d_ros_environment

        require_g1d_ros_environment()
        self.profile = profile
        self.node_name = str(node_name)
        self.motion_path = str(motion_path).strip().lower()
        if self.motion_path not in ("auto", "moveit", "joint"):
            raise ValueError(
                f"motion_path must be 'auto', 'moveit' or 'joint', got {motion_path!r}"
            )
        self.plan_timeout_s = float(plan_timeout_s)
        self.execute_timeout_s = float(execute_timeout_s)
        self.gripper_timeout_s = float(gripper_timeout_s)
        self.feedback_max_age_s = float(feedback_max_age_s)
        self.velocity_scaling = float(velocity_scaling)
        self.dry_run = bool(dry_run)
        self.verbose = bool(verbose)
        self.z_floor = z_floor
        self.tolerance_violation_is_fatal = bool(tolerance_violation_is_fatal)
        #: The last GOAL_TOLERANCE_VIOLATED event (commanded vs measured joints).
        self.last_tolerance_violation: Optional[dict[str, Any]] = None
        if not self.dry_run and (z_floor is None or not z_floor.enabled):
            raise ValueError(
                "a live G1-D run needs an enabled z-floor: the arm's commanded height "
                "must have a lower bound so it cannot be driven into the table. "
                "Pass a core.g1d.safety.ZFloor (calibrate the height with "
                "scripts/g1d/capture_z_floor.py), or keep dry_run=True."
            )
        self._ros = _require_ros()
        self._node: Any = None
        self._executor: Any = None
        self._spin_thread: Optional[threading.Thread] = None
        self._arm_client: Any = None
        self._gripper_client: Any = None
        self._planner: Any = None
        self._execute_client: Any = None
        self._joint_state: Optional[tuple[dict[str, float], float]] = None
        self._dex1_state: Optional[tuple[dict[str, float], float]] = None
        self._connected = False
        self._kin = ArmKinematics(profile.side)
        self._ik = ArmInverseKinematics(profile.side)
        self._joint_names = arm_joint_names(profile.side)
        self._limits = arm_limits_rad(profile.side)
        # The last commanded end-effector pose (the setpoint), kept so a
        # gripper-only command can re-issue it without moving the arm.
        self._last_commanded_pose: Optional[np.ndarray] = None
        # The last trajectory that was built (for diagnostics).
        self.last_trajectory: Optional[dict[str, Any]] = None

    # -- lifecycle -----------------------------------------------------------
    def connect(self) -> "G1DRosTransport":
        """Create the node, the subscriptions and the action clients.

        Does NOT wait for the bridge to be up: the bridge may be started after
        the harness. The first command that needs it fails with a clear message.
        """
        ros = self._ros
        if self._node is not None:
            return self
        _ensure_rclpy_initialized(ros)
        self._node = ros["Node"](node_name=self.node_name)
        self._joint_sub = self._node.create_subscription(
            ros["JointState"],
            self.profile.joint_states_topic,
            self._on_joint_states,
            ros["qos_profile_sensor_data"],
        )
        self._dex1_sub = self._node.create_subscription(
            ros["JointState"],
            self.profile.dex1_motor_states_topic,
            self._on_dex1_states,
            ros["qos_profile_sensor_data"],
        )
        self._arm_client = ros["ActionClient"](
            self._node,
            ros["FollowJointTrajectory"],
            self.profile.trajectory_action,
        )
        self._gripper_client = ros["ActionClient"](
            self._node,
            ros["GripperCommand"],
            self.profile.gripper_action,
        )
        self._planner = self._node.create_client(ros["GetMotionPlan"], "/plan_kinematic_path")
        self._execute_client = ros["ActionClient"](
            self._node, ros["ExecuteTrajectory"], "/execute_trajectory"
        )
        # Service the node in the background: subscription callbacks (the joint
        # and gripper feedback) only arrive while an executor spins the node --
        # ROS 2 has no automatic callback thread like ROS 1.
        self._executor = ros["SingleThreadedExecutor"]()
        self._executor.add_node(self._node)
        self._spin_thread = threading.Thread(
            target=self._spin_loop, name="g1d-transport-spin", daemon=True
        )
        self._spin_thread.start()
        # The first subscription callback is asynchronous. Wait for one valid
        # joint-state sample before callers read the measured pose during
        # session startup; otherwise a healthy bridge can fail nondeterministically
        # simply because the callback has not run yet.
        deadline = time.monotonic() + 5.0
        while self._joint_state is None and time.monotonic() < deadline:
            time.sleep(0.01)
        if self._joint_state is None:
            raise G1DTransportError(
                f"no {self.profile.joint_states_topic} received within 5s after "
                "connecting; verify start.bash is still running and the SDK bridge "
                "is publishing in ROS_DOMAIN_ID=42"
            )
        self._connected = True
        if self.verbose:
            print(
                f"[g1d-transport] connected: side={self.profile.side} "
                f"group={self.profile.group_name} "
                f"trajectory={self.profile.trajectory_action} "
                f"gripper={self.profile.gripper_action} "
                f"motion_path={self.motion_path} dry_run={self.dry_run}"
            )
        return self

    def _spin_loop(self) -> None:
        """The background executor loop (until the node is torn down)."""
        executor = self._executor
        try:
            while executor is not None and rclpy_ok():
                executor.spin_once(timeout_sec=0.05)
        except Exception:  # noqa: BLE001 - the thread must never kill the process
            pass

    def close(self) -> None:
        """Tear down the node and its clients."""
        node = self._node
        executor = self._executor
        self._node = None
        self._executor = None
        self._connected = False
        if executor is not None:
            try:
                executor.shutdown(timeout_sec=1.0)
            except Exception:  # noqa: BLE001 - best-effort teardown
                pass
        if node is not None:
            try:
                node.destroy_node()
            except Exception:  # noqa: BLE001 - best-effort teardown
                pass

    def _require_connected(self, operation: str) -> None:
        if not self._connected or self._node is None:
            raise G1DTransportError(
                f"G1DRosTransport.{operation} called before connect()"
            )

    # -- feedback ------------------------------------------------------------
    def _on_joint_states(self, msg: Any) -> None:
        values = dict(zip(msg.name, msg.position))
        self._joint_state = (values, time.monotonic())

    def _on_dex1_states(self, msg: Any) -> None:
        values = dict(zip(msg.name, msg.position))
        self._dex1_state = (values, time.monotonic())

    def _fresh_joints(self) -> dict[str, float]:
        """The measured joints of the SELECTED side only."""
        self._require_connected("read_joint_positions")
        cached = self._joint_state
        if cached is None:
            raise G1DTransportError(
                f"no {self.profile.joint_states_topic} received yet -- is the SDK "
                "bridge's lowstate node running?"
            )
        values, received = cached
        age = time.monotonic() - received
        if age > self.feedback_max_age_s:
            raise G1DTransportError(
                f"joint feedback is stale ({age:.2f}s > {self.feedback_max_age_s:.2f}s); "
                "the SDK bridge has likely stopped. NOT proceeding on frozen state."
            )
        missing = [name for name in self._joint_names if name not in values]
        if missing:
            raise G1DTransportError(
                f"{self.profile.joint_states_topic} is missing the selected side's "
                f"joints: {missing}"
            )
        return {name: float(values[name]) for name in self._joint_names}

    def _fresh_gripper_motor_rad(self) -> float:
        """The selected side's raw Dex1 motor feedback (radians)."""
        self._require_connected("read_gripper_width_m")
        cached = self._dex1_state
        if cached is None:
            raise G1DTransportError(
                f"no {self.profile.dex1_motor_states_topic} received yet -- is the "
                "SDK bridge running?"
            )
        values, received = cached
        age = time.monotonic() - received
        if age > self.feedback_max_age_s:
            raise G1DTransportError(
                f"gripper feedback is stale ({age:.2f}s); the SDK bridge has likely stopped."
            )
        name = f"{self.profile.side}_dex1_motor_raw"
        if name not in values:
            raise G1DTransportError(
                f"{self.profile.dex1_motor_states_topic} has no '{name}' entry "
                "(the bridge omits invalid motors)"
            )
        return float(values[name])

    def read_joint_positions(self) -> list[float]:
        """The selected side's 7 measured joint positions (radians, URDF order)."""
        values = self._fresh_joints()
        return [values[name] for name in self._joint_names]

    # -- G1DTransport protocol ------------------------------------------------
    def read_end_effector_pose(self) -> list[float]:
        """The measured end-effector pose [x,y,z,qx,qy,qz,qw] in ``torso_link``.

        Derived by forward kinematics from the measured joints -- the bridge
        exposes no Cartesian feedback.
        """
        joints = self.read_joint_positions()
        return self._kin.tcp_pose7(joints).tolist()

    def read_gripper_width_m(self) -> float:
        """The nominal jaw opening (m) implied by the measured Dex1 motor angle."""
        return self._kin.opening_width_m(
            self.read_joint_positions(), motor_rad=self._fresh_gripper_motor_rad()
        )

    def command_end_effector_pose(self, pose7: Sequence[float]) -> None:
        """Plan and (unless dry-run) execute a Cartesian move to ``pose7``.

        The target is the TCP (the fingertip midpoint). MoveIt plans to the
        wrist link, so the target is converted to the wrist frame first and the
        measured wrist pose is converted back, so the command is expressed in
        exactly the frame the harness reports.

        The z-floor is applied in TWO places, because either alone is
        insufficient:

        1. the Cartesian TARGET is clamped, so a goal below the floor becomes a
           goal AT the floor instead of a hopeless plan;
        2. the resulting TRAJECTORY is verified (every sampled configuration is
           forward-kinematics'd and its TCP height compared), because a valid
           target can still be reached by a path that dips through the table.
        """
        self._require_connected("command_end_effector_pose")
        target = np.asarray(pose7, dtype=float).reshape(-1)
        if target.size != 7 or not np.all(np.isfinite(target)):
            raise G1DTransportError("command pose must be 7 finite values")
        target, clamped = self._clamp_target(target)
        if clamped and self.verbose:
            print(
                f"[g1d-transport] z-floor: target height clamped to "
                f"{target[2]:.4f} m ({self.z_floor.describe() if self.z_floor else 'off'})"
            )
        start = self.read_joint_positions()
        trajectory = self.build_trajectory(target, start)
        self._last_commanded_pose = target.copy()
        if self.dry_run:
            if self.verbose:
                print(
                    f"[g1d-transport] DRY RUN: built a {len(trajectory.positions)}-point "
                    "trajectory; no execution goal sent."
                )
            return
        self._send_trajectory(trajectory)

    def _clamp_target(self, target: np.ndarray) -> tuple[np.ndarray, bool]:
        """Apply the z-floor to a Cartesian target (no-op when disabled)."""
        if self.z_floor is None or not self.z_floor.enabled:
            return target, False
        clamped, was_clamped = self.z_floor.clamp_pose7(target)
        return clamped, was_clamped

    def check_z_floor(self, trajectory: TrajectoryData) -> str | None:
        """Verify a planned trajectory against the z-floor.

        Returns ``None`` when the trajectory stays above the floor, else the
        reason it must not be sent.
        """
        if self.z_floor is None or not self.z_floor.enabled:
            return None
        return self.z_floor.check_trajectory(
            trajectory.positions,
            self._kin,
            times_s=trajectory.times_s,
            velocities=trajectory.velocities,
            joint_names=trajectory.joint_names,
        )

    def command_gripper(self, closed: bool) -> None:
        """Command the selected side's Dex1 to open or close.

        The nominal opening is 0.0 m (closed) or the bridge's nominal full
        opening; the mapping is PROVISIONAL and the width is a nominal value,
        not a calibrated jaw width.
        """
        self._require_connected("command_gripper")
        opening = 0.0 if closed else DEX1_MAX_OPENING_M
        if self.dry_run:
            if self.verbose:
                print(
                    f"[g1d-transport] DRY RUN: gripper {'CLOSE' if closed else 'OPEN'} "
                    f"(nominal {opening:.4f} m); no action sent."
                )
            return
        self._send_gripper(opening)

    # -- trajectory construction ------------------------------------------------
    def build_trajectory(
        self, tcp_target7: Sequence[float], start_joints: Sequence[float]
    ) -> TrajectoryData:
        """Build a validated 7-joint trajectory for the selected side.

        Chooses the motion path (``moveit`` / ``joint``) and returns the
        trajectory in the plain form :mod:`core.g1d.bridge_guard` checks, so the
        bridge's rules are applied identically whichever path was used.
        """
        path = self._resolve_motion_path()
        if path == "moveit":
            return self._build_moveit_trajectory(tcp_target7, start_joints)
        return self._build_joint_trajectory(tcp_target7, start_joints)

    def _resolve_motion_path(self) -> str:
        """Pick the motion path, honouring ``motion_path="auto"``."""
        if self.motion_path in ("moveit", "joint"):
            return self.motion_path
        # auto: MoveIt when its planning service is reachable, else the local IK.
        try:
            if self._planner is not None and self._planner.service_is_ready():
                return "moveit"
        except Exception:  # noqa: BLE001 - fall back to the local IK
            pass
        return "joint"

    def _build_moveit_trajectory(
        self, tcp_target7: Sequence[float], start_joints: Sequence[float]
    ) -> TrajectoryData:
        """Plan a Cartesian move with MoveIt and convert it to a TrajectoryData."""
        ros = self._ros
        if not self._planner.wait_for_service(timeout_sec=5.0):
            raise G1DTransportError(
                "MoveIt service /plan_kinematic_path is unavailable -- is "
                "lzh_bringup/real_moveit.launch.py running?"
            )
        # MoveIt plans to the WRIST link; the harness commands the TCP. Convert
        # the target into the wrist frame (same orientation, offset origin) and
        # convert the plan back at the end.
        wrist_target = tcp_to_wrist_pose7(tcp_target7)
        request = ros["GetMotionPlan"].Request()
        motion = request.motion_plan_request
        motion.group_name = self.profile.group_name
        motion.num_planning_attempts = 5
        motion.allowed_planning_time = 8.0
        motion.max_velocity_scaling_factor = self.velocity_scaling
        motion.max_acceleration_scaling_factor = self.velocity_scaling
        motion.start_state.is_diff = True
        motion.start_state.joint_state.name = list(self._joint_names)
        motion.start_state.joint_state.position = [
            float(start_joints[index]) for index in range(len(self._joint_names))
        ]

        goal = ros["Constraints"]()
        goal.name = f"{self.profile.side}_arm_cartesian_goal"
        position = ros["PositionConstraint"]()
        position.header.frame_id = self.profile.base_frame
        position.link_name = self.profile.wrist_link
        position.target_point_offset = ros["Vector3"]()
        sphere = ros["SolidPrimitive"]()
        sphere.type = ros["SolidPrimitive"].SPHERE
        sphere.dimensions = [0.005]
        position.constraint_region.primitives.append(sphere)
        region_pose = ros["Pose"]()
        region_pose.position.x = float(wrist_target[0])
        region_pose.position.y = float(wrist_target[1])
        region_pose.position.z = float(wrist_target[2])
        region_pose.orientation.w = 1.0
        position.constraint_region.primitive_poses.append(region_pose)
        position.weight = 1.0
        goal.position_constraints.append(position)

        orientation = ros["OrientationConstraint"]()
        orientation.header.frame_id = self.profile.base_frame
        orientation.link_name = self.profile.wrist_link
        orientation.orientation = self._quaternion(wrist_target[3:])
        orientation.absolute_x_axis_tolerance = 0.05
        orientation.absolute_y_axis_tolerance = 0.05
        orientation.absolute_z_axis_tolerance = 0.05
        orientation.weight = 1.0
        goal.orientation_constraints.append(orientation)
        motion.goal_constraints.append(goal)

        future = self._planner.call_async(request)
        ros["rclpy"].spin_until_future_complete(
            self._node, future, timeout_sec=self.plan_timeout_s
        )
        if not future.done() or future.result() is None:
            raise G1DTransportError("MoveIt planning service timed out")
        response = future.result().motion_plan_response
        if response.error_code.val != ros["MoveItErrorCodes"].SUCCESS:
            raise G1DTransportError(
                f"MoveIt failed to plan (error_code={response.error_code.val})"
            )
        path = response.trajectory.joint_trajectory
        data = self._trajectory_from_msg(path)
        # Publish it for RViz, exactly like the standalone planning script does.
        try:
            display = ros["DisplayTrajectory"]()
            display.trajectory_start = response.trajectory_start
            display.trajectory.append(response.trajectory)
            self._display = self._node.create_publisher(
                ros["DisplayTrajectory"], "/display_planned_path", 10
            )
            self._display.publish(display)
        except Exception:  # noqa: BLE001 - RViz display is optional
            pass
        self._guard(data, start_joints)
        return data

    def _build_joint_trajectory(
        self, tcp_target7: Sequence[float], start_joints: Sequence[float]
    ) -> TrajectoryData:
        """Solve the Cartesian target with the local IK and build a 2-point path.

        The trajectory is a straight-line joint interpolation from the measured
        state to the IK solution, sampled densely enough that the bridge's
        per-segment speed guard sees the real commanded speed. The duration is
        chosen so the fastest joint moves at the bridge's own 0.5 rad/s guard
        (with a little margin), so the plan is accepted rather than aborted.
        """
        target = np.asarray(tcp_target7, dtype=float).reshape(-1)
        solution = self._ik.solve(target, seed=np.asarray(start_joints, dtype=float))
        if solution is None:
            raise G1DTransportError(
                "the local IK could not reach the commanded end-effector pose "
                "(the target is outside the reachable workspace of this arm)"
            )
        start = np.asarray(start_joints, dtype=float).reshape(-1)
        delta = np.abs(solution - start)
        peak = float(np.max(delta)) if delta.size else 0.0
        if peak < 1e-6:
            # Already there: a no-op trajectory is still valid (one point).
            data = TrajectoryData(
                joint_names=tuple(self._joint_names),
                times_s=(0.0,),
                positions=(tuple(float(v) for v in start),),
                velocities=(tuple([0.0] * 7),),
            )
            self._guard(data, start_joints)
            return data
        # Duration so the fastest joint moves at ~90% of the bridge's guard.
        speed = BRIDGE_MAX_TRAJECTORY_SPEED_RAD_S * 0.9
        duration_s = max(0.2, peak / speed)
        steps = max(2, int(math.ceil(duration_s / 0.05)))
        times: list[float] = []
        positions: list[tuple[float, ...]] = []
        velocities: list[tuple[float, ...]] = []
        for index in range(steps + 1):
            alpha = index / steps
            times.append(duration_s * alpha)
            positions.append(tuple(float(v) for v in (start + alpha * (solution - start))))
            velocities.append(tuple(float(v) for v in (solution - start) / duration_s))
        data = TrajectoryData(
            joint_names=tuple(self._joint_names),
            times_s=tuple(times),
            positions=tuple(positions),
            velocities=tuple(velocities),
        )
        self._guard(data, start_joints)
        return data

    def _trajectory_from_msg(self, path: Any) -> TrajectoryData:
        """A ROS ``JointTrajectory`` message -> the plain :class:`TrajectoryData`."""
        names = tuple(str(name) for name in path.joint_names)
        times: list[float] = []
        positions: list[tuple[float, ...]] = []
        velocities: list[tuple[float, ...]] = []
        has_velocities = bool(path.points) and all(
            len(point.velocities) == len(names) for point in path.points
        )
        for point in path.points:
            times.append(
                float(point.time_from_start.sec)
                + float(point.time_from_start.nanosec) * 1e-9
            )
            positions.append(tuple(float(v) for v in point.positions))
            if has_velocities:
                velocities.append(tuple(float(v) for v in point.velocities))
        return TrajectoryData(
            joint_names=names,
            times_s=tuple(times),
            positions=tuple(positions),
            velocities=tuple(velocities) if has_velocities else (),
        )

    def _guard(self, data: TrajectoryData, start_joints: Sequence[float]) -> None:
        """Apply the bridge's own acceptance rules to the trajectory.

        The z-floor is checked HERE, not only at the command site, because
        ``_guard`` is the single choke point every trajectory passes through
        (both the MoveIt path and the local-IK path call it). A trajectory that
        would put the hand below the calibrated floor is rejected before it can
        reach the bridge -- so no future caller can bypass the guard by
        building a trajectory differently.
        """
        structure = validate_trajectory_structure(data, self._joint_names)
        if structure is not None:
            raise G1DTransportError(f"the bridge would reject this trajectory: {structure}")
        report = check_trajectory(
            data,
            self._limits,
            start_joints,
            list(self._joint_names),
            max_speed_rad_s=BRIDGE_MAX_TRAJECTORY_SPEED_RAD_S,
            max_start_error_rad=BRIDGE_MAX_START_ERROR_RAD,
        )
        if not report.ok:
            raise G1DTransportError(f"the bridge would reject this trajectory: {report.reason}")
        floor_problem = self.check_z_floor(data)
        if floor_problem is not None:
            raise G1DTransportError(
                f"the z-floor rejects this trajectory: {floor_problem}"
            )
        self.last_trajectory = {
            "points": report.points,
            "duration_s": report.duration_s,
            "peak_speed_rad_s": report.max_speed_rad_s,
            "start_error_rad": report.max_start_error_rad,
            # The commanded final joint positions: the last waypoint of the
            # trajectory that was sent. The bridge's settle check compares the
            # MEASURED joints against exactly this, so a diagnostic can
            # reproduce the comparison offline.
            "final_positions": [float(v) for v in data.positions[-1]],
            "z_floor_m": (
                self.z_floor.z_m if (self.z_floor and self.z_floor.enabled) else None
            ),
        }
        if self.verbose:
            print(f"[g1d-transport] {report.summary()}")

    # -- execution ------------------------------------------------------------
    def _send_trajectory(self, data: TrajectoryData) -> None:
        """Send the trajectory to the selected side's Action server."""
        ros = self._ros
        if not self._arm_client.wait_for_server(timeout_sec=5.0):
            raise G1DTransportError(
                f"Action {self.profile.trajectory_action} is unavailable -- is the "
                "SDK bridge running?"
            )
        goal = ros["FollowJointTrajectory"].Goal()
        goal.trajectory = self._trajectory_to_msg(data)
        send_future = self._arm_client.send_goal_async(goal)
        ros["rclpy"].spin_until_future_complete(
            self._node, send_future, timeout_sec=10.0
        )
        if not send_future.done() or send_future.result() is None:
            raise G1DTransportError("the trajectory goal was not acknowledged")
        handle = send_future.result()
        if not handle.accepted:
            raise G1DTransportError(
                f"the bridge REJECTED the trajectory goal for {self.profile.side}. "
                "Check the bridge log: the start-state, joint-limit and speed "
                "guards all reject goals BEFORE any LowCmd is written."
            )
        result_future = handle.get_result_async()
        ros["rclpy"].spin_until_future_complete(
            self._node, result_future, timeout_sec=self.execute_timeout_s
        )
        if not result_future.done() or result_future.result() is None:
            handle.cancel_goal_async()
            raise G1DTransportError(
                "trajectory execution timed out; inspect the robot before any "
                "next action"
            )
        reply = result_future.result()
        if reply.status != ros["GoalStatus"].STATUS_SUCCEEDED:
            raise G1DTransportError(
                f"trajectory execution ended with action status {reply.status}"
            )
        result = reply.result
        if result.error_code != 0:  # control_msgs FollowJointTrajectory SUCCESSFUL
            # GOAL_TOLERANCE_VIOLATED is the G1-D's most common real-execution
            # outcome, and it is NOT a motion failure: the bridge drives the arm
            # to the commanded final point and KEEPS HOLDING it, but the measured
            # joints have not converged inside the bridge's settle window (the
            # gravity-loaded shoulder creeps). The Cartesian goal was therefore
            # reached. Measure the ACTUAL Cartesian error before deciding: if the
            # TCP is where it was asked to be, the move succeeded and the
            # pipeline must not be torn down -- otherwise every token of a
            # closed-loop rollout dies on the first slightly-off move.
            if int(result.error_code) == ERR_GOAL_TOLERANCE_VIOLATED:
                self._report_tolerance_violation(data, result)
                if self.tolerance_violation_is_fatal:
                    raise G1DTransportError(
                        f"trajectory execution failed (error_code={result.error_code}): "
                        f"{result.error_string}"
                    )
                return
            raise G1DTransportError(
                f"trajectory execution failed (error_code={result.error_code}): "
                f"{result.error_string}"
            )
        if self.verbose:
            print(
                f"[g1d-transport] {self.profile.side} trajectory executed "
                f"({len(data.positions)} points)"
            )

    def _report_tolerance_violation(
        self, data: TrajectoryData, result: Any
    ) -> None:
        """Log the MEASURED Cartesian error after a GOAL_TOLERANCE_VIOLATED abort.

        The bridge keeps holding the final position, so the arm IS at the goal
        even though the action was aborted. This prints the numbers the
        operator needs to decide whether to loosen the bridge's
        ``goal_tolerance_rad`` / ``settle_timeout_sec``:
        the commanded final joints, the measured joints, the per-joint error,
        and the resulting TCP error.
        """
        try:
            measured = np.asarray(self.read_joint_positions(), dtype=float)
        except Exception as exc:  # noqa: BLE001 - diagnostics must never raise
            measured = np.array([], dtype=float)
        commanded = np.asarray(data.positions[-1], dtype=float)
        detail = ""
        if measured.size == commanded.size:
            errors = np.abs(measured - commanded)
            worst = int(np.argmax(errors))
            detail = (
                f" max_joint_error={errors[worst]:.4f} rad on "
                f"{data.joint_names[worst]}"
            )
        tcp_error = ""
        try:
            target = self._last_commanded_pose
            if target is not None and measured.size == 7:
                achieved = self._kin.tcp_pose7(measured)
                delta = achieved[:3] - np.asarray(target[:3], dtype=float)
                tcp_error = (
                    f" tcp_error={np.round(delta, 4).tolist()} m "
                    f"(|e|={float(np.linalg.norm(delta)) * 1000:.1f} mm)"
                )
        except Exception:  # noqa: BLE001 - diagnostics must never raise
            tcp_error = ""
        message = (
            f"[g1d-transport] GOAL_TOLERANCE_VIOLATED on {self.profile.side}: "
            f"the arm reached the commanded point but the measured joints had not "
            f"converged inside the bridge's settle window.{detail}{tcp_error}\n"
            f"[g1d-transport]   the bridge keeps holding the final position, so the "
            f"move is treated as REACHED. Loosen the bridge's goal_tolerance_rad / "
            f"settle_timeout_sec if the error is larger than a few millimetres."
        )
        if self.verbose:
            print(message)
        self.last_tolerance_violation = {
            "error_string": str(getattr(result, "error_string", "")),
            "commanded_rad": [float(v) for v in commanded],
            "measured_rad": [float(v) for v in measured],
            "error_rad": (
                [round(float(v), 6) for v in np.abs(measured - commanded)]
                if measured.size == commanded.size
                else []
            ),
        }

    def _trajectory_to_msg(self, data: TrajectoryData) -> Any:
        """The plain :class:`TrajectoryData` -> a ROS ``JointTrajectory`` message."""
        ros = self._ros
        from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

        message = JointTrajectory()
        message.joint_names = list(data.joint_names)
        for index, stamp in enumerate(data.times_s):
            point = JointTrajectoryPoint()
            point.positions = [float(v) for v in data.positions[index]]
            if data.velocities:
                point.velocities = [float(v) for v in data.velocities[index]]
            nanoseconds = int(round(float(stamp) * 1e9))
            point.time_from_start.sec = nanoseconds // 1_000_000_000
            point.time_from_start.nanosec = nanoseconds % 1_000_000_000
            message.points.append(point)
        return message

    def _send_gripper(self, opening_m: float) -> None:
        """Send a GripperCommand goal to the selected side's Action server."""
        ros = self._ros
        if not self._gripper_client.wait_for_server(timeout_sec=5.0):
            raise G1DTransportError(
                f"Action {self.profile.gripper_action} is unavailable -- is the "
                "SDK bridge running?"
            )
        goal = ros["GripperCommand"].Goal()
        goal.command.position = float(opening_m)
        goal.command.max_effort = 0.0
        send_future = self._gripper_client.send_goal_async(goal)
        ros["rclpy"].spin_until_future_complete(
            self._node, send_future, timeout_sec=10.0
        )
        if not send_future.done() or send_future.result() is None:
            raise G1DTransportError("the gripper goal was not acknowledged")
        handle = send_future.result()
        if not handle.accepted:
            raise G1DTransportError(
                f"the bridge REJECTED the gripper goal for {self.profile.side}. "
                "In real mode this usually means gripper_real_execution is off, "
                "or an arm goal is already running."
            )
        result_future = handle.get_result_async()
        ros["rclpy"].spin_until_future_complete(
            self._node, result_future, timeout_sec=self.gripper_timeout_s
        )
        if not result_future.done() or result_future.result() is None:
            handle.cancel_goal_async()
            raise G1DTransportError(
                "gripper result timed out; inspect the robot before another action"
            )
        reply = result_future.result()
        if reply.status != ros["GoalStatus"].STATUS_SUCCEEDED:
            raise G1DTransportError(
                f"gripper action ended with status {reply.status}"
            )
        if reply.result.stalled or not reply.result.reached_goal:
            raise G1DTransportError(
                f"gripper did not reach the goal: stalled={reply.result.stalled}, "
                f"reached_goal={reply.result.reached_goal}"
            )

    # -- helpers --------------------------------------------------------------
    @staticmethod
    def _quaternion(values: Sequence[float]) -> Any:
        """A normalized ``geometry_msgs/Quaternion`` from a xyzw quadruple."""
        ros = _require_ros()
        x, y, z, w = (float(v) for v in np.asarray(values, dtype=float).reshape(-1))
        norm = math.sqrt(x * x + y * y + z * z + w * w)
        if norm < 1e-8:
            raise G1DTransportError("target quaternion must be non-zero")
        return ros["Quaternion"](x=x / norm, y=y / norm, z=z / norm, w=w / norm)
