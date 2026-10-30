#!/usr/bin/env python3
"""G1-D left-arm TCP goal planner, ROS2 Humble. No SDK/LowCmd publisher.

Default: inspect only. --dz is relative upward displacement; --target-z is
absolute height in torso_link. X/Y may be supplied to route toward a work area.
Goal orientation stays near the current orientation; the PATH is NOT Cartesian
or orientation constrained. Table geometry MUST be in MoveIt's planning scene.
This standalone commissioning helper does NOT enforce Harness's z-floor policy.
"""
import argparse
from copy import deepcopy
import math
import select
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from moveit_msgs.msg import (Constraints, PositionConstraint, OrientationConstraint,
                             DisplayTrajectory, MoveItErrorCodes)
from moveit_msgs.srv import GetMotionPlan
from moveit_msgs.action import ExecuteTrajectory
from tf2_ros import Buffer, TransformListener, TransformException

JOINTS = tuple('left_' + n + '_joint' for n in (
    'shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow',
    'wrist_roll', 'wrist_pitch', 'wrist_yaw'))
FRAME = 'torso_link'
TCP = 'left_dex1_tcp_link'


def seconds(d):
    return d.sec + d.nanosec * 1e-9


class LiftPlanner(Node):
    def __init__(self):
        super().__init__('g1d_left_work_pose')
        self.state = None
        self.received = 0.0
        self.tf = Buffer()
        self.listener = TransformListener(self.tf, self)
        self.sub = self.create_subscription(JointState, '/joint_states',
                                             self.on_state, qos_profile_sensor_data)
        self.planner = self.create_client(GetMotionPlan, '/plan_kinematic_path')
        self.executor = ActionClient(self, ExecuteTrajectory, '/execute_trajectory')
        self.publishers = [self.create_publisher(DisplayTrajectory, topic, 10)
                           for topic in ('/display_planned_path',
                                         '/move_group/display_planned_path')]
        self.handle = None

    def on_state(self, msg):
        if len(msg.name) != len(msg.position):
            return
        values = dict(zip(msg.name, msg.position))
        if all(j in values and math.isfinite(values[j]) for j in JOINTS):
            self.state = deepcopy(msg)
            self.received = time.monotonic()

    def wait(self, future, timeout):
        end = time.monotonic() + timeout
        while rclpy.ok() and not future.done() and time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.05)
        if not future.done():
            raise RuntimeError('ROS request timed out; execution state may be unknown')
        value = future.result()
        if value is None:
            raise RuntimeError('ROS request returned no result')
        return value

    def fresh_state(self):
        previous = self.received
        end = time.monotonic() + 5
        while rclpy.ok() and time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.received > previous and time.monotonic() - self.received < 0.5:
                return deepcopy(self.state)
        raise RuntimeError('No fresh left-arm /joint_states')

    def pose(self):
        end = time.monotonic() + 5
        while rclpy.ok() and time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.05)
            try:
                tf = self.tf.lookup_transform(FRAME, TCP, Time())
                age = (self.get_clock().now().nanoseconds -
                       Time.from_msg(tf.header.stamp).nanoseconds) * 1e-9
                if not -0.1 <= age <= 0.5:
                    continue
                p = Pose()
                t = tf.transform.translation
                p.position.x, p.position.y, p.position.z = t.x, t.y, t.z
                p.orientation = tf.transform.rotation
                return p
            except TransformException:
                pass
        raise RuntimeError('No fresh torso_link -> left_dex1_tcp_link TF')

    def plan(self, start, goal, args):
        if not self.planner.wait_for_service(timeout_sec=5):
            raise RuntimeError('No /plan_kinematic_path service')
        req = GetMotionPlan.Request()
        m = req.motion_plan_request
        m.group_name = 'left_arm'
        m.start_state.joint_state = start
        m.start_state.is_diff = True
        m.num_planning_attempts = 5
        m.allowed_planning_time = 15.0
        m.max_velocity_scaling_factor = args.velocity
        m.max_acceleration_scaling_factor = args.acceleration
        pc = PositionConstraint()
        pc.header.frame_id, pc.link_name = FRAME, TCP
        shape = SolidPrimitive()
        shape.type, shape.dimensions = SolidPrimitive.SPHERE, [0.003]
        pc.constraint_region.primitives = [shape]
        region = Pose()
        region.position = goal.position
        region.orientation.w = 1.0
        pc.constraint_region.primitive_poses = [region]
        pc.weight = 1.0
        oc = OrientationConstraint()
        oc.header.frame_id, oc.link_name = FRAME, TCP
        oc.orientation = goal.orientation
        oc.absolute_x_axis_tolerance = 0.18
        oc.absolute_y_axis_tolerance = 0.18
        oc.absolute_z_axis_tolerance = 0.18
        oc.weight = 1.0
        m.goal_constraints = [Constraints(position_constraints=[pc],
                                          orientation_constraints=[oc])]
        response = self.wait(self.planner.call_async(req), 35).motion_plan_response
        if response.error_code.val != MoveItErrorCodes.SUCCESS:
            raise RuntimeError(f'MoveIt planning failed: {response.error_code.val}')
        return response

    def confirm(self):
        print('Review RViz AND table geometry. Type YES to execute, anything else cancels: ',
              end='', flush=True)
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)
            if select.select([sys.stdin], [], [], 0)[0]:
                return sys.stdin.readline().strip() == 'YES'
        return False

    def execute(self, trajectory, duration):
        if not self.executor.wait_for_server(timeout_sec=5):
            raise RuntimeError('No /execute_trajectory server')
        self.handle = self.wait(self.executor.send_goal_async(
            ExecuteTrajectory.Goal(trajectory=trajectory)), 8)
        if not self.handle.accepted:
            self.handle = None
            raise RuntimeError('Execution goal rejected')
        result = self.wait(self.handle.get_result_async(), max(60, 2 * duration + 30))
        self.handle = None
        if (result.status != GoalStatus.STATUS_SUCCEEDED or
                result.result.error_code.val != MoveItErrorCodes.SUCCESS):
            raise RuntimeError(f'Execution failed: status={result.status}, '
                               f'MoveIt={result.result.error_code.val}')


def validate(trajectory, start):
    jt = trajectory.joint_trajectory
    if len(jt.joint_names) != 7 or set(jt.joint_names) != set(JOINTS) or len(jt.points) < 2:
        raise RuntimeError('Expected nonempty, seven-joint left-arm trajectory')
    if trajectory.multi_dof_joint_trajectory.points:
        raise RuntimeError('Unexpected multi-DOF trajectory')
    values = dict(zip(start.name, start.position))
    previous = None
    peak = 0.0
    for p in jt.points:
        if len(p.positions) != 7 or not all(math.isfinite(v) for v in p.positions):
            raise RuntimeError('Invalid trajectory positions')
        t = seconds(p.time_from_start)
        if not math.isfinite(t) or t < 0:
            raise RuntimeError('Invalid time')
        for field in (p.velocities, p.accelerations):
            if field and (len(field) != 7 or not all(math.isfinite(v) for v in field)):
                raise RuntimeError('Invalid trajectory derivative')
        if p.velocities:
            peak = max(peak, max(abs(v) for v in p.velocities))
        if previous is None:
            if max(abs(p.positions[k] - values[j])
                   for k, j in enumerate(jt.joint_names)) > 0.025:
                raise RuntimeError('Trajectory start differs from measured state')
        else:
            dt = t - seconds(previous.time_from_start)
            if dt <= 0:
                raise RuntimeError('Non-increasing trajectory time')
            peak = max(peak, max(abs(a-b)/dt for a,b in zip(p.positions, previous.positions)))
        previous = p
    # Intentionally stricter client guard; bridge still checks interpolation.
    if peak > 0.30:
        raise RuntimeError(f'Waypoint/segment speed {peak:.4f} rad/s > 0.30; reduce scaling')
    print(f'Plan: {len(jt.points)} points, {t:.2f} s, sampled speed max {peak:.4f} rad/s')
    return t


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    height = ap.add_mutually_exclusive_group()
    height.add_argument('--dz', type=float, help='Relative upward TCP displacement (m)')
    height.add_argument('--target-z', type=float, help='Absolute torso_link Z (m)')
    ap.add_argument('--x', type=float, help='Absolute torso_link X (m)')
    ap.add_argument('--y', type=float, help='Absolute torso_link Y (m)')
    ap.add_argument('--velocity', type=float, default=0.05)
    ap.add_argument('--acceleration', type=float, default=0.05)
    ap.add_argument('--execute', action='store_true')
    args = ap.parse_args()
    for v in (args.dz, args.target_z, args.x, args.y):
        if v is not None and not math.isfinite(v):
            ap.error('Targets must be finite')
    if not (0 < args.velocity <= 0.1 and 0 < args.acceleration <= 0.1):
        ap.error('Scaling must be within (0, 0.1]')
    if args.execute and args.dz is None and args.target_z is None:
        ap.error('Execution needs --dz or --target-z')
    rclpy.init(args=[])
    node = LiftPlanner()
    try:
        start = node.fresh_state()
        current = node.pose()
        p = current.position
        print(f'Current TCP in {FRAME}: [{p.x:+.5f}, {p.y:+.5f}, {p.z:+.5f}] m')
        if args.dz is None and args.target_z is None:
            print('INSPECT ONLY. Supply --dz or --target-z to plan. No motion sent.')
            return 0
        goal = deepcopy(current)
        goal.position.z = p.z + args.dz if args.dz is not None else args.target_z
        if args.x is not None:
            goal.position.x = args.x
        if args.y is not None:
            goal.position.y = args.y
        delta = [getattr(goal.position, k)-getattr(p, k) for k in ('x','y','z')]
        if not 0 < delta[2] <= 0.30 or math.sqrt(sum(v*v for v in delta)) > 0.40:
            raise RuntimeError('Requires upward displacement (0, 0.30] m and total <= 0.40 m')
        print(f'Target [{goal.position.x:+.5f}, {goal.position.y:+.5f}, '
              f'{goal.position.z:+.5f}] m; displacement {delta}')
        print('This helper does NOT apply Harness z-floor. Collision scene must model the table.')
        response = node.plan(start, goal, args)
        trajectory = response.trajectory
        duration = validate(trajectory, start)
        display = DisplayTrajectory(trajectory_start=response.trajectory_start,
                                    trajectory=[trajectory])
        for _ in range(10):
            for pub in node.publishers:
                pub.publish(display)
            rclpy.spin_once(node, timeout_sec=0.1)
        if not args.execute:
            print('PLAN ONLY. No execution goal sent. Re-running creates a NEW plan.')
            return 0
        if not node.confirm():
            print('Canceled; no execution goal sent.')
            return 0
        fresh = node.fresh_state()
        actual = dict(zip(fresh.name, fresh.position))
        initial = dict(zip(start.name, start.position))
        monitored = [j for j in initial if j in actual]
        if max(abs(actual[j]-initial[j]) for j in monitored) > 0.025:
            raise RuntimeError('Robot state changed during review; re-plan')
        node.execute(trajectory, duration)
        measured = node.fresh_state()
        actual = dict(zip(measured.name, measured.position))
        jt = trajectory.joint_trajectory
        error = max(abs(actual[j]-q) for j,q in zip(jt.joint_names, jt.points[-1].positions))
        final = node.pose().position
        distance = math.sqrt(sum((getattr(final,k)-getattr(goal.position,k))**2 for k in ('x','y','z')))
        print(f'Final TCP [{final.x:+.5f}, {final.y:+.5f}, {final.z:+.5f}] m; '
              f'joint error {error:.5f} rad; TCP position error {distance*1000:.1f} mm')
        if error > 0.03 or distance > 0.01:
            raise RuntimeError('Execution returned success but measured endpoint needs inspection')
        print('Reached this target; no further motion queued.')
        return 0
    except (RuntimeError, KeyboardInterrupt, EOFError) as exc:
        print(f'STOP: {exc}', file=sys.stderr)
        return 1
    finally:
        if node.handle is not None and node.handle.accepted:
            try:
                node.wait(node.handle.cancel_goal_async(), 3)
                print('Cancellation requested; verify physical robot has stopped.', file=sys.stderr)
            except Exception:
                print('Could not confirm cancellation; check robot immediately.', file=sys.stderr)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    sys.exit(main())
