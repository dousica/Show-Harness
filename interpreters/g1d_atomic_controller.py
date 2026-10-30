"""Unitree G1-D atomic-action controller.

The token -> Cartesian-setpoint machinery lives in
:class:`interpreters.real_atomic_controller.RealAtomicController`; this subclass
adds only what is specific to the G1-D arm driven through the SDK bridge
(:class:`core.g1d.g1d_interface.G1DRobotInterface`):

* **Cartesian motion in the ``torso_link`` frame.** MV_* are straight base-frame
  translations and ROTATE_* a base-Z gripper yaw, exactly like the Piper. The
  G1-D's base frame is the SRDF chain base of ``left_arm``/``right_arm``, so the
  primitives yaml is expressed in that frame.

* **A trajectory-per-command motion model.** The G1-D has NO Cartesian
  impedance controller: the bridge accepts whole joint trajectories. So one
  atomic token becomes ONE planned trajectory (MoveIt when it is up, the local
  IK otherwise), which is sent as a single FollowJointTrajectory goal. There is
  no "stream a setpoint and let the controller track it" mode, so the smooth
  (min-jerk) plugin's ramp is not used here: the whole displacement is planned
  and executed in one go, and the z-floor guard in the transport checks the
  resulting trajectory.

* **A z-floor that is enforced in TWO places.** The controller clamps the
  commanded setpoint (so a downward token stops at the floor) and the
  transport verifies the whole trajectory (so a path that dips through the
  table is rejected). The controller's own floor value is the SAME object the
  transport holds -- see :mod:`core.g1d.safety`.

* **Gripper sizing for the Dex1.** The nominal stroke is 0.09 m, so the
  "closed" threshold and the empty-grasp width are Dex1 numbers. They are
  PLACEHOLDERS until the jaw width is calibrated on the real unit.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from interpreters.real_atomic_controller import (
    AtomicStepResult,
    RealAtomicController,
)

# --- Dex1 gripper placeholders (CALIBRATE on the real unit) -----------------
# The nominal stroke is 0.09 m (core.g1d.geometry.DEX1_NOMINAL_OPENING_M).
# Measure the empty-closed width and a typical held-object width, then set
# them in configs/robot_g1d.yaml.
G1D_GRIPPER_CLOSE_THRESHOLD_M = 0.07   # width below this counts as "closed"
G1D_GRASP_OPEN_WIDTH_M = 0.06         # width at/above this is still-open band
G1D_EMPTY_GRASP_WIDTH_M = 0.005       # a close settling at/below this caught nothing


class G1DAtomicController(RealAtomicController):
    """Maps atomic VLM tokens to motion on the G1-D arm."""

    LOG_TAG = "g1d-atomic"

    def __init__(
        self,
        robot: Any,
        atomic_primitives: dict[str, Any],
        step_m: float,
        yaw_step_rad: float,
        gripper_close_threshold_m: float = G1D_GRIPPER_CLOSE_THRESHOLD_M,
        grasp_open_width_m: float = G1D_GRASP_OPEN_WIDTH_M,
        grasp_min_width_m: Optional[float] = G1D_EMPTY_GRASP_WIDTH_M,
        gripper_settle_s: float = 3.0,
        **kwargs: Any,
    ) -> None:
        """Args (G1-D-specific; the rest match ``RealAtomicController``):
            gripper_close_threshold_m: Width below which the Dex1 counts as closed.
            grasp_open_width_m: Upper bound of the "holding" width band.
            grasp_min_width_m: A close settling at/below this width caught nothing
                (the gripper is reopened and the step is flagged ``grasp_empty``).
            gripper_settle_s: Max time to wait for the fingers to stop moving. The
                bridge's Dex1 stroke is slow (a full 5.4 rad at its 0.5 rad/s guard
                takes ~11 s), so this is generous.
        """
        super().__init__(
            robot,
            atomic_primitives,
            step_m,
            yaw_step_rad,
            gripper_close_threshold_m=gripper_close_threshold_m,
            grasp_open_width_m=grasp_open_width_m,
            grasp_min_width_m=grasp_min_width_m,
            gripper_settle_s=gripper_settle_s,
            **kwargs,
        )

    # -- motion -------------------------------------------------------------
    def _drive_to_target(
        self,
        start_pos: np.ndarray,
        start_euler: np.ndarray,
        continuous: bool = False,
    ) -> None:
        """Execute the whole displacement as ONE planned trajectory.

        The G1-D has no streaming setpoint controller: the SDK bridge accepts a
        joint trajectory, so a token's move is planned end-to-end (MoveIt or the
        local IK) and sent as one goal. That means the smooth plugin's per-
        waypoint ramp does not apply here -- there is nothing to ramp, the
        planner produces the path.

        The z-floor is enforced by the transport (it owns the trajectory and
        checks every sampled configuration), so this method only has to make
        sure the SETPOINT it asks for is above the floor.
        """
        dpos = self._target_pos - start_pos
        deuler = self._target_euler - start_euler
        moved = (
            float(np.linalg.norm(dpos)) > 1e-5
            or float(np.linalg.norm(deuler)) > 1e-5
        )
        if not moved:
            # Nothing to move: re-command the current setpoint so the bridge
            # sees a (no-op) goal and the step is recorded honestly.
            self._command_setpoint()
            return
        self._command_setpoint()

    def _on_gripper_settled(
        self,
        close: bool,
        pre_width: float,
        settled_width: float,
    ) -> None:
        """Warn when a gripper command produced no width change.

        The bridge reports a stalled goal, but a DROPPED gripper command (wrong
        namespace, the action server not up) looks the same as a successful
        close from the harness's side. A real close always changes the width
        well beyond the motion epsilon, so no change means the command did not
        take -- and a recorded grasp may be phantom.
        """
        if settled_width is None:
            return
        if abs(settled_width - pre_width) <= self.GRIPPER_MOTION_EPS_M:
            print(
                f"[{self.LOG_TAG}] WARNING: gripper {'CLOSE' if close else 'OPEN'} "
                f"produced no width change ({pre_width:.4f}->{settled_width:.4f} m). The "
                "command was likely DROPPED (action server not running, or the wrong "
                "bridge namespace); a recorded grasp may be phantom."
            )

    def _apply_motion(
        self,
        delta_pos: np.ndarray,
        yaw: float,
        kind: str,
        pre_pose: np.ndarray,
        token: str,
        continuous: bool = False,
    ) -> AtomicStepResult:
        result = super()._apply_motion(
            delta_pos,
            yaw,
            kind=kind,
            pre_pose=pre_pose,
            token=token,
            continuous=continuous,
        )
        # The setpoint may have been clamped by the controller's own z-floor;
        # make sure the record shows where the arm was actually sent.
        result.target_pose = self.target_pose
        return result
