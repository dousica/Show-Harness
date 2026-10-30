"""Numerical inverse kinematics for the G1-D arm.

Why this exists
---------------
MoveIt is the planner of record for the G1-D, but it is NOT always running: the
lab's live rig (the ``lzh_sdk_bridge`` on ``ROS_DOMAIN_ID=42``) currently exposes
only the bridge's FollowJointTrajectory Action, with no ``move_group`` and no
``/plan_kinematic_path`` service. Without an IK, the harness could not command a
Cartesian pose at all on that rig.

So this module provides a self-contained damped-least-squares (DLS) IK on the
harness's own forward-kinematics model (:mod:`core.g1d.kinematics`), which is
transcribed from the official G1-D URDF and cross-checked against it in
``tests/test_g1d_bridge_guard.py``.

What it is NOT
--------------
* It is not a collision-aware planner. It does NOT know about the robot's
  collision geometry, the table, or the objects on it. Every solution it returns
  MUST still be checked by MoveIt (when MoveIt is running) before it is executed.
* It does not find "the" solution. A 7-DoF arm has a redundancy, so many joint
  configurations reach the same wrist pose; this returns the one closest to the
  seed (the measured joints), which is the natural choice for a small Cartesian
  step: the arm moves as little as it can.
* Its convergence is verified, not assumed: :meth:`ArmInverseKinematics.solve`
  returns ``None`` when it cannot reach the target within its tolerance, so the
  caller never gets a silently-wrong pose.

Where it fits in the safety chain
---------------------------------
Even a successful IK solution is only a *candidate*. The transport still
  1. clamps the solution to the official G1-D joint limits,
  2. checks the resulting trajectory against the bridge's own guards
     (:mod:`core.g1d.bridge_guard`), and
  3. hands it to the bridge, which re-checks everything before writing a LowCmd.

So a solution from here can be REJECTED by the bridge, but it can never bypass
the bridge's limits.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from core.g1d.geometry import arm_limits_rad
from core.g1d.kinematics import ArmKinematics, tcp_to_wrist_pose7

#: Default convergence tolerances.
POSITION_TOLERANCE_M = 1e-4
ORIENTATION_TOLERANCE_RAD = 3e-3
#: The DLS damping factor (a small Tikhonov term that keeps the solve stable
#: near a singular Jacobian, e.g. at a wrist singularity). It must stay SMALL:
#: a large value regularises the step away from the true least-squares
#: solution and leaves a residual of a few tenths of a millimetre.
DEFAULT_DAMPING = 1e-4
#: How much tighter than the reported tolerance the solver aims. Without this
#: the solver stopped as soon as the residual was under the tolerance, so it
#: returned a solution that merely squeaked past it instead of the
#: least-squares one.
CONVERGENCE_MARGIN = 0.1
#: The maximum joint step per iteration (rad). A large step would let the solver
#: jump past a limit and into a different basin.
MAX_JOINT_STEP_RAD = 0.15
#: The maximum null-space step per iteration (rad). It is kept small because the
#: null-space term only steers the redundant DOF; it must not fight the task.
MAX_NULL_SPACE_STEP_RAD = 0.02
#: The maximum number of DLS iterations.
MAX_ITERATIONS = 120
#: The Jacobian's finite-difference step (rad).
JACOBIAN_EPS_RAD = 1e-6
#: How strongly the null-space term pulls the solution back toward the seed.
#: The G1-D arm has a 1-DoF redundancy, and without this the solver would drift
#: into a different branch of the solution space -- on the live robot that drove
#: a joint into its limit while chasing a 1 cm translation.
NULL_SPACE_GAIN = 0.15


def _rotation_vector(rotation: np.ndarray) -> np.ndarray:
    """A rotation matrix -> its rotation vector (axis * angle), numerically stable."""
    matrix = np.asarray(rotation, dtype=float)
    cosine = float(np.clip((np.trace(matrix) - 1.0) / 2.0, -1.0, 1.0))
    angle = math.acos(cosine)
    if angle < 1e-9:
        return np.zeros(3)
    if angle > math.pi - 1e-6:
        # Near pi the sin() form loses precision: use the symmetric part.
        symmetric = (matrix + np.eye(3)) / 2.0
        axis = np.sqrt(np.clip(np.diag(symmetric), 0.0, None))
        # Fix the signs from the off-diagonal terms.
        signs = np.array(
            [
                np.sign(matrix[2, 1] - matrix[1, 2]) or 1.0,
                np.sign(matrix[0, 2] - matrix[2, 0]) or 1.0,
                np.sign(matrix[1, 0] - matrix[0, 1]) or 1.0,
            ]
        )
        axis = axis * signs
        norm = float(np.linalg.norm(axis))
        if norm < 1e-9:
            return np.zeros(3)
        return axis / norm * angle
    axis = np.array(
        [
            matrix[2, 1] - matrix[1, 2],
            matrix[0, 2] - matrix[2, 0],
            matrix[1, 0] - matrix[0, 1],
        ]
    ) / (2.0 * math.sin(angle))
    return axis * angle


@dataclass(frozen=True)
class IKSolution:
    """One inverse-kinematics solution and how well it actually converged."""

    joints: np.ndarray
    position_error_m: float
    orientation_error_rad: float

    @property
    def ok(self) -> bool:
        return (
            self.position_error_m <= POSITION_TOLERANCE_M
            and self.orientation_error_rad <= ORIENTATION_TOLERANCE_RAD
        )


class ArmInverseKinematics:
    """Damped-least-squares IK for ONE G1-D arm.

    The target is the **TCP** pose (the fingertip midpoint), because that is what
    the harness reports and commands. It is converted to the wrist frame
    internally, because the wrist link is what the kinematics model resolves.
    """

    def __init__(
        self,
        side: str,
        *,
        damping: float = DEFAULT_DAMPING,
        max_iterations: int = MAX_ITERATIONS,
        position_tolerance_m: float = POSITION_TOLERANCE_M,
        orientation_tolerance_rad: float = ORIENTATION_TOLERANCE_RAD,
        null_space_gain: float = NULL_SPACE_GAIN,
        rest_seeds: Optional[Sequence[Sequence[float]]] = None,
    ) -> None:
        """
        Args:
            side: ``"left"`` or ``"right"``.
            damping: The DLS damping factor.
            max_iterations: The DLS iteration cap.
            position_tolerance_m: The convergence tolerance for the position.
            orientation_tolerance_rad: The convergence tolerance for the orientation.
            null_space_gain: How strongly the null-space term pulls the solution
                back toward the seed. 0 disables it (the solver is then free to
                wander into another branch and can hit a joint limit).
            rest_seeds: Extra seeds to try when the primary seed fails. These are
                joint configurations in the arm's neighbourhood that are known to
                be reachable, used as fallbacks so a slightly awkward start pose
                does not make every subsequent step fail.
        """
        self.side = side
        self.damping = float(damping)
        self.max_iterations = int(max_iterations)
        self.position_tolerance_m = float(position_tolerance_m)
        self.orientation_tolerance_rad = float(orientation_tolerance_rad)
        self.null_space_gain = float(null_space_gain)
        self._kin = ArmKinematics(side)
        limits = arm_limits_rad(side)
        self._lower = np.asarray([lower for lower, _ in limits], dtype=float)
        self._upper = np.asarray([upper for _, upper in limits], dtype=float)
        # The configuration the null-space term pulls toward (the measured
        # joints for a real command). Set by :meth:`solve` before each attempt.
        self._seed = np.zeros(7)
        if rest_seeds is None:
            rest_seeds = [
                # A relaxed, mid-workspace posture (elbow bent, wrist level).
                (0.0, 0.2, 0.0, -0.6, 0.0, 0.0, 0.0),
                (0.0, -0.2, 0.0, -0.6, 0.0, 0.0, 0.0),
                (0.3, 0.4, 0.0, -0.9, 0.0, 0.0, 0.0),
            ]
        self._rest_seeds = [np.asarray(seed, dtype=float) for seed in rest_seeds]

    # -- public API ---------------------------------------------------------
    def solve(
        self,
        tcp_target7: Sequence[float],
        *,
        seed: Optional[Sequence[float]] = None,
        joint_positions: Optional[Sequence[float]] = None,
    ) -> Optional[np.ndarray]:
        """Solve for the joint configuration reaching the TCP target.

        Args:
            tcp_target7: The target TCP pose ``[x, y, z, qx, qy, qz, qw]`` in
                ``torso_link``.
            seed: The preferred starting configuration (normally the measured
                joints, so the arm moves as little as possible).
            joint_positions: An alias for ``seed`` (the measured joints), kept
                for symmetry with the transport's naming.

        Returns:
            The 7 joint positions (radians, URDF order) if the solver converged,
            else ``None``.
        """
        solution = self.solve_with_error(
            tcp_target7, seed=seed, joint_positions=joint_positions
        )
        return solution.joints if solution is not None and solution.ok else None

    def solve_with_error(
        self,
        tcp_target7: Sequence[float],
        *,
        seed: Optional[Sequence[float]] = None,
        joint_positions: Optional[Sequence[float]] = None,
    ) -> Optional[IKSolution]:
        """Like :meth:`solve`, but returns the residuals too.

        Returns ``None`` only when NO seed converged; otherwise the best
        (lowest-error) solution is returned, so the caller can inspect how close
        it got before deciding to use it.
        """
        target = np.asarray(tcp_target7, dtype=float).reshape(-1)
        if target.size != 7 or not np.all(np.isfinite(target)):
            raise ValueError("tcp_target7 must be 7 finite values")
        # Solve for the WRIST pose (the kinematics model's own output), keeping
        # the TCP's orientation (the wrist and TCP share it).
        wrist_target = tcp_to_wrist_pose7(target)
        position = wrist_target[:3]
        rotation = _quaternion_matrix(wrist_target[3:])

        seeds: list[np.ndarray] = []
        for candidate in (seed, joint_positions):
            if candidate is not None:
                seeds.append(self._clip(np.asarray(candidate, dtype=float).reshape(-1)))
        for rest in self._rest_seeds:
            seeds.append(self._clip(rest))

        best: Optional[IKSolution] = None
        for start in seeds:
            # The null-space term pulls toward THIS seed, so the solution stays
            # in the same neighbourhood as the joints we started from.
            self._seed = start
            joints, position_error, orientation_error = self._iterate(start, position, rotation)
            candidate = IKSolution(
                joints=joints,
                position_error_m=position_error,
                orientation_error_rad=orientation_error,
            )
            if candidate.ok:
                return candidate
            if best is None or (
                candidate.position_error_m + candidate.orientation_error_rad
                < best.position_error_m + best.orientation_error_rad
            ):
                best = candidate
        return best

    # -- internals ----------------------------------------------------------
    def _clip(self, joints: np.ndarray) -> np.ndarray:
        """Clamp a joint vector into the official G1-D limits."""
        return np.clip(joints, self._lower, self._upper)

    def _iterate(
        self, start: np.ndarray, position: np.ndarray, rotation: np.ndarray
    ) -> tuple[np.ndarray, float, float]:
        """Run the DLS iteration; return (joints, position error, orientation error).

        The two returned errors are the TRUE residuals (meters and radians). The
        quantity that DRIVES the iteration is a weighted one (see
        :meth:`_error`), so the convergence test and the reported residuals are
        computed separately -- comparing the weighted drive error against an
        unscaled tolerance would never converge.

        The step has TWO parts:

        1. the task step, ``J^+ e`` -- the joint change that reduces the pose
           error, and
        2. a NULL-SPACE step, ``N k (q_seed - q)`` -- which uses the arm's
           1-DoF redundancy to pull the solution back toward the seed.

        Without (2) the solver is free to wander into a different branch of the
        solution space, and on the real robot it would drive a joint into its
        LIMIT while trying to satisfy a 1 cm translation -- which is exactly what
        it did before this was added. With (2) the solution stays in the same
        neighbourhood as the measured joints.

        The task step is a damped-least-squares step with a SMALL, FIXED damping
        factor. Two things matter here, and both were wrong before:

        1. The damping must be small (1e-4). A large damping (1e-2) acts as a
           regulariser that pulls the step short of the true least-squares
           solution, so the solver stalls at a residual of a few tenths of a
           millimetre that extra iterations cannot remove.
        2. The convergence test uses the TRUE residuals against a target that
           is TIGHTER than the reported tolerance. The solver used to stop as
           soon as the residual fell under the tolerance, so it returned a
           solution that merely squeaked past the tolerance instead of the
           least-squares one -- and the caller had no way to tell the
           difference.
        """
        joints = self._clip(np.asarray(start, dtype=float).reshape(-1).copy())
        position_error = float("inf")
        orientation_error = float("inf")
        # The internal target is deliberately tighter than the tolerance that
        # decides whether the solution is usable, so the returned solution is
        # the least-squares one and not "just inside the tolerance".
        target_position = min(
            self.position_tolerance_m, CONVERGENCE_MARGIN * self.position_tolerance_m
        )
        target_orientation = min(
            self.orientation_tolerance_rad,
            CONVERGENCE_MARGIN * self.orientation_tolerance_rad,
        )
        for _ in range(self.max_iterations):
            drive = self._error(joints, position, rotation)
            position_error, orientation_error = self._residuals(joints, position, rotation)
            if (
                position_error <= target_position
                and orientation_error <= target_orientation
            ):
                break
            jacobian = self._jacobian(joints)
            # The Jacobian's rotational rows are in rad/rad while its positional
            # rows are in m/rad, so the two cannot be mixed unweighted. Scale the
            # rotational rows by 1/reach (turning radians into "equivalent
            # metres of tool travel") and scale the ERROR the same way, so the
            # least-squares problem is dimensionally consistent.
            reach = max(0.05, float(np.linalg.norm(self._kin.wrist_matrix(joints)[:3, 3])))
            weight = 1.0 / reach
            weighted_jacobian = jacobian.copy()
            weighted_jacobian[3:, :] *= weight
            weighted_drive = drive.copy()
            weighted_drive[3:] *= weight
            damped = weighted_jacobian @ weighted_jacobian.T + (self.damping ** 2) * np.eye(6)
            try:
                pseudo = weighted_jacobian.T @ np.linalg.solve(damped, weighted_drive)
            except np.linalg.LinAlgError:  # pragma: no cover - defensive
                break
            norm = float(np.linalg.norm(pseudo))
            step = (
                pseudo
                if norm <= MAX_JOINT_STEP_RAD
                else pseudo * (MAX_JOINT_STEP_RAD / norm)
            )
            candidate = self._clip(joints + step)
            candidate_position_error, candidate_orientation_error = self._residuals(
                candidate, position, rotation
            )
            # The null-space projector: with J being 6x7 (6 task rows, 7 joints),
            # the projector onto the null space of J is (I7 - J^+ J), where
            # J^+ = pinv(J) is 7x6. (J^T pinv(J) would be 7x6 and is a different
            # quantity -- the row space, not the null space.) It is built from
            # the UNWEIGHTED Jacobian: the weighting is a units convention for
            # the task step, and reusing it in the projector mixes metres and
            # radians in the null-space direction.
            #
            # The pull is GUARDED: it is only kept if it does not make the
            # residual worse. The null-space term only steers the redundant
            # DOF, so it must never compete with the task.
            null_space = np.eye(7) - np.linalg.pinv(jacobian) @ jacobian
            pull = null_space @ (self._seed - joints)
            null_step = self.null_space_gain * pull
            null_norm = float(np.linalg.norm(null_step))
            if null_norm > MAX_NULL_SPACE_STEP_RAD:
                null_step = null_step * (MAX_NULL_SPACE_STEP_RAD / null_norm)
            pulled = self._clip(candidate + null_step)
            pulled_position_error, pulled_orientation_error = self._residuals(
                pulled, position, rotation
            )
            if (
                pulled_position_error + pulled_orientation_error
                <= candidate_position_error + candidate_orientation_error
            ):
                joints = pulled
                position_error = pulled_position_error
                orientation_error = pulled_orientation_error
            else:
                joints = candidate
                position_error = candidate_position_error
                orientation_error = candidate_orientation_error
        # Report the residual of the joints actually returned, not the one from
        # the previous iteration.
        position_error, orientation_error = self._residuals(joints, position, rotation)
        return joints, position_error, orientation_error

    def _residuals(
        self, joints: np.ndarray, position: np.ndarray, rotation: np.ndarray
    ) -> tuple[float, float]:
        """The TRUE residuals: (position error in m, orientation error in rad)."""
        matrix = self._kin.wrist_matrix(joints)
        position_error = float(np.linalg.norm(position - matrix[:3, 3]))
        orientation_error = float(
            np.linalg.norm(_rotation_vector(rotation @ matrix[:3, :3].T))
        )
        return position_error, orientation_error

    def _error(
        self, joints: np.ndarray, position: np.ndarray, rotation: np.ndarray
    ) -> np.ndarray:
        """The raw 6-vector pose error (position in m, then orientation in rad)."""
        matrix = self._kin.wrist_matrix(joints)
        position_delta = position - matrix[:3, 3]
        orientation_delta = _rotation_vector(rotation @ matrix[:3, :3].T)
        return np.concatenate([position_delta, orientation_delta])
    def _jacobian(self, joints: np.ndarray) -> np.ndarray:
        """The 6x7 geometric Jacobian of the wrist pose, by finite differences.

        The rotational rows are the SPATIAL angular velocity
        (``omega = vee(skew(dR @ R^T))``), which is the quantity that pairs with
        the base-frame orientation error used by :meth:`_error`
        (``rotvec(R_target @ R^T)``). Using the BODY-frame skew
        (``vee(skew(dR))``) here instead is a frame mismatch: the error and the
        Jacobian then describe different rotations, so the DLS step drives the
        wrong direction. That mismatch was the reason a 1 cm Cartesian step from
        the live pose stalled with ~270 mm / ~59 deg of residual instead of
        converging.

        The skew form is used rather than ``rotvec(dR)`` because the difference
        of two rotation matrices is NOT itself a rotation matrix, and taking a
        rotation vector of it loses accuracy.
        """
        jacobian = np.zeros((6, 7))
        rotation = self._kin.wrist_matrix(joints)[:3, :3]
        for index in range(7):
            forward = joints.copy()
            backward = joints.copy()
            forward[index] += JACOBIAN_EPS_RAD
            backward[index] -= JACOBIAN_EPS_RAD
            forward_matrix = self._kin.wrist_matrix(forward)
            backward_matrix = self._kin.wrist_matrix(backward)
            jacobian[:3, index] = (
                forward_matrix[:3, 3] - backward_matrix[:3, 3]
            ) / (2.0 * JACOBIAN_EPS_RAD)
            delta = (
                forward_matrix[:3, :3] - backward_matrix[:3, :3]
            ) / (2.0 * JACOBIAN_EPS_RAD)
            spatial = delta @ rotation.T
            jacobian[3:, index] = np.array(
                [
                    spatial[2, 1] - spatial[1, 2],
                    spatial[0, 2] - spatial[2, 0],
                    spatial[1, 0] - spatial[0, 1],
                ]
            ) / 2.0
        return jacobian


def _quaternion_matrix(quaternion: Sequence[float]) -> np.ndarray:
    """Unit quaternion (x, y, z, w) -> rotation matrix."""
    x, y, z, w = (float(v) for v in np.asarray(quaternion, dtype=float).reshape(-1))
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1e-12:
        raise ValueError("quaternion must be non-zero")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
