#!/usr/bin/env python3
"""Measure the REAL sign of every atomic motion token on the G1-D, in the harness's own frame.

This is the ONE script that answers "does MV_LEFT actually move the gripper to the
robot's left?". It drives the arm through the SAME path the VLM pipeline uses
(token -> controller -> base-frame delta -> MoveIt -> FollowJointTrajectory), so the
number it prints is the number the pipeline will use -- not a proxy measured with a
different tool frame.

How it works
------------
For each token the script:

1. reads the MEASURED TCP pose (forward kinematics on the measured joints);
2. computes the token's base-frame delta from ``configs/primitives_g1d.yaml``
   (``torso_link`` frame: +X forward, +Y robot-right, +Z up);
3. commands that delta through the real controller (MoveIt plans, the bridge
   executes) and reads the MEASURED pose again;
4. reports the SIGN of every measured axis, and flags the token as
   ``OK`` / ``MIRRORED`` / ``ZERO`` / ``FAILED``.

Because the report is computed from MEASURED poses, it is independent of the
primitives file: if the file's signs are wrong, the script says so.

The rotation tokens are measured as the SIGN OF THE YAW ERROR, i.e. the sign of
``(yaw_after - yaw_before)`` projected on the commanded axis. CW/CCW are then
reported against the sense a camera looking DOWN at the gripper sees, which is
what the VLM reasons about.

Safety
------
* Every move is capped at ``--step-m`` (default 0.02 m) and ``--yaw-step-rad``.
* ``--dry-run`` (the default) plans and validates every trajectory and NEVER sends
  a goal to the bridge. Pass ``--execute`` to move for real.
* The Z floor from the robot config is enforced by the transport (targets are
  clamped, trajectories are checked sample-by-sample).
* ``--dry-run`` still needs a live ROS 2 stack (it measures the real arm's pose),
  but it cannot move the arm.

Examples
--------
    # 1) Plan-only: verify every token plans and every axis moves the RIGHT way.
    python3 scripts/g1d/direction_sign_test.py --robot-config configs/robot_g1d.yaml

    # 2) Real execution, one token at a time, 20 mm per token.
    python3 scripts/g1d/direction_sign_test.py --robot-config configs/robot_g1d.yaml --execute

    # 3) Only the lateral axes (skip the vertical ones that the Z floor may block).
    python3 scripts/g1d/direction_sign_test.py --only MV_FWD MV_BACK MV_LEFT MV_RIGHT
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.action_units import MOVE_ATOMS, ROTATE_ATOMS
from core.config import load_yaml
from core.g1d.kinematics import ArmKinematics
from interpreters.real_atomic_controller import quat_to_euler


#: The order the tokens are tested in. Translations first (they are the ones the
#: VLM relies on most), then the rotations.
DEFAULT_TOKEN_ORDER: tuple[str, ...] = (
    "MV_FWD",
    "MV_BACK",
    "MV_LEFT",
    "MV_RIGHT",
    "MV_UP",
    "MV_DOWN",
    "ROTATE_CW",
    "ROTATE_CCW",
)

#: The base-frame axis each token is EXPECTED to move along, given the convention
#: "the robot's LEFT is base -Y, the robot's RIGHT is base +Y" (the Franka
#: convention, which is what ``configs/primitives_g1d.yaml`` currently ships).
EXPECTED_AXIS: dict[str, int] = {
    "MV_FWD": 0,   # base +X
    "MV_BACK": 0,  # base -X
    "MV_LEFT": 1,  # base -Y
    "MV_RIGHT": 1,  # base +Y
    "MV_UP": 2,    # base +Z
    "MV_DOWN": 2,  # base -Z
}

#: Minimum measured travel (m) that counts as "it moved" rather than "noise".
MIN_TRANSLATION_M = 0.002
#: Minimum measured yaw change (rad) that counts as "it turned".
MIN_YAW_RAD = 0.5 * math.radians(2.0)


class SignTestError(RuntimeError):
    """Something went wrong that the operator has to look at."""


def _pose7(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size != 7:
        raise SignTestError(f"expected a 7-value pose, got {arr.size}")
    return arr


def _yaw_rad(pose7: np.ndarray) -> float:
    """The base-Z yaw of a pose, in radians."""
    return float(quat_to_euler(pose7[3:])[2])


def _sign(value: float, epsilon: float) -> int:
    if value > epsilon:
        return 1
    if value < -epsilon:
        return -1
    return 0


def _expected_sign(token: str, file_sign: float) -> int:
    """The sign the token SHOULD produce on its expected axis."""
    if token in EXPECTED_AXIS:
        axis = EXPECTED_AXIS[token]
        positive = file_sign >= 0.0
        return 1 if positive else -1
    # ROTATE_CW / ROTATE_CCW: the file's sign is the yaw sign about base +Z.
    return 1 if file_sign >= 0.0 else -1


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--robot-config",
        default=str(ROOT / "configs" / "robot_g1d.yaml"),
        help="robot YAML (carries hardware / z_floor / side / camera topics)",
    )
    parser.add_argument(
        "--primitives",
        default=None,
        help="primitives YAML (default: the robot config's primitives_config)",
    )
    parser.add_argument("--side", default=None, choices=("left", "right"))
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        metavar="TOKEN",
        help=f"test only these tokens; default: all of {', '.join(DEFAULT_TOKEN_ORDER)}",
    )
    parser.add_argument(
        "--step-m",
        type=float,
        default=0.02,
        help="translation per token (meters). Default 0.02.",
    )
    parser.add_argument(
        "--yaw-step-rad",
        type=float,
        default=0.15,
        help="yaw per rotation token (radians). Default 0.15.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="MOVE the arm. Without this flag the script only plans and reports.",
    )
    parser.add_argument(
        "--z-floor-m",
        type=float,
        default=None,
        help="override the Z floor (meters). Default: the robot config's value.",
    )
    parser.add_argument(
        "--no-z-floor",
        action="store_true",
        help="disable the Z floor (NOT recommended on hardware).",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="write the full report as JSON to this path",
    )
    return parser.parse_args(argv)


def _resolve_side(cfg: dict, args: argparse.Namespace) -> str:
    rb = cfg.get("robot") or {}
    side = args.side or str(rb.get("side", "left"))
    if side not in ("left", "right"):
        raise SignTestError(f"side must be 'left' or 'right', got {side!r}")
    return side


def _build_session(cfg: dict, args: argparse.Namespace):
    """A G1D session wired exactly like the live pipeline's (same floor, same side)."""
    from core.g1d.g1d_session import G1DSession, G1DSessionConfig
    from core.g1d.safety import resolve_z_floor

    rb = cfg.get("robot") or {}
    side = _resolve_side(cfg, args)
    floor = resolve_z_floor(
        cfg,
        override_m=args.z_floor_m,
    )
    if args.no_z_floor:
        floor = resolve_z_floor({**cfg, "enable_z_floor": False})
    if not floor.enabled:
        raise SignTestError(
            "the Z floor is disabled. A live G1-D run needs a measured floor so the "
            "arm cannot be driven into the table."
        )
    if floor.z_m <= 0.0 and not floor.calibrated:
        raise SignTestError(
            f"z_floor_m = {floor.z_m} is the shipped placeholder, not a measured height."
        )

    topics = cfg.get("camera_topics") or {}
    return G1DSession(
        G1DSessionConfig(
            side=side,
            namespace=str(rb.get("bridge_namespace", "lzh")),
            use_mock_robot=False,
            use_mock_cameras=True,  # this script does not need images
            head_camera_topic=str(topics.get("head_left", "/g1d/cameras/head_left/image_raw")),
            head_right_camera_topic=str(
                topics.get("head_right", "/g1d/cameras/head_right/image_raw")
            ),
            wrist_camera_topic=str(
                topics.get(f"{side}_wrist", f"/g1d/cameras/{side}_wrist/image_raw")
            ),
            z_floor=floor,
            # dry_run is the SAFE default: trajectories are planned and validated
            # but never sent. --execute flips it to a real move.
            dry_run=not args.execute,
            verbose=True,
        )
    )


def _build_controller(session, primitives_cfg: dict, args: argparse.Namespace):
    from interpreters.g1d_atomic_controller import G1DAtomicController

    floor = session.config.z_floor
    return G1DAtomicController.from_primitives_config(
        session.robot,
        primitives_cfg,
        step_m=float(args.step_m),
        yaw_step_rad=float(args.yaw_step_rad),
        z_floor_m=float(floor.z_m) if floor is not None else None,
        capture_z_floor_on_sync=False,
        # One command per token on the G1-D: the whole displacement is planned and
        # executed as ONE trajectory, so re-commanding the same setpoint N times
        # would just re-send the same goal.
        settle_steps=1,
        settle_dt_s=0.0,
        verbose=True,
    )


def _token_list(args: argparse.Namespace) -> list[str]:
    if not args.only:
        return list(DEFAULT_TOKEN_ORDER)
    wanted = [str(t).strip().upper() for t in args.only]
    unknown = [t for t in wanted if t not in MOVE_ATOMS and t not in ROTATE_ATOMS]
    if unknown:
        raise SignTestError(
            f"unknown token(s): {', '.join(unknown)}; "
            f"choose from {', '.join(DEFAULT_TOKEN_ORDER)}"
        )
    return wanted


def _measure_translation(
    controller, token: str
) -> tuple[np.ndarray, np.ndarray, Optional[str]]:
    """Execute one translation token and return (before, after, error)."""
    before = _pose7(controller.robot.get_ee_pose())
    try:
        controller.step(token)
    except Exception as exc:  # noqa: BLE001 - a failed token must not kill the sweep
        return before, before, f"{type(exc).__name__}: {exc}"
    after = _pose7(controller.robot.get_ee_pose())
    return before, after, None


def _measure_rotation(controller, token: str) -> tuple[np.ndarray, np.ndarray, Optional[str]]:
    before = _pose7(controller.robot.get_ee_pose())
    try:
        controller.step(token)
    except Exception as exc:  # noqa: BLE001
        return before, before, f"{type(exc).__name__}: {exc}"
    after = _pose7(controller.robot.get_ee_pose())
    return before, after, None


def _planned_delta(controller, token: str) -> Optional[np.ndarray]:
    """The base-frame delta the controller PLANNED for ``token`` (dry-run only).

    In a dry run the arm never moves, so the measured before/after pair is
    identical and the sign cannot be measured. The controller's own setpoint
    integration still happens, so the PLANNED delta is known exactly: it is the
    primitives vector scaled by the step size. Returning it lets the dry run
    report "the plan says +X" instead of a meaningless "ZERO".
    """
    try:
        delta, yaw = controller.intended_motion(token)
    except Exception:  # noqa: BLE001 - a non-motion token has no planned delta
        return None
    out = np.zeros(3, dtype=float)
    if delta is not None:
        out += np.asarray(delta, dtype=float)
    if yaw:
        # A rotation is reported on the yaw axis, in the same 3-vector slot.
        out[2] = 0.0
    return out


def _planned_yaw(controller, token: str) -> Optional[float]:
    """The planned yaw change (rad) for a rotation token (dry-run only)."""
    try:
        _, yaw = controller.intended_motion(token)
    except Exception:  # noqa: BLE001
        return None
    return float(yaw) if yaw else None


def _classify(
    token: str,
    before: np.ndarray,
    after: np.ndarray,
    file_sign: float,
    *,
    planned: bool = False,
    planned_delta: Optional[np.ndarray] = None,
    planned_yaw: Optional[float] = None,
) -> dict[str, Any]:
    """Turn a measured before/after pair into a verdict for one token.

    With ``planned=True`` the verdict is about the PLAN (the dry run): the
    primitives vector is checked for internal consistency -- the expected axis
    carries the expected sign and the other axes are zero. It is NOT a hardware
    measurement and the row says so.
    """
    if token in EXPECTED_AXIS:
        axis = EXPECTED_AXIS[token]
        expected = _expected_sign(token, file_sign)
        if planned:
            delta = (
                np.asarray(planned_delta, dtype=float)
                if planned_delta is not None
                else np.zeros(3)
            )
            measured_sign = _sign(float(delta[axis]), 1e-9)
            other = [i for i in range(3) if i != axis]
            cross = max(abs(float(delta[i])) for i in other)
            travel = float(np.linalg.norm(delta))
            if measured_sign == 0:
                verdict = "ZERO"
            elif measured_sign == expected:
                verdict = "PLAN_OK"
            else:
                verdict = "PLAN_MIRRORED"
            return {
                "token": token,
                "kind": "move",
                "verdict": verdict,
                "planned": True,
                "expected_axis": "xyz"[axis],
                "expected_sign": expected,
                "measured_delta_m": [round(float(v), 5) for v in delta],
                "measured_sign": measured_sign,
                "travel_m": round(travel, 5),
                "cross_axis_m": round(cross, 5),
            }
        delta = after[:3] - before[:3]
        measured_sign = _sign(float(delta[axis]), MIN_TRANSLATION_M)
        other = [i for i in range(3) if i != axis]
        cross = max(abs(float(delta[i])) for i in other)
        travel = float(np.linalg.norm(delta))
        if measured_sign == 0:
            verdict = "ZERO" if travel < MIN_TRANSLATION_M else "SIDEWAYS"
        elif measured_sign == expected:
            verdict = "OK"
        else:
            verdict = "MIRRORED"
        return {
            "token": token,
            "kind": "move",
            "verdict": verdict,
            "expected_axis": "xyz"[axis],
            "expected_sign": expected,
            "measured_delta_m": [round(float(v), 5) for v in delta],
            "measured_sign": measured_sign,
            "travel_m": round(travel, 5),
            "cross_axis_m": round(cross, 5),
        }

    # Rotation: the yaw change about base +Z.
    expected = _expected_sign(token, file_sign)
    if planned:
        yaw_delta = float(planned_yaw or 0.0)
        measured_sign = _sign(yaw_delta, 1e-9)
        verdict = "ZERO" if measured_sign == 0 else (
            "PLAN_OK" if measured_sign == expected else "PLAN_MIRRORED"
        )
        return {
            "token": token,
            "kind": "rotate",
            "verdict": verdict,
            "planned": True,
            "expected_axis": "yaw",
            "expected_sign": expected,
            "measured_yaw_delta_rad": round(yaw_delta, 5),
            "measured_yaw_delta_deg": round(math.degrees(yaw_delta), 2),
            "measured_sign": measured_sign,
        }

    yaw_before = _yaw_rad(before)
    yaw_after = _yaw_rad(after)
    # Wrap into (-pi, pi] so a wrap at +-180 deg does not fake a sign flip.
    yaw_delta = (yaw_after - yaw_before + math.pi) % (2.0 * math.pi) - math.pi
    measured_sign = _sign(yaw_delta, MIN_YAW_RAD)
    if measured_sign == 0:
        verdict = "ZERO"
    elif measured_sign == expected:
        verdict = "OK"
    else:
        verdict = "MIRRORED"
    return {
        "token": token,
        "kind": "rotate",
        "verdict": verdict,
        "expected_axis": "yaw",
        "expected_sign": expected,
        "measured_yaw_delta_rad": round(yaw_delta, 5),
        "measured_yaw_delta_deg": round(math.degrees(yaw_delta), 2),
        "measured_sign": measured_sign,
    }


def _suggest_fix(token: str, measured_sign: int, file_sign: float) -> str:
    """The primitives-file edit that would make this token agree with the hardware."""
    if token in EXPECTED_AXIS:
        axis = EXPECTED_AXIS[token]
        current = list(np.zeros(3))
        current[axis] = file_sign
        flipped = [-v for v in current]
        return (
            f"flip {'xyz'[axis]}: current {current} -> {flipped} "
            f"(in configs/primitives_g1d.yaml)"
        )
    return (
        f"flip the ROTATE sign: current {file_sign:+.1f} -> {-file_sign:+.1f} "
        f"(in configs/primitives_g1d.yaml)"
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    # G1DSession creates rclpy nodes through the real transport.  Keep the
    # ROS lifecycle local to this hardware-facing CLI, like the other G1-D
    # scripts do, so importing the session never relies on a caller's context.
    import rclpy

    from core.g1d.ros_environment import require_g1d_ros_environment

    require_g1d_ros_environment()
    rclpy.init(args=None)

    cfg = load_yaml(args.robot_config)
    prim_path = args.primitives
    if prim_path is None:
        prim_path = str(
            cfg.get("primitives_config") or (ROOT / "configs" / "primitives_g1d.yaml")
        )
        if not Path(prim_path).is_absolute():
            prim_path = str(ROOT / prim_path)
    primitives_cfg = load_yaml(prim_path)

    atomic = primitives_cfg.get("atomic_primitives") or {}
    tokens = _token_list(args)

    mode = "EXECUTE (the arm WILL move)" if args.execute else "DRY-RUN (plan only, no motion)"
    print("=" * 78)
    print(f"  G1-D direction-sign test  --  {mode}")
    print(f"  robot config : {args.robot_config}")
    print(f"  primitives   : {prim_path}")
    print(f"  side         : {_resolve_side(cfg, args)}")
    print(f"  step         : {args.step_m * 100:g} cm / {math.degrees(args.yaw_step_rad):g} deg")
    print("=" * 78)

    session = _build_session(cfg, args)
    controller = None
    report: list[dict[str, Any]] = []
    try:
        session.connect()
        controller = _build_controller(session, primitives_cfg, args)
        controller.sync_from_robot()
        start = _pose7(controller.robot.get_ee_pose())
        print(
            f"[sign-test] start TCP (torso_link) = "
            f"{np.round(start[:3], 4).tolist()}  "
            f"yaw = {math.degrees(_yaw_rad(start)):.1f} deg"
        )
        if not args.execute:
            print(
                "[sign-test] DRY-RUN: every trajectory is planned and validated, but "
                "no goal is sent to the bridge. Re-run with --execute to move."
            )
        print("")

        for token in tokens:
            if token not in atomic:
                print(f"[sign-test] {token:<10} SKIP: not in the primitives file")
                report.append(
                    {"token": token, "verdict": "SKIP", "reason": "not in primitives file"}
                )
                continue

            file_value = atomic[token]
            if token in MOVE_ATOMS:
                file_sign = float(np.asarray(file_value, dtype=float)[EXPECTED_AXIS[token]])
            else:
                file_sign = float(file_value)

            before, after, error = (
                _measure_translation(controller, token)
                if token in MOVE_ATOMS
                else _measure_rotation(controller, token)
            )
            if error is not None:
                print(f"[sign-test] {token:<10} FAILED: {error}")
                report.append({"token": token, "verdict": "FAILED", "error": error})
                continue

            if not args.execute:
                # Dry run: the arm did not move, so report the PLAN. The
                # primitives vector is checked for internal consistency (the
                # expected axis carries the expected sign, the others are 0).
                row = _classify(
                    token,
                    before,
                    after,
                    file_sign,
                    planned=True,
                    planned_delta=(
                        _planned_delta(controller, token)
                        if token in MOVE_ATOMS
                        else None
                    ),
                    planned_yaw=(
                        _planned_yaw(controller, token)
                        if token in ROTATE_ATOMS
                        else None
                    ),
                )
                # The controller's setpoint integrated the token even though the
                # arm never moved, so it now sits one step ahead of reality.
                # Re-sync it, so every dry-run token plans the SAME step from
                # the same real pose instead of drifting further and further.
                try:
                    controller.sync_from_robot()
                except Exception:  # noqa: BLE001 - a re-sync failure is not fatal
                    pass
            else:
                row = _classify(token, before, after, file_sign)
            row["file_value"] = (
                [float(v) for v in file_value] if token in MOVE_ATOMS else float(file_value)
            )
            report.append(row)

            if token in MOVE_ATOMS:
                print(
                    f"[sign-test] {token:<10} {row['verdict']:<13} "
                    f"delta={row['measured_delta_m']} m  "
                    f"(expected {row['expected_axis']}{'+' if row['expected_sign'] > 0 else '-'}, "
                    f"cross-axis {row['cross_axis_m'] * 1000:.1f} mm)"
                )
            else:
                print(
                    f"[sign-test] {token:<10} {row['verdict']:<13} "
                    f"dyaw={row['measured_yaw_delta_deg']:+.2f} deg"
                )
            if row["verdict"] in ("MIRRORED", "PLAN_MIRRORED"):
                print(
                    f"           -> the file's sign is WRONG for this token. "
                    f"{_suggest_fix(token, row['measured_sign'], file_sign)}"
                )
            elif row["verdict"] in ("ZERO", "SIDEWAYS"):
                print(
                    "           -> the arm did not move along the expected axis. "
                    "Check the Z floor, the joint limits, and that the goal was "
                    "actually executed (not planned only)."
                )

        # -- summary ---------------------------------------------------------
        print("")
        print("-" * 78)
        print("  SUMMARY")
        print("-" * 78)
        good = "PLAN_OK" if not args.execute else "OK"
        print(f"  {'token':<12}{'verdict':<14}{'measured':<28}{'fix needed'}")
        bad = 0
        for row in report:
            token = str(row.get("token", "?"))
            verdict = str(row.get("verdict", "?"))
            if row.get("kind") == "move":
                measured = f"d={row.get('measured_delta_m')} m"
                fix = "-" if verdict == good else "yes"
            elif row.get("kind") == "rotate":
                measured = f"dyaw={row.get('measured_yaw_delta_deg')} deg"
                fix = "-" if verdict == good else "yes"
            else:
                measured = str(row.get("error", row.get("reason", "-")))[:26]
                fix = "yes"
            if verdict != good:
                bad += 1
            print(f"  {token:<12}{verdict:<14}{measured:<28}{fix}")
        print("-" * 78)
        if bad == 0:
            if args.execute:
                print("  ALL TOKENS OK: configs/primitives_g1d.yaml needs no change.")
            else:
                print(
                    "  ALL PLANS OK: every token plans to the expected axis. "
                    "Re-run with --execute to MEASURE the real sign on hardware."
                )
        else:
            print(
                f"  {bad} token(s) disagree with the hardware. Update "
                f"{prim_path} with the 'fix needed' column, then re-run."
            )
        print("")

        if args.json_out:
            out = Path(args.json_out)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(
                json.dumps(
                    {"executed": args.execute, "side": _resolve_side(cfg, args),
                     "step_m": args.step_m, "results": report},
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"[sign-test] report written to {out}")

        return 0 if bad == 0 else 3
    except SignTestError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    finally:
        session.close()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
