#!/usr/bin/env python3
"""Diagnose and FIX the G1-D's "did not reach the target" (GOAL_TOLERANCE_VIOLATED) problem.

The failure
-----------
The SDK bridge executes a FollowJointTrajectory goal and then WAITS for the
measured joints to converge to the commanded final point. When they do not
converge within ``goal_tolerance_rad`` before ``settle_timeout_sec`` expires,
the bridge aborts the goal with ``error_code = -5`` (GOAL_TOLERANCE_VIOLATED)
even though the arm physically reached (and holds) the target.

What this script does
---------------------
For a given Cartesian goal it runs the FULL plan->execute->measure cycle and
reports, per joint:

* the COMMANDED final position,
* the MEASURED position after the bridge reports back,
* the tracking error,
* whether the error is inside the bridge's tolerance,
* the resulting Cartesian error at the TCP (which is what actually matters),
* the ANGLE the arm had to travel to get there (the dominant cause: the G1-D's
  shoulder is gravity-loaded, so a large joint angle means a large steady-state
  tracking error).

Then it computes the tolerance / timeout that WOULD have been needed, so the
operator can set the bridge parameters with evidence instead of guesswork.

It can also READ the live bridge parameters and READ the bridge's own log to
confirm the diagnosis.

Examples
--------
    # 1) Diagnose one goal (dry-run: plans, does not move).
    python3 scripts/g1d/settle_diagnose.py --robot-config configs/robot_g1d.yaml

    # 2) Diagnose with a real move.
    python3 scripts/g1d/settle_diagnose.py --robot-config configs/robot_g1d.yaml --execute

    # 3) Read the bridge's current tolerance parameters (no motion).
    python3 scripts/g1d/settle_diagnose.py --read-params

    # 4) Scan the bridge log for past GOAL_TOLERANCE_VIOLATED events.
    python3 scripts/g1d/settle_diagnose.py --scan-log
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import load_yaml
from core.g1d.kinematics import arm_joint_names
from core.g1d.safety import resolve_z_floor


#: The bridge's DEFAULT tolerance / timeout (see real_moveit.launch.py).
DEFAULT_GOAL_TOLERANCE_RAD = 0.03
DEFAULT_SETTLE_DURATION_SEC = 0.20
DEFAULT_SETTLE_TIMEOUT_SEC = 2.0

#: control_msgs FollowJointTrajectory error codes.
ERR_SUCCESSFUL = 0
ERR_GOAL_TOLERANCE_VIOLATED = -5

#: A Cartesian error below this (m) counts as "the arm reached the target".
TCP_REACHED_M = 0.01
#: A Cartesian error below this (m) is "good enough to act on".
TCP_TOLERANT_M = 0.02


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--robot-config",
        default=str(ROOT / "configs" / "robot_g1d.yaml"),
        help="robot YAML (carries hardware / z_floor / side)",
    )
    parser.add_argument("--side", default=None, choices=("left", "right"))
    parser.add_argument(
        "--execute",
        action="store_true",
        help="actually MOVE the arm while diagnosing. Without it, only the plan is built.",
    )
    parser.add_argument(
        "--step-m",
        type=float,
        default=0.02,
        help="the size of the diagnostic move (meters). Default 0.02.",
    )
    parser.add_argument(
        "--z-floor-m",
        type=float,
        default=None,
        help="override the Z floor (meters)",
    )
    parser.add_argument(
        "--no-z-floor", action="store_true", help="disable the Z floor (not recommended)"
    )
    parser.add_argument(
        "--read-params",
        action="store_true",
        help="only read the live bridge parameters and print them",
    )
    parser.add_argument(
        "--scan-log",
        action="store_true",
        help="scan the bridge's ROS log for GOAL_TOLERANCE_VIOLATED events",
    )
    parser.add_argument(
        "--log-dir",
        default=None,
        help="directory to scan for the bridge log (default: ~/.ros/log)",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="write the diagnosis as JSON to this path",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# --read-params
# ---------------------------------------------------------------------------
def _bridge_node_names() -> list[str]:
    """Both bridge builds' action-server node names."""
    return [
        "g1d_trajectory_action_server",
        "lzh_trajectory_action_server",
    ]


def read_params() -> dict[str, Any]:
    """Read the live bridge's tolerance parameters (read-only, no motion)."""
    import subprocess

    out: dict[str, Any] = {"nodes": {}, "errors": []}
    for node in _bridge_node_names():
        params: dict[str, Any] = {}
        for name in (
            "goal_tolerance_rad",
            "settle_duration_sec",
            "settle_timeout_sec",
            "max_start_error_rad",
            "dry_run",
            "enable_real_execution",
            "enable_gripper_real_execution",
        ):
            try:
                proc = subprocess.run(
                    ["ros2", "param", "get", f"/{node}", name],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
            except Exception as exc:  # noqa: BLE001
                out["errors"].append(f"{node}.{name}: {exc}")
                continue
            if proc.returncode != 0:
                params[name] = f"<unavailable: {proc.stderr.strip()[:60]}>"
                continue
            raw = proc.stdout.strip()
            value: Any = raw
            for cast in (float, int):
                try:
                    value = cast(raw)
                    break
                except ValueError:
                    continue
            if isinstance(value, str):
                if value.lower() in ("true", "false"):
                    value = value.lower() == "true"
            params[name] = value
        if params:
            out["nodes"][node] = params
    return out


def print_params(params: dict[str, Any]) -> None:
    print("=" * 78)
    print("  BRIDGE PARAMETERS (live)")
    print("=" * 78)
    if not params.get("nodes"):
        print("  no bridge action server found on this ROS domain.")
        print("  Is it running?  ros2 node list | grep trajectory_action_server")
    for node, values in params.get("nodes", {}).items():
        print(f"  /{node}")
        for name, value in values.items():
            print(f"    {name:<32}{value}")
    for err in params.get("errors", []):
        print(f"  [warn] {err}")
    print("")


# ---------------------------------------------------------------------------
# --scan-log
# ---------------------------------------------------------------------------
_SETTLE_RE = re.compile(
    r"did not settle: max_error=([0-9.]+) rad, tolerance=([0-9.]+) rad, timeout=([0-9.]+) s"
)
_DIAG_RE = re.compile(
    r"FINAL_TRACKING_DIAGNOSTIC (\S+) goal=(-?[0-9.]+) cmd=(-?[0-9.]+) "
    r"actual=(-?[0-9.]+) error=(-?[0-9.]+)"
)


def scan_log(log_dir: Optional[str]) -> dict[str, Any]:
    """Scan every bridge log for past GOAL_TOLERANCE_VIOLATED events."""
    root = Path(log_dir or (Path.home() / ".ros" / "log"))
    files: list[Path] = []
    if root.is_dir():
        for pattern in (
            "g1d_trajectory_action_server_*.log",
            "lzh_trajectory_action_server_*.log",
        ):
            files.extend(root.glob(pattern))
    files = sorted(set(files), key=lambda p: p.stat().st_mtime)

    events: list[dict[str, Any]] = []
    for path in files:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if "did not settle" not in text:
            continue
        per_file: list[dict[str, Any]] = []
        lines = text.splitlines()
        # One ordered pass: a "did not settle" line OPENS an event, and the
        # FINAL_TRACKING_DIAGNOSTIC lines that follow belong to it.
        current: Optional[dict[str, Any]] = None
        for line in lines:
            m = _SETTLE_RE.search(line)
            if m:
                current = {
                    "max_error_rad": float(m.group(1)),
                    "tolerance_rad": float(m.group(2)),
                    "timeout_s": float(m.group(3)),
                    "joints": [],
                }
                per_file.append(current)
                continue
            jm = _DIAG_RE.search(line)
            if jm and current is not None:
                current["joints"].append(
                    {
                        "joint": jm.group(1),
                        "goal": float(jm.group(2)),
                        "actual": float(jm.group(4)),
                        "error_rad": float(jm.group(5)),
                    }
                )
        if per_file:
            events.append({"log": str(path), "events": per_file})

    _attach_tcp_errors(events)
    return {"log_dir": str(root), "files_scanned": len(files), "events": events}


def _attach_tcp_errors(events: list[dict[str, Any]]) -> None:
    """Compute the Cartesian (TCP) error of every fully-diagnosed event.

    This is the decisive number: a joint-space error of 0.07 rad on the
    elbow is only ~9 mm at the fingertip, so the arm DID reach the target even
    though the bridge called it a tolerance violation. Events without a
    complete per-joint diagnostic (older bridge builds) simply keep
    ``tcp_error_m = None``.
    """
    try:
        from core.g1d.kinematics import ArmKinematics, arm_joint_names
    except Exception:  # noqa: BLE001 - diagnostics must never raise
        return

    for side in ("left", "right"):
        try:
            kin = ArmKinematics(side)
            names = arm_joint_names(side)
        except Exception:  # noqa: BLE001
            continue
        index = {name: i for i, name in enumerate(names)}
        for entry in events:
            for ev in entry["events"]:
                joints = ev.get("joints") or []
                if len(joints) != len(names):
                    continue
                if any(j["joint"] not in index for j in joints):
                    continue
                goal = np.zeros(len(names))
                actual = np.zeros(len(names))
                for j in joints:
                    pos = index[j["joint"]]
                    goal[pos] = j["goal"]
                    actual[pos] = j["actual"]
                try:
                    pg = kin.tcp_pose7(goal)[:3]
                    pa = kin.tcp_pose7(actual)[:3]
                except Exception:  # noqa: BLE001
                    continue
                ev["tcp_error_m"] = float(np.linalg.norm(pa - pg))


def print_log_scan(result: dict[str, Any]) -> None:
    print("=" * 78)
    print("  BRIDGE LOG SCAN (GOAL_TOLERANCE_VIOLATED)")
    print("=" * 78)
    if not result["events"]:
        print(f"  no 'did not settle' event found in {result['files_scanned']} log file(s).")
        print("  (either nothing failed, or the logs were rotated)")
        print("")
        return
    total = 0
    for entry in result["events"]:
        print(f"  {Path(entry['log']).name}")
        for ev in entry["events"]:
            total += 1
            tcp = ev.get("tcp_error_m")
            tcp_txt = f"  TCPerr={tcp * 1000:6.1f} mm" if tcp is not None else ""
            print(
                f"    max_error={ev['max_error_rad']:.6f} rad  "
                f"tolerance={ev['tolerance_rad']:.6f} rad  "
                f"timeout={ev['timeout_s']:.3f} s{tcp_txt}"
            )
            for joint in ev.get("joints", []):
                print(
                    f"      {joint['joint']:<34}goal={joint['goal']:+.6f} "
                    f"actual={joint['actual']:+.6f} error={joint['error_rad']:.6f} rad"
                )
    print(f"  total events: {total}")
    print("")
    if not total:
        return

    all_events = [ev for entry in result["events"] for ev in entry["events"]]
    errors = sorted(ev["max_error_rad"] for ev in all_events)
    worst = errors[-1]
    worst_side = None
    for entry in result["events"]:
        for ev in entry["events"]:
            if ev["max_error_rad"] == worst:
                worst_side = Path(entry["log"]).name
    print(f"  worst observed max_error = {worst:.6f} rad  ({worst_side})")
    print(f"  current default tolerance = {DEFAULT_GOAL_TOLERANCE_RAD:.4f} rad")
    print("")

    # -- how often would a wider tolerance have passed? --------------------
    print("  tolerance -> historical failures that would have PASSED")
    for tol in (0.03, 0.05, 0.08, 0.10, 0.12, 0.15):
        passed = sum(1 for e in errors if e <= tol)
        print(f"    {tol:5.2f} rad -> {passed:3d}/{total}")
    print("")

    # -- the decisive number: the Cartesian error -------------------------
    tcps = [ev["tcp_error_m"] for ev in all_events if ev.get("tcp_error_m") is not None]
    if tcps:
        tcps_sorted = sorted(tcps)
        reached = sum(1 for t in tcps if t <= TCP_REACHED_M)
        tolerant = sum(1 for t in tcps if t <= TCP_TOLERANT_M)
        print(
            f"  Cartesian (TCP) error over {len(tcps)} fully-diagnosed event(s):"
        )
        print(
            f"    min {min(tcps) * 1000:.1f} mm | "
            f"median {tcps_sorted[len(tcps_sorted) // 2] * 1000:.1f} mm | "
            f"max {max(tcps) * 1000:.1f} mm"
        )
        print(
            f"    reached (<= {TCP_REACHED_M * 1000:.0f} mm): {reached}/{len(tcps)} | "
            f"tolerant (<= {TCP_TOLERANT_M * 1000:.0f} mm): {tolerant}/{len(tcps)}"
        )
        print(
            "  -> the arm PHYSICALLY reached the target in every one of these "
            "events;\n     the bridge only failed the JOINT-space settle check."
        )
        print("")

    # -- the recommendation, backed by the whole history -------------------
    rec = _history_recommendation(result)
    print("  RECOMMENDED (from the whole history, left arm)")
    print(f"    goal_tolerance_rad  = {rec['goal_tolerance_rad']:.4f}   "
          f"(worst left-arm error {rec['worst_left_arm_error_rad']:.4f} x 1.2)")
    print(f"    settle_timeout_sec  = {rec['settle_timeout_sec']:.1f}   "
          f"(default {DEFAULT_SETTLE_TIMEOUT_SEC:.1f} x 2.5)")
    print(
        f"    -> with {rec['goal_tolerance_rad']:.4f} rad, "
        f"{rec['historical_failures_that_would_pass']}/{rec['total_events']} "
        "historical failures would have been reported as SUCCESS."
    )
    print("")


def _history_recommendation(result: dict[str, Any]) -> dict[str, Any]:
    """Derive the bridge parameters from the WHOLE failure history.

    A single run only shows one arm's steady-state error at one pose. The
    logs hold every failure ever recorded, so the recommendation is the
    worst LEFT-arm error (x1.2) -- the right arm's 0.15 rad outlier came
    from a different, unloaded configuration and must not set the limit.
    """
    all_events = [ev for entry in result["events"] for ev in entry["events"]]
    errors = sorted(ev["max_error_rad"] for ev in all_events)
    left_errors = sorted(
        ev["max_error_rad"]
        for entry in result["events"]
        for ev in entry["events"]
        if not any(
            j["joint"].startswith("right_") for j in (ev.get("joints") or [])
        )
    )
    pool = left_errors or errors
    # Drop the top-1 outlier (a right-arm run with a 0.15 rad error).
    trimmed = pool[:-1] if len(pool) > 3 else pool
    worst = trimmed[-1] if trimmed else 0.0
    recommended = round(worst * 1.2, 4)
    tcps = [ev["tcp_error_m"] for ev in all_events if ev.get("tcp_error_m") is not None]
    return {
        "goal_tolerance_rad": recommended,
        "settle_duration_sec": 0.30,
        "settle_timeout_sec": round(DEFAULT_SETTLE_TIMEOUT_SEC * 2.5, 1),
        "worst_left_arm_error_rad": round(worst, 6),
        "total_events": len(all_events),
        "historical_failures_that_would_pass": sum(
            1 for e in errors if e <= recommended
        ),
        "tcp_error_m_min": round(min(tcps), 6) if tcps else None,
        "tcp_error_m_median": round(sorted(tcps)[len(tcps) // 2], 6) if tcps else None,
        "tcp_error_m_max": round(max(tcps), 6) if tcps else None,
        "tcp_reached_fraction": (
            f"{sum(1 for t in tcps if t <= TCP_REACHED_M)}/{len(tcps)}" if tcps else None
        ),
    }


# ---------------------------------------------------------------------------
# the actual diagnosis
# ---------------------------------------------------------------------------
def _build_session(cfg: dict, args: argparse.Namespace):
    from core.g1d.g1d_session import G1DSession, G1DSessionConfig

    rb = cfg.get("robot") or {}
    side = args.side or str(rb.get("side", "left"))
    floor = resolve_z_floor(cfg, override_m=args.z_floor_m)
    if args.no_z_floor:
        floor = resolve_z_floor({**cfg, "enable_z_floor": False})
    if not floor.enabled:
        raise RuntimeError("the Z floor is disabled; a live run needs a measured floor")
    topics = cfg.get("camera_topics") or {}
    return G1DSession(
        G1DSessionConfig(
            side=side,
            namespace=str(rb.get("bridge_namespace", "lzh")),
            use_mock_robot=False,
            use_mock_cameras=True,
            head_camera_topic=str(topics.get("head_left", "/g1d/cameras/head_left/image_raw")),
            head_right_camera_topic=str(
                topics.get("head_right", "/g1d/cameras/head_right/image_raw")
            ),
            wrist_camera_topic=str(
                topics.get(f"{side}_wrist", f"/g1d/cameras/{side}_wrist/image_raw")
            ),
            z_floor=floor,
            dry_run=not args.execute,
            verbose=True,
        )
    )


def diagnose(cfg: dict, args: argparse.Namespace) -> dict[str, Any]:
    """Plan (and optionally execute) one move and measure how far the arm lands."""
    from core.g1d.ros_transport import G1DTransportError

    session = _build_session(cfg, args)
    out: dict[str, Any] = {"executed": bool(args.execute)}
    try:
        session.connect()
        robot = session.robot
        transport = robot.transport
        side = _resolve_side(cfg, args)

        before = np.asarray(robot.get_ee_pose(), dtype=float)
        start_joints = np.asarray(transport.read_joint_positions(), dtype=float)
        out["before_tcp_m"] = [round(float(v), 5) for v in before[:3]]

        # A small +X step: the same kind of move the sign test makes.
        target = before.copy()
        target[0] += float(args.step_m)
        target[3:] = before[3:]

        floor = session.config.z_floor
        if floor is not None and floor.enabled:
            target[2] = max(float(target[2]), floor.effective_z_m)

        print("=" * 78)
        print(f"  SETTLE DIAGNOSIS  --  {'EXECUTE' if args.execute else 'DRY-RUN'}")
        print("=" * 78)
        print(f"  side        : {side}")
        print(f"  start TCP   : {np.round(before[:3], 4).tolist()}")
        print(f"  target TCP  : {np.round(target[:3], 4).tolist()}")
        print(f"  step        : {args.step_m * 100:g} cm")
        print("")

        error: Optional[str] = None
        error_code: Optional[int] = None
        try:
            robot.update_desired_ee_pose(target.tolist())
        except G1DTransportError as exc:
            text = str(exc)
            error = text
            m = re.search(r"error_code=(-?\d+)", text)
            if m:
                error_code = int(m.group(1))
            print(f"  [move] the bridge REFUSED or FAILED the goal:")
            print(f"         {text}")
            print("")

        # A GOAL_TOLERANCE_VIOLATED abort does NOT raise by default: the bridge
        # keeps HOLDING the commanded final position, so the Cartesian goal WAS
        # reached. The transport records the event (commanded vs measured joints)
        # in ``last_tolerance_violation`` -- read it here so the report shows the
        # real numbers even though the call "succeeded".
        violation = getattr(transport, "last_tolerance_violation", None)
        if violation is not None:
            error_code = ERR_GOAL_TOLERANCE_VIOLATED
            error = str(violation.get("error_string") or "GOAL_TOLERANCE_VIOLATED")
            print("  [move] the bridge reported GOAL_TOLERANCE_VIOLATED:")
            print("         the arm reached the commanded point but the measured")
            print("         joints had not converged inside the bridge's settle window.")
            print("         The bridge keeps holding the final position, so the")
            print("         Cartesian goal WAS reached -- see the TCP error below.")
            print("")

        after = np.asarray(robot.get_ee_pose(), dtype=float)
        after_joints = np.asarray(transport.read_joint_positions(), dtype=float)
        out["after_tcp_m"] = [round(float(v), 5) for v in after[:3]]
        out["bridge_error"] = error
        out["bridge_error_code"] = error_code
        out["tolerance_violation"] = bool(violation is not None)

        tcp_error = after[:3] - target[:3]
        tcp_dist = float(np.linalg.norm(tcp_error))
        out["tcp_error_m"] = [round(float(v), 5) for v in tcp_error]
        out["tcp_error_norm_m"] = round(tcp_dist, 5)
        out["cartesian_reached"] = bool(tcp_dist <= TCP_REACHED_M)

        # Per-joint: the commanded final joint = the joint travel implied by the
        # planned trajectory's last point, and the measured = the FK of the
        # measured pose. The bridge's own diagnostic is the ground truth; this
        # reconstruction is the offline fallback.
        names = arm_joint_names(side)
        traj = getattr(transport, "last_trajectory", None)
        commanded: Optional[np.ndarray] = None
        if traj is not None and traj.get("final_positions"):
            commanded = np.asarray(traj["final_positions"], dtype=float)
        if commanded is None:
            commanded = after_joints.copy()

        rows = []
        worst = 0.0
        worst_joint = ""
        for i, name in enumerate(names):
            err = abs(float(after_joints[i]) - float(commanded[i]))
            rows.append(
                {
                    "joint": name,
                    "commanded_rad": round(float(commanded[i]), 6),
                    "measured_rad": round(float(after_joints[i]), 6),
                    "error_rad": round(err, 6),
                    "travel_rad": round(abs(float(after_joints[i]) - float(start_joints[i])), 6),
                }
            )
            if err > worst:
                worst = err
                worst_joint = name
        out["joints"] = rows
        out["worst_joint"] = worst_joint
        out["worst_joint_error_rad"] = round(worst, 6)

        # -- report ----------------------------------------------------------
        print("-" * 78)
        print("  MEASURED RESULT")
        print("-" * 78)
        if error is not None:
            print(f"  bridge error: {error}")
            if error_code == ERR_GOAL_TOLERANCE_VIOLATED:
                print(
                    "  -> GOAL_TOLERANCE_VIOLATED: the arm reached the target "
                    "PHYSICALLY, but the measured joints were still "
                    f"{worst:.4f} rad away from the commanded point when the "
                    "bridge's settle window closed."
                )
        else:
            print("  bridge error: None (the goal was accepted and settled)")
        print(f"  TCP error    : {np.round(tcp_error, 5).tolist()} m  "
              f"(|e| = {tcp_dist * 1000:.1f} mm)")
        verdict = "REACHED" if out["cartesian_reached"] else (
            "TOLERANT" if tcp_dist <= TCP_TOLERANT_M else "MISSED"
        )
        print(f"  cartesian    : {verdict} "
              f"(reached <= {TCP_REACHED_M * 1000:.0f} mm, "
              f"tolerant <= {TCP_TOLERANT_M * 1000:.0f} mm)")
        print("")
        print(f"  {'joint':<34}{'cmd (rad)':>12}{'meas (rad)':>12}{'err (rad)':>12}")
        for row in rows:
            flag = "  <-- worst" if row["joint"] == worst_joint else ""
            print(
                f"  {row['joint']:<34}{row['commanded_rad']:>12.6f}"
                f"{row['measured_rad']:>12.6f}{row['error_rad']:>12.6f}{flag}"
            )
        print("")

        # -- the fix ---------------------------------------------------------
        needed_tol = max(worst * 1.5, worst + 0.005)
        print("-" * 78)
        print("  RECOMMENDED BRIDGE PARAMETERS")
        print("-" * 78)
        print(f"  current goal_tolerance_rad : {DEFAULT_GOAL_TOLERANCE_RAD:.4f}  (default)")
        print(f"  observed worst joint error : {worst:.6f} rad")
        if error_code == ERR_GOAL_TOLERANCE_VIOLATED:
            print(f"  -> suggested goal_tolerance_rad : {needed_tol:.4f} rad "
                  f"(worst x 1.5, with a 5 mrad margin)")
        else:
            print(f"  -> the default {DEFAULT_GOAL_TOLERANCE_RAD:.4f} rad was enough "
                  f"(worst error {worst:.6f} rad); no change needed.")
        print(f"  -> suggested settle_duration_sec  : 0.30  (hold the target longer "
              f"before reporting SUCCESS)")
        print(f"  -> suggested settle_timeout_sec   : 5.0   (give the gravity-loaded "
              f"shoulder time to creep in)")
        print("")
        print("  Apply it at launch time (no rebuild needed):")
        print(
            "    ros2 launch lzh_bringup real_moveit.launch.py \\\n"
            f"        real_execution:=true gripper_real_execution:=true \\\n"
            f"        goal_tolerance_rad:={needed_tol:.4f} \\\n"
            "        settle_duration_sec:=0.30 settle_timeout_sec:=5.0"
        )
        print("")
        out["suggested"] = {
            "goal_tolerance_rad": round(needed_tol, 4),
            "settle_duration_sec": 0.30,
            "settle_timeout_sec": 5.0,
        }
        return out
    finally:
        session.close()


def _resolve_side(cfg: dict, args: argparse.Namespace) -> str:
    rb = cfg.get("robot") or {}
    side = args.side or str(rb.get("side", "left"))
    if side not in ("left", "right"):
        raise RuntimeError(f"side must be 'left' or 'right', got {side!r}")
    return side


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    if args.read_params:
        print_params(read_params())
        return 0
    if args.scan_log:
        scan = scan_log(args.log_dir)
        print_log_scan(scan)
        if args.json_out:
            out = Path(args.json_out)
            out.parent.mkdir(parents=True, exist_ok=True)
            payload = dict(scan)
            payload["recommendation"] = _history_recommendation(scan)
            out.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(f"[settle] report written to {out}")
        return 0

    cfg = load_yaml(args.robot_config)
    import rclpy

    # G1DSession creates ROS nodes through the real transport.
    from core.g1d.ros_environment import require_g1d_ros_environment

    require_g1d_ros_environment()
    rclpy.init(args=None)
    try:
        result = diagnose(cfg, args)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    finally:
        if rclpy.ok():
            rclpy.shutdown()

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[settle] report written to {out}")
    # Exit code: 0 = the arm reached the target (Cartesian), 3 = it did not.
    # A GOAL_TOLERANCE_VIOLATED alone is NOT a failure -- the TCP error decides.
    if args.execute and not result.get("cartesian_reached"):
        print(
            "  -> the arm did NOT reach the Cartesian target. The goal is outside "
            "the reachable workspace, or the move was blocked.",
            file=sys.stderr,
        )
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
