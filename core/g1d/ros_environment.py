"""Fail-fast checks for the G1-D ROS 2 runtime environment."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping

ROS_DOMAIN_ID = "42"
FORBIDDEN_OVERLAY_MARKERS = (
    "g1d_ros2_ws",
    "g1d_ros2_yqqh",
    "unitree_ros_official",
)
ROS_PATH_VARIABLES = (
    "AMENT_PREFIX_PATH",
    "CMAKE_PREFIX_PATH",
    "COLCON_PREFIX_PATH",
    "PYTHONPATH",
    "LD_LIBRARY_PATH",
)


def g1d_ros_environment_errors(
    env: Mapping[str, str] | None = None,
    *,
    require_lzh_overlay: bool = True,
) -> list[str]:
    """Return environment violations without importing ROS libraries."""
    values = os.environ if env is None else env
    errors: list[str] = []
    domain = values.get("ROS_DOMAIN_ID")
    if domain != ROS_DOMAIN_ID:
        errors.append(
            f"ROS_DOMAIN_ID must be {ROS_DOMAIN_ID}; got "
            f"{domain!r} (unset is not allowed)"
        )

    for variable in ROS_PATH_VARIABLES:
        for entry in filter(None, values.get(variable, "").split(os.pathsep)):
            marker = next(
                (item for item in FORBIDDEN_OVERLAY_MARKERS if item in entry), None
            )
            if marker:
                errors.append(f"{variable} contains forbidden overlay path: {entry}")

    if require_lzh_overlay:
        workspace = Path(
            values.get("LZH_ROS_WS", str(Path.home() / "g1d_ros2_lzh"))
        ).expanduser().resolve()
        prefixes = values.get("AMENT_PREFIX_PATH", "").split(os.pathsep)
        expected_install = (workspace / "install").resolve()
        has_lzh_overlay = False
        for entry in filter(None, prefixes):
            try:
                Path(entry).expanduser().resolve().relative_to(expected_install)
                has_lzh_overlay = True
                break
            except (OSError, ValueError):
                continue
        if not has_lzh_overlay:
            errors.append(
                f"AMENT_PREFIX_PATH does not contain {expected_install}; "
                "source /home/yushu/g1d_ros2_lzh/setup_env.sh"
            )
    return errors


def require_g1d_ros_environment(*, require_lzh_overlay: bool = True) -> None:
    """Raise before creating a G1-D ROS node if the runtime is unsafe."""
    errors = g1d_ros_environment_errors(require_lzh_overlay=require_lzh_overlay)
    if errors:
        raise RuntimeError("Invalid G1-D ROS environment: " + "; ".join(errors))
