#!/usr/bin/env bash
set -Eeuo pipefail

# One-shot setup for a fresh checkout. Safe to re-run.
#
#   bash scripts/bootstrap.sh              # submodules + site configs + .venv
#   bash scripts/bootstrap.sh --no-venv    # skip building the Python environment
#   bash scripts/bootstrap.sh --ros        # also colcon-build the vended ROS workspace
#
# The ROS 2 system install (/opt/ros/humble) is a machine prerequisite and is
# never vendored; see docs/DEPENDENCIES.md.

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ROS_WS="${ROS_WS:-$REPO_ROOT/third_party/g1d_ros2_lzh}"
BUILD_ROS=0
SETUP_VENV=1

for arg in "$@"; do
  case "$arg" in
    --ros) BUILD_ROS=1 ;;
    --no-venv) SETUP_VENV=0 ;;
    -h|--help) sed -n '4,10p' "$0" | sed 's/^# \?//'; exit 0 ;;
    *) printf 'unknown argument: %s\n' "$arg" >&2; exit 2 ;;
  esac
done

cd "$REPO_ROOT"
say() { printf '[bootstrap] %s\n' "$*"; }

# 1. The G1-D ROS 2 workspace is vendored as a git submodule, so the launch
#    files start.bash calls (lzh_bringup, lzh_camera_bridge) ship with the repo.
if [[ -d "$REPO_ROOT/.git" ]]; then
  say "initialising submodules"
  git submodule update --init --recursive
fi

if [[ -f "$ROS_WS/setup_env.sh" ]]; then
  say "ROS workspace: $ROS_WS"
else
  say "WARNING: no ROS workspace at $ROS_WS"
  say "         use ROS_WS=/path/to/colcon/workspace to point at another build"
fi

# 2. Site layers are per-machine and git-ignored; create them from the tracked
#    examples. The examples already point at configs/calibration/.
for site in g1d piper_arms; do
  src="configs/site/${site}.yaml.example"
  dst="configs/site/${site}.yaml"
  [[ -f "$src" ]] || continue
  if [[ -f "$dst" ]]; then
    say "keeping existing $dst"
  else
    cp "$src" "$dst"
    say "created $dst from the example"
  fi
done

# 3. The harness Python environment (.venv). Sourcing a served VLM or a robot
#    driver does not need this; running the harness itself does.
if [[ "$SETUP_VENV" -eq 1 ]]; then
  say "building the base Python environment"
  bash scripts/setup.sh base
fi

# 4. Optional: build the vendored ROS 2 workspace. Requires ROS Humble sourced
#    in the calling shell.
if [[ "$BUILD_ROS" -eq 1 ]]; then
  [[ -f "$ROS_WS/setup_env.sh" ]] || {
    printf 'cannot build: %s is missing\n' "$ROS_WS" >&2; exit 1; }
  if [[ -f /opt/ros/humble/setup.bash ]]; then
    # shellcheck disable=SC1091
    set +u; source /opt/ros/humble/setup.bash; set -u
  else
    say "WARNING: /opt/ros/humble/setup.bash not found; building with the current environment"
  fi
  say "colcon build in $ROS_WS"
  ( cd "$ROS_WS" && colcon build --symlink-install )
fi

say "done"
