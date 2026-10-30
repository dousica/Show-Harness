#!/usr/bin/env bash
set -Eeuo pipefail

HARNESS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# ROS 2 workspace, vendored as a git submodule. Override with
# ROS_WS=/path/to/workspace to use a checkout built elsewhere.
ROS_WS="${ROS_WS:-$HARNESS_DIR/third_party/g1d_ros2_lzh}"
CAMERA_IP="192.168.123.164"
NETWORK_INTERFACE="eno2"
STARTUP_ATTEMPTS=8
LOG_DIR="$HOME/.ros/log/harness_g1d_start_$(date +%Y%m%d_%H%M%S)"

declare -a launch_pids=()
tail_pid=""

cleanup() {
  trap - EXIT
  if [[ -n "$tail_pid" ]]; then
    kill "$tail_pid" 2>/dev/null || true
    wait "$tail_pid" 2>/dev/null || true
  fi
  for pid in "${launch_pids[@]}"; do
    kill -0 "$pid" 2>/dev/null || continue
    kill -INT "$pid" 2>/dev/null || true
  done
  for pid in "${launch_pids[@]}"; do
    wait "$pid" 2>/dev/null || true
  done
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

fail() {
  printf '[start] ERROR: %s\n' "$*" >&2
  exit 1
}

command -v sudo >/dev/null || fail "sudo is required to activate the g1d network profile"
command -v nmcli >/dev/null || fail "nmcli is not installed"
command -v ip >/dev/null || fail "iproute2 is not installed"
[[ -f "$ROS_WS/setup_env.sh" ]] || fail "ROS workspace not found: $ROS_WS"
[[ -f "$HARNESS_DIR/scripts/check_setup.py" ]] || fail "Harness checkout is incomplete: $HARNESS_DIR"

mkdir -p "$LOG_DIR"

printf '[start] Activating NetworkManager profile g1d...\n'
sudo nmcli connection up g1d || fail "could not activate NetworkManager profile g1d"

printf '[start] eno2 IPv4 configuration:\n'
ip -4 addr show dev "$NETWORK_INTERFACE" || fail "network interface $NETWORK_INTERFACE not found"
ip -4 -o addr show dev "$NETWORK_INTERFACE" | grep -q ' inet ' \
  || fail "$NETWORK_INTERFACE has no IPv4 address after activating g1d"

printf '[start] Route to camera host:\n'
route="$(ip -4 route get "$CAMERA_IP")" || fail "no IPv4 route to $CAMERA_IP"
printf '  %s\n' "$route"
[[ " $route " == *" dev $NETWORK_INTERFACE "* ]] \
  || fail "route to $CAMERA_IP does not use $NETWORK_INTERFACE"

export ROS_DOMAIN_ID=42
set +u
if source "$ROS_WS/setup_env.sh"; then
  set -u
else
  set -u
  fail "failed to load the LZH ROS environment"
fi
cd "$HARNESS_DIR"

printf '[start] Launch logs: %s\n' "$LOG_DIR"
(
  exec ros2 launch lzh_bringup real_moveit.launch.py real_execution:=true
) >"$LOG_DIR/real_moveit.log" 2>&1 &
launch_pids+=("$!")

(
  exec ros2 launch lzh_camera_bridge cameras.launch.py show_viewer:=false
) >"$LOG_DIR/cameras.log" 2>&1 &
launch_pids+=("$!")

ready=0
for ((attempt = 1; attempt <= STARTUP_ATTEMPTS; attempt++)); do
  for pid in "${launch_pids[@]}"; do
    kill -0 "$pid" 2>/dev/null || {
      printf '[start] A ROS launch process exited. Recent launch output:\n' >&2
      tail -n 60 "$LOG_DIR/real_moveit.log" "$LOG_DIR/cameras.log" >&2 || true
      fail "ROS stack stopped during startup"
    }
  done

  printf '[start] Waiting for ROS services and joint feedback (%d/%d)...\n' \
    "$attempt" "$STARTUP_ATTEMPTS"
  if preflight_output="$(python3 scripts/check_setup.py \
    --robot-config configs/robot_g1d.yaml --no-vlm 2>&1)"; then
    printf '%s\n' "$preflight_output"
    ready=1
    break
  fi
  printf '%s\n' "$preflight_output" | tail -n 8
  sleep 2
done

if [[ "$ready" -ne 1 ]]; then
  printf '[start] ROS startup did not pass preflight. Recent launch output:\n' >&2
  tail -n 60 "$LOG_DIR/real_moveit.log" "$LOG_DIR/cameras.log" >&2 || true
  fail "startup timed out; see logs in $LOG_DIR"
fi

printf '[start] Running full Harness preflight, including the configured VLM...\n'
python3 scripts/check_setup.py --robot-config configs/robot_g1d.yaml \
  || fail "Harness preflight failed; see logs in $LOG_DIR"

printf '[start] G1-D ROS stack is ready. No vision recognition or motion task was started.\n'
printf '[start] Press Ctrl+C to stop both ROS launches.\n'
tail --lines=0 --follow=name "$LOG_DIR/real_moveit.log" "$LOG_DIR/cameras.log" &
tail_pid="$!"

if wait -n "${launch_pids[@]}"; then
  fail "a ROS launch process exited unexpectedly"
else
  fail "a ROS launch process exited unexpectedly; see logs in $LOG_DIR"
fi
