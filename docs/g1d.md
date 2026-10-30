# Unitree G1-D Integration

This document describes the G1-D support in Show-Harness: its runtime paths,
configuration, ROS 2 interfaces, perception/calibration contracts, commissioning
checks, and current limitations. It is a software integration guide, not a claim
that a particular robot or calibration has passed physical acceptance.

## Status

| Capability | In this repository | Commissioning still required |
| --- | --- | --- |
| Single-arm action controller and ROS transport | Implemented for left or right arm | Confirm bridge endpoints and each command on the target rig |
| MoveIt planning or local numerical IK | Implemented; `motion_path=auto` prefers MoveIt and falls back to local IK | Review collision scene and validate workspace on hardware |
| Trajectory structure, joint-limit, speed, start-state, and Z-floor guards | Implemented in the harness; trajectory checks mirror the bridge rules | Confirm the deployed bridge uses the same guard values and code |
| RGB VLM detection | Implemented by `vision_detect.py` and `inspect_target.py` | Check image freshness, framing, latency, and detection quality |
| Stereo depth and hand-eye transform | Loader and processing path implemented | Validate synchronization, metric depth, and 3D error physically |
| Gumi single-arm collection | Implemented by `collect_rollouts_web.py --robot g1d` | Validate signs, gripper feedback, action gates, and recording on hardware |
| Autonomous perception-to-grasp | Not implemented as a closed-loop target-pose/grasp planner | Target pose, grasp pose, collision-aware approach, grasp verification, and recovery are outstanding |
| Bridge-side one-arm-only policy | Not provided by the harness client | The bridge must reject commands for the unselected arm if hard isolation is required |

The repository contains both dry-run and real-execution code paths. A successful
offline test, MoveIt plan, or VLM bounding box is not evidence that a physical
grasp is safe or accurate.

## Architecture

```text
run_real.py / Gumi / G1-D diagnostic scripts
                    |
          G1DAtomicController
          semantic token -> Cartesian target in torso_link
                    |
              G1DSession
      measured state + RGB observations
                    |
              G1DRosTransport
    FK / IK or MoveIt plan / local trajectory guards
                    |
       ROS 2 FollowJointTrajectory + GripperCommand
                    |
       lzh_sdk_bridge (LowCmd owner)
```

The harness does not publish Unitree `LowCmd`. It reads `/joint_states`, computes
the TCP pose from measured arm joints, and submits ROS 2 goals to the bridge. The
bridge is the motor-command writer. Do not run another LowCmd writer at the same
time.

The G1-D action interpreter uses base-frame increments in `torso_link`. The
repository's default primitive signs are **not hardware-certified**; see
`configs/primitives_g1d.yaml`. One token is planned and executed as one trajectory,
not as a stream of Cartesian servo setpoints.

### Main modules

| Path | Responsibility |
| --- | --- |
| `core/g1d/single_arm.py` | Selected-side joint list and ROS endpoint names |
| `core/g1d/g1d_interface.py` | Transport-independent robot interface and offline mock |
| `core/g1d/g1d_session.py` | Owns robot transport and selected-side camera subscriptions |
| `core/g1d/ros_transport.py` | ROS 2 feedback, IK/MoveIt path selection, action clients, and guards |
| `core/g1d/bridge_guard.py` | Python mirror of bridge trajectory admission checks |
| `core/g1d/safety.py` | TCP Z-floor clamp and sampled trajectory floor check |
| `core/g1d/kinematics.py` | Single-arm FK, Dex1 fingertip-midpoint TCP, and wrist/TCP conversion |
| `core/g1d/inverse_kinematics.py` | Local numerical IK fallback |
| `core/g1d/camera_source.py` | Fresh RGB subscription and BGR-to-RGB conversion |
| `core/g1d/vision_api.py` | RGB image to validated VLM detection JSON |
| `core/g1d/vision_geometry.py` | Stereo rectification, depth, bbox statistics, and projection |
| `core/g1d/calibration.py` | Stereo and eye-to-hand loading, transform composition, and pairing check |
| `interpreters/g1d_atomic_controller.py` | G1-D atomic tokens, one-trajectory-per-token execution |

## ROS 2 Contract

The endpoint profile is selected by `robot.side`. `robot.bridge_namespace` must
be `lzh`; the Harness does not support or launch another G1-D bridge.

| Setting | Arm action | Dex1 action | Raw Dex1 feedback |
| --- | --- | --- | --- |
| `lzh` | `/<side>_arm_controller/follow_joint_trajectory` | `/lzh_<side>_dex1/gripper_command` | `/lzh_dex1/motor_states` |

Both profiles read `/joint_states`. Planning uses `/plan_kinematic_path` and
`/execute_trajectory` when available. In `motion_path=auto`, the autonomous
transport falls back to the repository's local IK when the MoveIt planning
service is unavailable. The standalone `pose_goal.py` tool is a separate direct
MoveIt client and does not use that fallback.

Every G1-D ROS process must run with `ROS_DOMAIN_ID=42`. The Harness preflight
and live transport reject an unset/wrong domain, a senior-workspace overlay, or
the absence of this workspace's install overlay. In a clean terminal:

```bash
export ROS_DOMAIN_ID=42
source /home/yushu/g1d_ros2_lzh/setup_env.sh
```

The Harness does not start ROS nodes. Start the stack in separate terminals,
using the same environment in each:

```bash
# Terminal 1: LZH SDK bridge, robot description, MoveIt and RViz
export ROS_DOMAIN_ID=42
source /home/yushu/g1d_ros2_lzh/setup_env.sh
ros2 launch lzh_bringup real_moveit.launch.py

# Terminal 2: TeleImager camera publishers
export ROS_DOMAIN_ID=42
source /home/yushu/g1d_ros2_lzh/setup_env.sh
ros2 launch lzh_camera_bridge cameras.launch.py show_viewer:=false

# Terminal 3: read-only G1-D preflight
export ROS_DOMAIN_ID=42
source /home/yushu/g1d_ros2_lzh/setup_env.sh
python3 scripts/check_setup.py --robot-config configs/robot_g1d.yaml --no-vlm
```

`real_moveit.launch.py` leaves arm/gripper real execution disabled by default.
The bridge, MoveIt, camera nodes, and Harness must all remain in Domain 42. The
SDK2 transport inside `lzh_sdk_bridge` is separate from ROS 2 discovery.

## Configuration

The robot config requires the site layer. On a new checkout:

```bash
cd /home/yushu/harness_lzh
cp configs/site/g1d.yaml.example configs/site/g1d.yaml
```

Edit the ignored `configs/site/g1d.yaml` for the actual machine. It contains the
selected arm, bridge profile, measured calibration file paths, and frame names.
The file is ignored by Git so local paths do not become part of a commit. The
tracked example contains path conventions only; it is not a calibration and
does not prove those paths exist on another machine.

Configuration precedence matters: `defaults` merge **under**
`configs/robot_g1d.yaml`; values in the main file override values from the site
layer. Consequently, keep rig identity and calibration paths in `site/g1d.yaml`,
and keep general processing/safety defaults in `robot_g1d.yaml`.

Before a live G1-D Harness run, `enable_z_floor` must be true and the floor must
be measured. A value of `0.0` is accepted only when
`z_floor_calibrated: true` is explicitly set. The effective TCP lower bound is
`z_floor_m + z_floor_margin_m`. Re-capture on the actual work surface instead of
copying the example's value:

```bash
export ROS_DOMAIN_ID=42
source /home/yushu/g1d_ros2_lzh/setup_env.sh
python3 scripts/g1d/capture_z_floor.py --side left --capture
```

The capture command is read-only: the gripper must already be resting on the
table, and the script computes TCP height from measured joint state. Record the
result in the active robot/site configuration as appropriate.

VLM credentials belong in `configs/secrets.env` or the ignored
`configs/secrets.local.env`, or in environment variables. Do not put API keys in
robot YAML or calibration files.

## Calibration and Perception

### Stereo calibration

`StereoCalibration` expects a YAML mapping with image dimensions, both camera
intrinsics/distortion coefficients, and the left-to-right stereo transform:

```yaml
image_width: 640
image_height: 480
K_left: [[...], [...], [...]]
D_left: [...]
K_right: [[...], [...], [...]]
D_right: [...]
R_left_to_right: [[...], [...], [...]]
T_left_to_right_m: [[...], [...], [...]]
```

The convention is `X_right = R_left_to_right @ X_left + T_left_to_right_m`;
translation is in meters. The current SGBM implementation requires a mostly
horizontal baseline. Input images must match the calibrated dimensions.

### Eye-to-hand transform

The hand-eye result provides a homogeneous `T_torso_camera` (or
`T_base_camera`) mapping raw left-camera coordinates into the robot base frame:

```text
X_torso = R_torso_camera * X_camera + t_torso_camera
```

The configured hand-eye path can be a result file or a directory containing
`handeye_result_*.yaml`; when it is a directory, the newest file by modification
time is selected. When the hand-eye result contains `intrinsics_sha256`, the
loader rejects a stereo file from a different calibration session. Older
results without a fingerprint cannot receive that pairing check.

Depth is computed on the rectified left image. The hand-eye transform is for the
raw optical frame, so `rectified_base_transform()` composes the rectification
rotation before a point is reported in `torso_link`.

The calibration loader validates file structure, units, rotation matrices,
baseline constraints, and (where available) file pairing. It does **not** prove
that the calibration measurements are physically accurate. Verify depth against
known distances and verify the base-frame point against measured targets before
using it for robot planning. Camera bridge timestamps are receipt timestamps,
not guaranteed exposure timestamps; the software's pair-skew check cannot prove
hardware synchronization.

### Perception commands

RGB bounding box only, with a still image (no ROS camera subscription):

```bash
python3 scripts/g1d/vision_detect.py \
  --image /path/to/frame.png \
  --target "steel bearing" \
  --config configs/robot_g1d.yaml
```

Read a fresh camera image and call the configured VLM:

```bash
export ROS_DOMAIN_ID=42
source /home/yushu/g1d_ros2_lzh/setup_env.sh
python3 scripts/g1d/vision_detect.py \
  --topic /g1d/cameras/head_left/image_raw \
  --target "steel bearing" \
  --config configs/robot_g1d.yaml \
  --save-frames /tmp/g1d_frames
```

The result contains a label, confidence, and normalized full-image `bbox_norm`.
The script does not compute depth, read TF, or command the robot.

Run stereo depth and output the target point using the configured hand-eye file:

```bash
python3 scripts/g1d/inspect_target.py \
  --config configs/robot_g1d.yaml \
  --topic /g1d/cameras/head_left/image_raw \
  --right-topic /g1d/cameras/head_right/image_raw \
  --target "steel bearing" \
  --count 1
```

With calibration configured, the tool checks image freshness, left/right stamp
skew, CameraInfo dimensions/intrinsics, estimates robust bbox depth, and reports
`point_base_m`. `--use-live-tf` explicitly selects a time-aligned TF lookup
instead of the hand-eye file. Neither perception script sends a robot command.

The point is a 3D detection estimate, **not a grasp pose**. The repository does
not yet convert it into an approach direction, gripper orientation, collision-
checked pre-grasp/grasp sequence, or a verified pick-and-place program.

## Motion and Data Collection

### Autonomous Harness

The G1-D branch is selected by `hardware: g1d` and `primitives_config` in
`configs/robot_g1d.yaml`. Inspect the resolved configuration and ROS endpoints
before running. Harness live execution defaults to dry-run unless `dry_run: false`
is deliberately set in the resolved config; the bridge has independent execution
gates as well. A bridge dry-run may accept/plan a goal without writing motor
commands, so a successful action response alone does not prove physical motion.

The generic autonomous entry point is:

```bash
python3 scripts/check_setup.py \
  --robot-config configs/robot_g1d.yaml \
  --no-vlm --no-hardware
python3 scripts/run_real.py \
  --robot-config configs/robot_g1d.yaml
```

The first command checks local configuration/dependencies without network or
hardware probes. The second command uses the configured VLM and G1-D transport;
leave `dry_run` enabled until the robot, bridge, signs, floor, and stop procedure
have been commissioned. A real rollout requires a valid Z floor, fresh joint
state, one active LowCmd writer, and a verified VLM endpoint.

### Gumi web teleoperation and recording

Gumi's G1-D backend connects the same session and atomic controller, captures
head/wrist observations, and records rollout data. Unlike the Harness run path,
the Gumi builder currently constructs its robot session with `dry_run=False`;
it is a real-control surface and must be treated as such. Start it only after the
bridge, its hardware execution gates, Z floor, and on-site supervision have
checked:

```bash
python3 gumi/collect_rollouts_web.py data/rollouts_g1d \
  --robot g1d \
  --robot-config configs/robot_g1d.yaml \
  --primitives configs/primitives_g1d.yaml \
  --port 8600
```

Open `http://localhost:8600`. The browser serves single-arm teleoperation and
rollout recording. The selected wrist-camera topic falls back to the other head
eye if unavailable; check the stream labels and actual views instead of assuming
the fallback is a wrist camera.

Replay a previously captured VLM token trace through the Gumi UI:

```bash
python3 scripts/g1d/replay_action_sequence.py \
  --teleop-url http://localhost:8600 \
  --trace data/gpt_operator_traces/<session>/events.jsonl \
  --dry-run
```

Remove `--dry-run` only when the sequence has been reviewed and the live Gumi
session is intentionally controlling the selected arm. `--start-recording` and
`--stop-recording` request the corresponding Gumi recording endpoints.

### Direct tools

`pose_goal.py` is a standalone MoveIt planner/executor. Planning is the default;
`--execute` asks for an interactive `YES`. It caps requested displacement and
validates the selected arm's trajectory start, names, timing, and speeds. It
does **not** pass through `G1DRosTransport` and does not enforce the Harness
Z-floor guard. Use it plan-only for inspection; do not use it for live movement
until a separate Z-floor/collision check has been made.

`gripper_goal.py` selects a side, validates nominal opening in `[0, 0.09]` m,
and sends nothing unless `--send` and an interactive `YES` are provided. The
bridge's motor-angle-to-opening mapping and the interpreter's grasp thresholds
are provisional, not a calibrated jaw-width or force measurement.

Useful offline/diagnostic entry points:

| Command | Behavior |
| --- | --- |
| `scripts/g1d/direction_sign_test.py` | Default is plan-only. `--execute` moves in 2 cm token increments and measures signs; this is a hardware commissioning test. |
| `scripts/g1d/settle_diagnose.py --read-params` | Reads live bridge parameters without motion. |
| `scripts/g1d/settle_diagnose.py --scan-log` | Computes historical joint/TCP settle diagnostics from local ROS logs. |
| `scripts/g1d/settle_diagnose.py --execute` | Performs a small real move; explicit opt-in and supervision required. |
| `scripts/g1d/capture_z_floor.py --side left --capture` | Reads measured joints and reports TCP height; does not move the robot. |

## Safety Boundaries

1. `SingleArmProfile` constrains the harness client to one side. It is not a
   server-side whitelist. The SDK bridge may still expose both left and right
   single-arm Action servers; the bridge's `enable_both_arms_real_execution`
   parameter protects its 14-joint coordinated action, not the opposite
   single-arm Action. Do not claim one-arm isolation unless the deployed bridge
   explicitly enforces it.
2. The Z floor clamps Cartesian targets and samples planned joint trajectories
   through FK. It only protects the TCP height relative to the floor. It is not
   a full environment collision model and does not protect against shelves,
   self-collision, people, or an incorrect floor measurement.
3. `GOAL_TOLERANCE_VIOLATED` (`-5`) is special-cased by the current transport.
   It records joint/TCP diagnostics and, by default, returns without raising.
   **The measured TCP error is not currently used as an acceptance threshold.**
   Therefore downstream controller code may treat a `-5` action as completed
   even when the TCP error is unknown or large. Keep autonomous live rollouts
   disabled until this policy is corrected or independently bounded and
   validated. Setting `tolerance_violation_is_fatal: true` restores fatal
   behavior.
4. A timeout or action failure leaves the physical arm state uncertain. Inspect
   the robot before issuing another goal; do not auto-replay after an uncertain
   result.
5. Confirm the emergency-stop path, clear workspace, sole LowCmd writer, network
   interface, bridge flags, selected side, and a human stop operator before any
   execution test.

## Verification

Offline tests do not need ROS, API credentials, or a robot (except the current
`test_g1d_calibration.py`, which reads measured files from this lab workstation):

```bash
python3 -m pytest -q \
  tests/test_g1d_interface.py \
  tests/test_g1d_ik.py \
  tests/test_g1d_bridge_guard.py \
  tests/test_g1d_safety.py \
  tests/test_g1d_vision_geometry.py \
  tests/test_g1d_vision_api.py \
  tests/test_g1d_calibration.py
```

ROS smoke tests, bridge endpoint checks, direction signs, camera/depth accuracy,
gripper calibration, and complete G1-D grasp acceptance require the deployed ROS
workspaces and physical rig. Record their versions and measurements separately
from offline test results.
