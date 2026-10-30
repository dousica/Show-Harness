# External dependencies

What a checkout needs beyond its own files, and how each piece is provided.
Run `bash scripts/bootstrap.sh` to satisfy everything marked **vendored** or
**generated**.

## Vendored in this repository

| What | Where | Notes |
|---|---|---|
| G1-D ROS 2 workspace | `third_party/g1d_ros2_lzh` (git submodule of `dousica/harness_g1d`) | Source of `setup_env.sh`, `lzh_bringup/real_moveit.launch.py` and `lzh_camera_bridge/cameras.launch.py`. Pin the submodule commit, not a branch. |
| Camera calibration | `configs/calibration/camera/` | `head_stereo.yaml` (stereo extrinsics), plus left/right/wrist intrinsics. |
| Hand-eye calibration | `configs/calibration/handeye/` | `handeye_result_*.yaml`; the loader picks the newest by mtime. Sample images from the capture session are **not** committed. |

`start.bash` builds `ROS_WS` as `${ROS_WS:-$HARNESS_DIR/third_party/g1d_ros2_lzh}`;
set `ROS_WS` to run against a workspace built elsewhere.

## Generated per machine

| What | How |
|---|---|
| `configs/site/g1d.yaml` | `bootstrap.sh` copies `configs/site/g1d.yaml.example`. Git-ignored: rig identity and any local path overrides live here. |
| `configs/site/piper_arms.yaml` | Same, from `configs/site/piper_arms.yaml.example` (Piper only). |
| `configs/secrets.local.env` | Copy `configs/secrets.env.example`. Never commit it. |
| `.venv` | `bash scripts/setup.sh base`. Python >= 3.10; the reference environment is 3.11. |

## Machine prerequisites (cannot be vendored)

| Requirement | Needed by | Notes |
|---|---|---|
| ROS 2 Humble (`/opt/ros/humble`) | G1-D | System install. `bootstrap.sh --ros` sources it before `colcon build`. |
| `uv` | `scripts/setup.sh` | Creates the venvs. |
| Camera network | G1-D runtime | `start.bash` activates the NetworkManager profile `g1d`, requires interface `eno2` with an IPv4 address, and a route to `192.168.123.164`. Recreate the profile per machine. |

## Other robots (optional)

| Requirement | Needed by | Notes |
|---|---|---|
| `$COBOT_MAGIC_DIR` (default `~/cobot_magic`) + `/opt/ros/noetic` | Piper | Separate Noetic workspace, not part of this repo. |
| `~/miniconda3` | Piper | `scripts/piper/run_arm.sh` sources its conda profile. |
| `~/docker/isaac-sim` | SO-101 workshop | Docker image cache used by `scripts/trajectory/real2sim/so101_workshop/run_in_docker.sh`. |

## Datasets and model weights (deliberately not in git)

| What | Default location | Notes |
|---|---|---|
| VLM base weights | `models/huggingface/hub/` | Fetched by `scripts/model/download_vlm_model.sh`. |
| Fine-tuned adapters | `models/Show-Harness-VLMs/` | |
| Rollouts, captures | `data/`, `rollouts/` | Reproducible artifacts; use DVC or release assets if they must be shared. |
