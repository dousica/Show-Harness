"""Load the G1-D's measured camera calibration: stereo + eye-to-hand extrinsic.

Two files, one session
-----------------------
The G1-D's head carries a stereo pair. Two independent measurements describe it:

``head_stereo.yaml``
    The intrinsics of both eyes and the rigid transform between them, in the
    OpenCV convention ``X_right = R_left_to_right @ X_left + T_left_to_right_m``.
    The camera bridge keeps a copy under ``~/.ros/g1d_camera_calibration/``.

``handeye_result_*.yaml``
    The pose of the LEFT eye in the robot's ``torso_link`` frame, solved
    eye-to-hand against a ChArUco board stood on the workbench, in the
    convention ``X_torso = R_torso_camera @ X_camera + t_torso_camera_m``.

They are only valid TOGETHER: the extrinsic was solved with the intrinsics of
one specific stereo session, and the hand-eye result records that session's
SHA256. Pairing a hand-eye result with a re-calibrated stereo file yields a
plausible-looking but wrong 3D point, so :func:`verify_calibration_pairing`
refuses that combination instead of trusting it.

Frames
------
The extrinsic describes the RAW left optical frame
(``head_left_camera_optical_frame``). The depth map is computed on the
RECTIFIED left frame, which is the raw frame rotated by ``R1``. The two are
composed by :func:`rectified_base_transform`.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

from core.g1d.vision_geometry import StereoCalibration

__all__ = [
    "CalibrationBundle",
    "CalibrationError",
    "CameraExtrinsic",
    "load_calibration",
    "load_handeye_yaml",
    "load_stereo_yaml",
    "rectified_base_transform",
    "resolve_handeye_path",
    "verify_calibration_pairing",
]


class CalibrationError(RuntimeError):
    """Raised when a calibration file is missing, malformed or mismatched."""


#: The frame the G1-D's arm is planned against; the head cameras are
#: calibrated into it.
DEFAULT_BASE_FRAME = "torso_link"

#: The camera frame the eye-to-hand result is expressed in.
DEFAULT_CAMERA_FRAME = "head_left_camera_optical_frame"


def _expand(path: str) -> Path:
    return Path(str(path)).expanduser()


def _matrix4(value: Any, name: str) -> np.ndarray:
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise CalibrationError(f"{name} must be a numeric 4x4 matrix") from exc
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise CalibrationError(f"{name} must be a finite 4x4 matrix")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise CalibrationError(f"{name} must have a homogeneous last row [0, 0, 0, 1]")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3):
        raise CalibrationError(f"{name} rotation must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-3):
        raise CalibrationError(f"{name} rotation determinant must be +1")
    return matrix


@dataclass(frozen=True)
class CameraExtrinsic:
    """The camera's pose in the robot base frame (the eye-to-hand result)."""

    T_base_camera: np.ndarray
    base_frame: str
    camera_frame: str
    source: str = ""
    intrinsics_sha256: str = ""

    def __post_init__(self) -> None:
        matrix = _matrix4(self.T_base_camera, "T_base_camera")
        object.__setattr__(self, "T_base_camera", matrix)
        if not str(self.base_frame).strip():
            raise CalibrationError("base_frame must not be empty")
        if not str(self.camera_frame).strip():
            raise CalibrationError("camera_frame must not be empty")
        object.__setattr__(self, "base_frame", str(self.base_frame))
        object.__setattr__(self, "camera_frame", str(self.camera_frame))

    @property
    def translation_m(self) -> np.ndarray:
        """The camera's position in the base frame, in meters."""
        return self.T_base_camera[:3, 3].copy()


def _read_yaml(path: Path, what: str) -> dict[str, Any]:
    import yaml

    if not path.is_file():
        raise CalibrationError(f"{what} not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as stream:
            data = yaml.safe_load(stream)
    except OSError as exc:
        raise CalibrationError(f"could not read {what} {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise CalibrationError(f"{what} {path} is not valid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise CalibrationError(f"{what} {path} must contain a YAML mapping")
    return data


def file_sha256(path: Path) -> str:
    """The SHA256 of a file's TEXT, the way the calibration tools compute it.

    The hand-eye result stores the fingerprint of the stereo file it was solved
    against, and the ROS node that publishes the camera TF computes it the same
    way -- so the harness must not use a different digest (raw bytes) or the
    two would disagree on files that are in fact identical.
    """
    return hashlib.sha256(path.read_text(encoding="utf-8").encode("utf-8")).hexdigest()


def load_stereo_yaml(path: str | Path) -> StereoCalibration:
    """Load the measured stereo calibration (K/D/R/T between the two eyes)."""
    resolved = _expand(str(path))
    try:
        return StereoCalibration.from_yaml(str(resolved))
    except (ValueError, KeyError) as exc:
        raise CalibrationError(f"invalid stereo calibration {resolved}: {exc}") from exc


def resolve_handeye_path(path: str | Path) -> Path:
    """Resolve the eye-to-hand result path, accepting a directory.

    A directory is searched recursively for ``handeye_result_*.yaml`` and the
    most recently written one wins, so a re-run hand-eye session does not
    require editing the config: point it at ``~/g1d_vision_data/handeye``.
    """
    resolved = _expand(str(path))
    if resolved.is_file():
        return resolved
    if resolved.is_dir():
        candidates = sorted(
            resolved.rglob("handeye_result_*.yaml"),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            return candidates[0]
        raise CalibrationError(
            f"no handeye_result_*.yaml under {resolved}; "
            "run the eye-to-hand calibration first"
        )
    raise CalibrationError(f"hand-eye result not found: {resolved}")


def load_handeye_yaml(path: str | Path) -> CameraExtrinsic:
    """Load the eye-to-hand result: the camera's pose in the base frame."""
    resolved = resolve_handeye_path(path)
    data = _read_yaml(resolved, "hand-eye result")
    base_frame = str(data.get("base_frame", DEFAULT_BASE_FRAME))
    camera_frame = str(data.get("camera_frame", DEFAULT_CAMERA_FRAME))
    units = str(data.get("units", "meters"))
    if units != "meters":
        raise CalibrationError(
            f"hand-eye result units must be meters, got {units!r} in {resolved}"
        )
    if "T_torso_camera" not in data and "T_base_camera" not in data:
        raise CalibrationError(
            f"hand-eye result {resolved} has neither T_torso_camera nor T_base_camera"
        )
    key = "T_torso_camera" if "T_torso_camera" in data else "T_base_camera"
    matrix = _matrix4(data[key], key)
    return CameraExtrinsic(
        T_base_camera=matrix,
        base_frame=base_frame,
        camera_frame=camera_frame,
        source=str(resolved),
        intrinsics_sha256=str(data.get("intrinsics_sha256", "")),
    )


def verify_calibration_pairing(
    stereo_path: str | Path, extrinsic: CameraExtrinsic
) -> Optional[str]:
    """Check that the stereo file is the one the extrinsic was solved against.

    Returns the matching SHA256, or ``None`` when the hand-eye result carries
    no fingerprint (older sessions) and no check is possible. Raises
    :class:`CalibrationError` when the fingerprints disagree -- pairing a
    hand-eye result with a re-calibrated stereo file silently corrupts every
    metric depth, so this refuses rather than warns.
    """
    expected = str(extrinsic.intrinsics_sha256 or "").strip()
    if not expected:
        return None
    resolved = _expand(str(stereo_path))
    if not resolved.is_file():
        raise CalibrationError(f"stereo calibration not found: {resolved}")
    actual = file_sha256(resolved)
    if actual != expected:
        raise CalibrationError(
            "the stereo calibration and the hand-eye result come from different "
            f"sessions: {resolved} has SHA256 {actual}, but the hand-eye result "
            f"({extrinsic.source}) was solved against {expected}. Re-run the "
            "eye-to-hand calibration for the current stereo file, or point the "
            "config back at the matching stereo file."
        )
    return actual


def rectified_base_transform(
    extrinsic: CameraExtrinsic, rectification_rotation: np.ndarray
) -> np.ndarray:
    """Compose the base transform of the RECTIFIED left camera frame.

    The extrinsic maps RAW camera vectors into the base frame; the depth map
    lives in the rectified frame, which is the raw frame rotated by ``R1``::

        X_rect = R1 @ X_raw   =>   X_base = R_base_raw @ R1^T @ X_rect + t

    The translation is unchanged: rectification is a rotation about the camera
    centre.
    """
    rotation = np.asarray(rectification_rotation, dtype=np.float64)
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise CalibrationError("rectification rotation must be a finite 3x3 matrix")
    matrix = np.eye(4)
    matrix[:3, :3] = extrinsic.T_base_camera[:3, :3] @ rotation.T
    matrix[:3, 3] = extrinsic.T_base_camera[:3, 3]
    return matrix


@dataclass(frozen=True)
class CalibrationBundle:
    """Everything the G1-D perception path needs to place a detection in 3D."""

    stereo: StereoCalibration
    extrinsic: CameraExtrinsic
    stereo_path: str
    handeye_path: str
    min_depth_m: float = 0.15
    max_depth_m: float = 2.0
    max_stereo_skew_s: float = 0.02

    @property
    def base_frame(self) -> str:
        return self.extrinsic.base_frame

    @property
    def camera_frame(self) -> str:
        return self.extrinsic.camera_frame

    def describe(self) -> str:
        stereo = self.stereo
        translation = self.extrinsic.translation_m
        return (
            f"stereo {self.stereo_path} ({stereo.image_width}x{stereo.image_height}, "
            f"baseline {float(np.linalg.norm(stereo.T_left_to_right_m)) * 1000:.1f} mm); "
            f"hand-eye {self.handeye_path} -> {self.base_frame} "
            f"t=[{translation[0]:.4f}, {translation[1]:.4f}, {translation[2]:.4f}] m"
        )


def load_calibration(
    cfg: Mapping[str, Any],
    *,
    stereo_path: Optional[str] = None,
    handeye_path: Optional[str] = None,
) -> CalibrationBundle:
    """Load the G1-D calibration described by a robot config's ``calibration`` block.

    ``stereo_path`` / ``handeye_path`` override the config (the CLI's explicit
    values win over the config, exactly like the other config knobs). The
    config's block looks like::

        calibration:
          stereo_yaml: ~/.ros/g1d_camera_calibration/head_stereo.yaml
          handeye_yaml: ~/g1d_vision_data/handeye
          base_frame: torso_link
          camera_frame: head_left_camera_optical_frame
          min_depth_m: 0.15
          max_depth_m: 2.0
          max_stereo_skew_s: 0.02
    """
    block = dict(cfg.get("calibration") or {})
    stereo_file = stereo_path or block.get("stereo_yaml")
    handeye_file = handeye_path or block.get("handeye_yaml")
    if not stereo_file or not handeye_file:
        raise CalibrationError(
            "no camera calibration configured: set calibration.stereo_yaml and "
            "calibration.handeye_yaml in the robot config (or pass "
            "--stereo-calibration / --handeye-calibration)"
        )
    stereo = load_stereo_yaml(stereo_file)
    extrinsic = load_handeye_yaml(handeye_file)
    verify_calibration_pairing(stereo_file, extrinsic)
    base_frame = str(block.get("base_frame") or extrinsic.base_frame)
    if base_frame != extrinsic.base_frame:
        raise CalibrationError(
            f"the config asks for base frame {base_frame!r} but the hand-eye "
            f"result is expressed in {extrinsic.base_frame!r}"
        )
    camera_frame = str(block.get("camera_frame") or extrinsic.camera_frame)
    if camera_frame != extrinsic.camera_frame:
        raise CalibrationError(
            f"the config asks for camera frame {camera_frame!r} but the hand-eye "
            f"result is for {extrinsic.camera_frame!r}"
        )
    min_depth = float(block.get("min_depth_m", 0.15))
    max_depth = float(block.get("max_depth_m", 2.0))
    if not 0 < min_depth < max_depth:
        raise CalibrationError(
            f"calibration depth range must satisfy 0 < min_depth_m < max_depth_m "
            f"(got {min_depth} and {max_depth})"
        )
    skew = float(block.get("max_stereo_skew_s", 0.02))
    if not np.isfinite(skew) or skew <= 0:
        raise CalibrationError("calibration max_stereo_skew_s must be finite and positive")
    return CalibrationBundle(
        stereo=stereo,
        extrinsic=extrinsic,
        stereo_path=str(_expand(str(stereo_file))),
        handeye_path=extrinsic.source,
        min_depth_m=min_depth,
        max_depth_m=max_depth,
        max_stereo_skew_s=skew,
    )
