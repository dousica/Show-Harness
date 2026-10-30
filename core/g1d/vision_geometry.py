"""Offline RGB-D target parsing and geometry helpers for the G1-D pilot.

Depth values passed to these functions must already be metric meters, rectified,
and aligned pixel-for-pixel to the RGB image. This module does not run stereo
matching, call a VLM, read ROS topics, or issue robot commands.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class VLMDetection:
    """A VLM's object label and normalized xyxy box; deliberately no robot action."""

    label: str
    bbox_norm: tuple[float, float, float, float]
    confidence: float

    @classmethod
    def from_json(cls, payload: str | bytes | dict) -> "VLMDetection":
        try:
            value = json.loads(payload) if isinstance(payload, (str, bytes)) else payload
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError("VLM response must be valid JSON") from exc
        if not isinstance(value, dict):
            raise ValueError("VLM response must be a JSON object")
        label = value.get("label")
        bbox = value.get("bbox_norm")
        confidence = value.get("confidence")
        if not isinstance(label, str) or not label.strip():
            raise ValueError("VLM response label must be a non-empty string")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            raise ValueError("bbox_norm must be [x_min, y_min, x_max, y_max]")
        try:
            box = tuple(float(item) for item in bbox)
            score = float(confidence)
        except (TypeError, ValueError) as exc:
            raise ValueError("bbox_norm and confidence must be numeric") from exc
        if not np.all(np.isfinite(box)) or not np.isfinite(score):
            raise ValueError("bbox_norm and confidence must be finite")
        x0, y0, x1, y1 = box
        if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
            raise ValueError("bbox_norm must be ordered and inside [0, 1]")
        if not 0.0 <= score <= 1.0:
            raise ValueError("confidence must be inside [0, 1]")
        return cls(label=label.strip(), bbox_norm=box, confidence=score)


@dataclass(frozen=True)
class DepthEstimate:
    median_m: float
    mad_m: float
    valid_pixels: int
    pixel_xy: tuple[float, float]


def depth_from_disparity(
    disparity_px: np.ndarray,
    *,
    focal_length_px: float,
    baseline_m: float,
    disparity_offset_px: float = 0.0,
    min_depth_m: float = 0.05,
    max_depth_m: float = 10.0,
) -> np.ndarray:
    """Convert rectified disparity in pixels to depth in meters (invalid -> NaN)."""
    disparity = np.asarray(disparity_px, dtype=np.float64)
    if disparity.ndim != 2:
        raise ValueError("disparity_px must be a 2D image")
    focal = float(focal_length_px)
    baseline = float(baseline_m)
    offset = float(disparity_offset_px)
    if not np.isfinite(focal) or focal <= 0:
        raise ValueError("focal_length_px must be finite and positive")
    if not np.isfinite(baseline) or baseline <= 0:
        raise ValueError("baseline_m must be finite and positive")
    if not np.isfinite(offset):
        raise ValueError("disparity_offset_px must be finite")
    if not (0 < min_depth_m < max_depth_m):
        raise ValueError("depth range must satisfy 0 < min_depth_m < max_depth_m")
    result = np.full(disparity.shape, np.nan, dtype=np.float64)
    corrected = disparity - offset
    valid = np.isfinite(corrected) & (corrected > 0)
    result[valid] = focal * baseline / corrected[valid]
    result[(result < min_depth_m) | (result > max_depth_m)] = np.nan
    return result


@dataclass(frozen=True)
class StereoCalibration:
    """Measured stereo calibration for a rectified left/right camera pair.

    The extrinsic follows the convention the stereo tools write and OpenCV
    expects: ``X_right = R_left_to_right @ X_left + T_left_to_right_m``,
    with ``T_left_to_right_m`` in meters. This is EXACTLY what
    :func:`cv2.stereoRectify` wants, so the stored values pass straight
    through -- do not invert them.
    """

    image_width: int
    image_height: int
    K_left: np.ndarray
    D_left: np.ndarray
    K_right: np.ndarray
    D_right: np.ndarray
    R_left_to_right: np.ndarray
    T_left_to_right_m: np.ndarray

    def __post_init__(self) -> None:
        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError("stereo image dimensions must be positive")
        expected = {
            "K_left": (3, 3),
            "K_right": (3, 3),
            "R_left_to_right": (3, 3),
            "T_left_to_right_m": (3, 1),
        }
        for name, shape in expected.items():
            matrix = np.asarray(getattr(self, name), dtype=np.float64)
            if matrix.shape != shape or not np.all(np.isfinite(matrix)):
                raise ValueError(f"{name} must be a finite {shape[0]}x{shape[1]} matrix")
            object.__setattr__(self, name, matrix.copy())
        for name in ("D_left", "D_right"):
            distortion = np.asarray(getattr(self, name), dtype=np.float64).reshape(-1)
            if (distortion.size not in (4, 5, 8, 12, 14)
                    or not np.all(np.isfinite(distortion))):
                raise ValueError(
                    f"{name} must contain 4, 5, 8, 12, or 14 finite coefficients"
                )
            object.__setattr__(self, name, distortion.copy())
        for name in ("K_left", "K_right"):
            matrix = getattr(self, name)
            if matrix[0, 0] <= 0 or matrix[1, 1] <= 0 or not np.isclose(matrix[2, 2], 1.0):
                raise ValueError(f"{name} must have positive focal lengths and K[2,2] = 1")
        rotation = self.R_left_to_right
        if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3)
                or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-3)):
            raise ValueError("R_left_to_right must be a proper rotation matrix")
        baseline = float(np.linalg.norm(self.T_left_to_right_m))
        if not 0.01 <= baseline <= 1.0:
            raise ValueError("measured stereo baseline must be between 0.01 and 1.0 meters")
        if abs(float(self.T_left_to_right_m[0, 0])) < 0.9 * baseline:
            raise ValueError("StereoDepthProcessor currently requires a horizontal camera baseline")

    @classmethod
    def from_yaml(cls, path: str) -> "StereoCalibration":
        """Load explicitly measured K/D/R/T values; T is in meters."""
        import yaml

        with open(path, "r", encoding="utf-8") as stream:
            data = yaml.safe_load(stream)
        if not isinstance(data, dict):
            raise ValueError("stereo calibration YAML must contain a mapping")
        try:
            return cls(
                image_width=int(data["image_width"]),
                image_height=int(data["image_height"]),
                K_left=np.asarray(data["K_left"], dtype=np.float64),
                D_left=np.asarray(data["D_left"], dtype=np.float64),
                K_right=np.asarray(data["K_right"], dtype=np.float64),
                D_right=np.asarray(data["D_right"], dtype=np.float64),
                R_left_to_right=np.asarray(data["R_left_to_right"], dtype=np.float64),
                T_left_to_right_m=np.asarray(data["T_left_to_right_m"], dtype=np.float64).reshape(3, 1),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid stereo calibration YAML: {exc}") from exc


class StereoDepthProcessor:
    """Rectify synchronized RGB pairs and compute metric depth with OpenCV SGBM."""

    def __init__(
        self,
        calibration: StereoCalibration,
        *,
        min_disparity_px: int = 0,
        num_disparities: int = 128,
        block_size: int = 5,
        min_depth_m: float = 0.15,
        max_depth_m: float = 4.0,
        uniqueness_ratio: int = 10,
        speckle_window_size: int = 80,
        speckle_range: int = 2,
    ) -> None:
        if num_disparities <= 0 or num_disparities % 16:
            raise ValueError("num_disparities must be a positive multiple of 16")
        if block_size < 3 or block_size % 2 == 0:
            raise ValueError("block_size must be an odd integer >= 3")
        if not 0 < min_depth_m < max_depth_m:
            raise ValueError("depth range must satisfy 0 < min_depth_m < max_depth_m")
        self.calibration = calibration
        self.min_depth_m = float(min_depth_m)
        self.max_depth_m = float(max_depth_m)
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - deployment dependency
            raise ImportError("OpenCV (cv2) is required for G1-D stereo matching") from exc
        self._cv2 = cv2
        size = (calibration.image_width, calibration.image_height)
        R1, R2, P1, P2, _, _, _ = cv2.stereoRectify(
            calibration.K_left,
            calibration.D_left,
            calibration.K_right,
            calibration.D_right,
            size,
            calibration.R_left_to_right,
            calibration.T_left_to_right_m,
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=0,
        )
        self.rectified_P_left = P1
        self.rectified_P_right = P2
        self.rectification_rotation_left = R1.copy()
        self._left_map = cv2.initUndistortRectifyMap(
            calibration.K_left, calibration.D_left, R1, P1, size, cv2.CV_32FC1
        )
        self._right_map = cv2.initUndistortRectifyMap(
            calibration.K_right, calibration.D_right, R2, P2, size, cv2.CV_32FC1
        )
        self._matcher = cv2.StereoSGBM_create(
            minDisparity=int(min_disparity_px),
            numDisparities=int(num_disparities),
            blockSize=int(block_size),
            P1=8 * block_size * block_size,
            P2=32 * block_size * block_size,
            disp12MaxDiff=1,
            uniquenessRatio=int(uniqueness_ratio),
            speckleWindowSize=int(speckle_window_size),
            speckleRange=int(speckle_range),
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )
        self._fx = float(P1[0, 0])
        self._cx = float(P1[0, 2])
        self._cy = float(P1[1, 2])
        self._baseline = abs(float(P2[0, 3] / P2[0, 0]))
        self._disparity_offset = float(P1[0, 2] - P2[0, 2])
        self._min_disparity_px = int(min_disparity_px)

    @property
    def rectified_intrinsics(self) -> tuple[float, float, float, float]:
        return self._fx, self._fx, self._cx, self._cy

    def compute(self, left_rgb: np.ndarray, right_rgb: np.ndarray):
        """Return (rectified_left_rgb, depth_m, disparity_px), invalid depth as NaN."""
        cv2 = self._cv2
        left = np.asarray(left_rgb)
        right = np.asarray(right_rgb)
        expected = (self.calibration.image_height, self.calibration.image_width, 3)
        if left.shape != expected or right.shape != expected:
            raise ValueError(f"stereo RGB images must both have shape {expected}")
        if left.dtype != np.uint8 or right.dtype != np.uint8:
            raise ValueError("stereo RGB images must use uint8 channels")
        rectified_left = cv2.remap(left, *self._left_map, interpolation=cv2.INTER_LINEAR)
        rectified_right = cv2.remap(right, *self._right_map, interpolation=cv2.INTER_LINEAR)
        gray_left = cv2.cvtColor(rectified_left, cv2.COLOR_RGB2GRAY)
        gray_right = cv2.cvtColor(rectified_right, cv2.COLOR_RGB2GRAY)
        disparity = self._matcher.compute(gray_left, gray_right).astype(np.float32) / 16.0
        corrected_disparity = disparity - self._disparity_offset
        valid = (
            np.isfinite(corrected_disparity)
            & (corrected_disparity > self._min_disparity_px)
        )
        depth = np.full(disparity.shape, np.nan, dtype=np.float32)
        depth[valid] = self._fx * self._baseline / corrected_disparity[valid]
        depth[(depth < self.min_depth_m) | (depth > self.max_depth_m)] = np.nan
        return np.ascontiguousarray(rectified_left), depth, disparity


def estimate_box_depth(
    depth_m: np.ndarray,
    bbox_norm: Sequence[float],
    *,
    min_valid_pixels: int = 5,
    center_fraction: float = 0.6,
    min_depth_m: float = 0.05,
    max_depth_m: float = 10.0,
) -> DepthEstimate:
    """Use the center of a normalized box and robust median/MAD depth statistics."""
    depth = np.asarray(depth_m, dtype=np.float64)
    if depth.ndim != 2 or not depth.size:
        raise ValueError("depth_m must be a non-empty 2D image in meters")
    if len(bbox_norm) != 4:
        raise ValueError("bbox_norm must contain four normalized xyxy values")
    box = np.asarray(bbox_norm, dtype=np.float64)
    if not np.all(np.isfinite(box)):
        raise ValueError("bbox_norm must be finite")
    x0, y0, x1, y1 = (float(v) for v in box)
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        raise ValueError("bbox_norm must be ordered and inside [0, 1]")
    if not (0 < center_fraction <= 1):
        raise ValueError("center_fraction must be in (0, 1]")
    if min_valid_pixels < 1:
        raise ValueError("min_valid_pixels must be positive")
    if not (0 < min_depth_m < max_depth_m):
        raise ValueError("depth range must satisfy 0 < min_depth_m < max_depth_m")

    height, width = depth.shape
    left, top = int(np.floor(x0 * width)), int(np.floor(y0 * height))
    right, bottom = int(np.ceil(x1 * width)), int(np.ceil(y1 * height))
    box_w, box_h = right - left, bottom - top
    inset_x = int((1.0 - center_fraction) * box_w / 2.0)
    inset_y = int((1.0 - center_fraction) * box_h / 2.0)
    left = max(0, left + inset_x)
    right = min(width, max(left + 1, right - inset_x))
    top = max(0, top + inset_y)
    bottom = min(height, max(top + 1, bottom - inset_y))
    roi = depth[top:bottom, left:right]
    valid = roi[np.isfinite(roi) & (roi >= min_depth_m) & (roi <= max_depth_m)]
    if valid.size < min_valid_pixels:
        raise ValueError(
            f"only {valid.size} valid depth pixels in target box; need {min_valid_pixels}"
        )
    median = float(np.median(valid))
    mad = float(np.median(np.abs(valid - median)))
    u = (left + right - 1) / 2.0
    v = (top + bottom - 1) / 2.0
    return DepthEstimate(median, mad, int(valid.size), (u, v))


def pixel_to_camera_xyz(
    u_px: float,
    v_px: float,
    depth_m: float,
    *,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    """Project one rectified optical-frame pixel into XYZ meters."""
    values = np.asarray([u_px, v_px, depth_m, fx, fy, cx, cy], dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("pixel, depth, and intrinsics must be finite")
    u, v, z, fx, fy, cx, cy = (float(x) for x in values)
    if z <= 0 or fx <= 0 or fy <= 0:
        raise ValueError("depth and focal lengths must be positive")
    return np.asarray([(u - cx) * z / fx, (v - cy) * z / fy, z], dtype=np.float64)


def transform_point_4x4(point_xyz: Sequence[float], T_target_source: np.ndarray) -> np.ndarray:
    """Transform a 3D point between frames using a validated homogeneous matrix."""
    point = np.asarray(point_xyz, dtype=np.float64).reshape(-1)
    matrix = np.asarray(T_target_source, dtype=np.float64)
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        raise ValueError("point_xyz must contain three finite values")
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError("T_target_source must be a finite 4x4 matrix")
    if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-6):
        raise ValueError("T_target_source must have homogeneous last row [0, 0, 0, 1]")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3):
        raise ValueError("T_target_source rotation must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-3):
        raise ValueError("T_target_source rotation determinant must be +1")
    return rotation @ point + matrix[:3, 3]


def translate_pose_in_tool_axis(
    pose7_xyzw: Sequence[float],
    distance_m: float,
    *,
    axis: str,
    max_step_m: float,
) -> np.ndarray:
    """Translate an EE pose along its local x/y/z axis without changing orientation."""
    pose = np.asarray(pose7_xyzw, dtype=np.float64).reshape(-1)
    distance = float(distance_m)
    limit = float(max_step_m)
    if pose.shape != (7,) or not np.all(np.isfinite(pose)):
        raise ValueError("pose must be 7 finite values [x,y,z,qx,qy,qz,qw]")
    if axis not in ("x", "y", "z"):
        raise ValueError("axis must be 'x', 'y', or 'z'")
    if not np.isfinite(distance) or not np.isfinite(limit) or limit <= 0:
        raise ValueError("distance must be finite and max_step_m must be positive")
    if abs(distance) > limit:
        raise ValueError(f"requested step {distance:.4f} m exceeds limit {limit:.4f} m")
    quat = pose[3:].copy()
    norm = float(np.linalg.norm(quat))
    if norm < 1e-8:
        raise ValueError("pose quaternion must be non-zero")
    x, y, z, w = quat / norm
    rotation = np.asarray([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ])
    direction = np.eye(3)[("x", "y", "z").index(axis)]
    result = pose.copy()
    result[:3] += rotation @ direction * distance
    result[3:] = quat / norm
    return result


def detection_to_base_point(
    detection: VLMDetection,
    depth_m: np.ndarray,
    *,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    T_base_camera: np.ndarray,
    min_confidence: float,
    min_valid_pixels: int = 5,
) -> tuple[np.ndarray, DepthEstimate]:
    """Convert an accepted VLM box and aligned metric depth to a base point."""
    threshold = float(min_confidence)
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("min_confidence must be finite and inside [0, 1]")
    if detection.confidence < threshold:
        raise ValueError(
            f"detection confidence {detection.confidence:.3f} is below "
            f"the required {threshold:.3f}"
        )
    estimate = estimate_box_depth(
        depth_m, detection.bbox_norm, min_valid_pixels=min_valid_pixels
    )
    camera_point = pixel_to_camera_xyz(
        *estimate.pixel_xy, estimate.median_m, fx=fx, fy=fy, cx=cx, cy=cy
    )
    return transform_point_4x4(camera_point, T_base_camera), estimate
