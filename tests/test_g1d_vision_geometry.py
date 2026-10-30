"""Offline contracts for single-arm selection and RGB-D target geometry."""
from __future__ import annotations

import unittest

import numpy as np

from core.g1d import (
    SingleArmProfile,
    StereoCalibration,
    StereoDepthProcessor,
    VLMDetection,
    depth_from_disparity,
    detection_to_base_point,
    estimate_box_depth,
    pixel_to_camera_xyz,
    transform_point_4x4,
    translate_pose_in_tool_axis,
)


class SingleArmProfileTests(unittest.TestCase):
    def test_profile_selects_exactly_one_arm_and_gripper(self) -> None:
        profile = SingleArmProfile("left")
        self.assertEqual(profile.group_name, "left_arm")
        self.assertEqual(profile.gripper_action, "/lzh_left_dex1/gripper_command")
        profile.validate_trajectory_joint_names(profile.joint_names)
        with self.assertRaisesRegex(ValueError, "exactly its 7 arm joints"):
            profile.validate_trajectory_joint_names(
                profile.joint_names + SingleArmProfile("right").joint_names
            )

    def test_invalid_side_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "left.*right"):
            SingleArmProfile("both")  # type: ignore[arg-type]


class VisionGeometryTests(unittest.TestCase):
    def test_vlm_contract_is_json_detection_only(self) -> None:
        detection = VLMDetection.from_json(
            '{"label":"cup","bbox_norm":[0.25,0.2,0.75,0.8],"confidence":0.9}'
        )
        self.assertEqual(detection.label, "cup")
        self.assertAlmostEqual(detection.confidence, 0.9)
        with self.assertRaisesRegex(ValueError, "bbox_norm"):
            VLMDetection.from_json(
                {"label": "cup", "bbox_norm": [0.8, 0.2, 0.2, 0.8], "confidence": 1}
            )

    def test_disparity_converts_to_metric_depth(self) -> None:
        depth = depth_from_disparity(
            np.asarray([[10.0, 0.0, np.nan]]), focal_length_px=100.0, baseline_m=0.1
        )
        self.assertAlmostEqual(depth[0, 0], 1.0)
        self.assertTrue(np.isnan(depth[0, 1]))
        self.assertTrue(np.isnan(depth[0, 2]))
        offset_depth = depth_from_disparity(
            np.asarray([[12.0]]), focal_length_px=100.0, baseline_m=0.1,
            disparity_offset_px=2.0,
        )
        self.assertAlmostEqual(offset_depth[0, 0], 1.0)

    def test_box_depth_uses_center_region_and_rejects_empty_depth(self) -> None:
        depth = np.full((20, 30), np.nan)
        depth[6:14, 9:21] = 1.25
        estimate = estimate_box_depth(depth, [0.2, 0.2, 0.8, 0.8], min_valid_pixels=5)
        self.assertAlmostEqual(estimate.median_m, 1.25)
        self.assertGreaterEqual(estimate.valid_pixels, 5)
        with self.assertRaisesRegex(ValueError, "valid depth pixels"):
            estimate_box_depth(np.full((20, 30), np.nan), [0.2, 0.2, 0.8, 0.8])

    def test_pixel_projection_and_camera_to_base_transform(self) -> None:
        point_camera = pixel_to_camera_xyz(60, 50, 2.0, fx=100, fy=100, cx=50, cy=50)
        np.testing.assert_allclose(point_camera, [0.2, 0.0, 2.0])
        transform = np.eye(4)
        transform[:3, 3] = [1.0, 2.0, 3.0]
        np.testing.assert_allclose(
            transform_point_4x4(point_camera, transform), [1.2, 2.0, 5.0]
        )

    def test_detection_to_base_point(self) -> None:
        detection = VLMDetection("target", (0.25, 0.25, 0.75, 0.75), 0.95)
        depth = np.ones((20, 20), dtype=np.float64)
        point, estimate = detection_to_base_point(
            detection, depth, fx=10, fy=10, cx=10, cy=10,
            T_base_camera=np.eye(4), min_confidence=0.8
        )
        np.testing.assert_allclose(point, [-0.05, -0.05, 1.0])
        self.assertEqual(estimate.valid_pixels, 36)
        with self.assertRaisesRegex(ValueError, "below"):
            detection_to_base_point(
                detection, depth, fx=10, fy=10, cx=10, cy=10,
                T_base_camera=np.eye(4), min_confidence=0.99
            )

    def test_tool_axis_step_rotates_with_tool_and_has_limit(self) -> None:
        pose = [0, 0, 0, 0, 0, np.sqrt(0.5), np.sqrt(0.5)]
        target = translate_pose_in_tool_axis(pose, 0.01, axis="x", max_step_m=0.02)
        np.testing.assert_allclose(target[:3], [0, 0.01, 0], atol=1e-8)
        with self.assertRaisesRegex(ValueError, "exceeds limit"):
            translate_pose_in_tool_axis(pose, 0.03, axis="x", max_step_m=0.02)

    def test_stereo_processor_rectifies_and_returns_metric_depth_image(self) -> None:
        try:
            import cv2  # noqa: F401
        except ImportError:
            self.skipTest("OpenCV is an optional G1-D stereo dependency")
        calibration = StereoCalibration(
            image_width=64,
            image_height=48,
            K_left=np.asarray([[80, 0, 32], [0, 80, 24], [0, 0, 1]], dtype=float),
            D_left=np.zeros(5),
            K_right=np.asarray([[80, 0, 32], [0, 80, 24], [0, 0, 1]], dtype=float),
            D_right=np.zeros(5),
            R_left_to_right=np.eye(3),
            T_left_to_right_m=np.asarray([[-0.06], [0], [0]], dtype=float),
        )
        processor = StereoDepthProcessor(
            calibration, num_disparities=16, block_size=3,
            min_depth_m=0.1, max_depth_m=5.0,
        )
        rng = np.random.default_rng(4)
        left = rng.integers(0, 255, size=(48, 64, 1), dtype=np.uint8)
        left = np.repeat(left, 3, axis=2)
        right = np.zeros_like(left)
        right[:, :-4] = left[:, 4:]
        rectified, depth, disparity = processor.compute(left, right)
        self.assertEqual(rectified.shape, left.shape)
        self.assertEqual(depth.shape, left.shape[:2])
        self.assertEqual(disparity.shape, left.shape[:2])
        self.assertEqual(len(processor.rectified_intrinsics), 4)
        valid = depth[12:36, 30:58]
        valid = valid[np.isfinite(valid)]
        self.assertGreater(valid.size, 20)
        self.assertAlmostEqual(float(np.median(valid)), 1.2, delta=0.25)


if __name__ == "__main__":
    unittest.main()
