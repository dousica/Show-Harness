"""Offline contracts for the G1-D calibration files.

No ROS, no network, no robot: these tests load the SPEC EXTRACTED FROM the
measured files on this workstation (``~/.ros/g1d_camera_calibration/`` and the
hand-eye session), so they verify that the harness's loaders UNDERSTAND the
lab's solver outputs -- not that the measurements themselves are right (that
is the ChArUco physical check's job).
"""
from __future__ import annotations

import unittest

import numpy as np
import yaml

from core.g1d.calibration import (
    CalibrationError,
    file_sha256,
    load_handeye_yaml,
    load_stereo_yaml,
    rectified_base_transform,
    verify_calibration_pairing,
)

try:
    from core.g1d.calibration import load_calibration

    _HAS_LOAD_CALIBRATION = True
except ImportError:  # pragma: no cover
    _HAS_LOAD_CALIBRATION = False

MEASURED_STEREO = "/home/yushu/.ros/g1d_camera_calibration/head_stereo.yaml"
MEASURED_HANDEYE = (
    "/home/yushu/g1d_vision_data/handeye/head_left_20261028_161926_575326/"
    "handeye_result_20261028_165527_739176.yaml"
)
# A checked-in path is a lab-machines-only PATH, not a secret: it names no
# credential, model, or host. Keep it in one constant so a fork changes it
# in one place.


class LoadStereoTests(unittest.TestCase):
    def test_measured_stereo_loads_with_the_solvers_key_names(self) -> None:
        stereo = load_stereo_yaml(MEASURED_STEREO)
        self.assertEqual((stereo.image_width, stereo.image_height), (640, 480))
        # The solvers write R_left_to_right / T_left_to_right_m (the OpenCV
        # stereoRectify convention), and the loaders must accept exactly that.
        raw = yaml.safe_load(open(MEASURED_STEREO, encoding="utf-8"))
        np.testing.assert_allclose(
            stereo.R_left_to_right, np.asarray(raw["R_left_to_right"]), rtol=1e-12
        )
        np.testing.assert_allclose(
            stereo.T_left_to_right_m,
            np.asarray(raw["T_left_to_right_m"]).reshape(3, 1),
            rtol=1e-12,
        )
        baseline = float(np.linalg.norm(stereo.T_left_to_right_m))
        self.assertAlmostEqual(baseline, 0.0601, places=3)

    def test_stereo_baseline_is_horizontal(self) -> None:
        stereo = load_stereo_yaml(MEASURED_STEREO)
        baseline = float(np.linalg.norm(stereo.T_left_to_right_m))
        self.assertGreater(abs(float(stereo.T_left_to_right_m[0, 0])), 0.9 * baseline)


class LoadHandeyeTests(unittest.TestCase):
    def test_measured_handeye_loads_in_meters_against_torso(self) -> None:
        extrinsic = load_handeye_yaml(MEASURED_HANDEYE)
        self.assertEqual(extrinsic.base_frame, "torso_link")
        self.assertEqual(extrinsic.camera_frame, "head_left_camera_optical_frame")
        # The camera sits above the torso origin, not at it: it must never be
        # an identity-looking matrix.
        self.assertGreater(float(np.linalg.norm(extrinsic.translation_m)), 0.3)
        self.assertNotEqual(extrinsic.intrinsics_sha256, "")

    def test_wrong_units_are_rejected(self) -> None:
        import tempfile
        from pathlib import Path

        path = Path(tempfile.mkdtemp()) / "handeye.yaml"
        raw = yaml.safe_load(open(MEASURED_HANDEYE, encoding="utf-8"))
        raw["units"] = "millimeters"
        path.write_text(yaml.safe_dump(raw), encoding="utf-8")
        with self.assertRaisesRegex(CalibrationError, "meters"):
            load_handeye_yaml(path)


class PairingTests(unittest.TestCase):
    def test_measured_files_pair_off_the_same_stereo_session(self) -> None:
        extrinsic = load_handeye_yaml(MEASURED_HANDEYE)
        digest = verify_calibration_pairing(MEASURED_STEREO, extrinsic)
        self.assertEqual(digest, file_sha256(__import__("pathlib").Path(MEASURED_STEREO)))

    def test_a_recalibrated_stereo_is_rejected(self) -> None:
        import shutil
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            edited = Path(tmp) / "head_stereo.yaml"
            shutil.copy(MEASURED_STEREO, edited)
            raw = yaml.safe_load(edited.read_text(encoding="utf-8"))
            raw["baseline_m"] = float(raw["baseline_m"]) + 0.001
            edited.write_text(yaml.safe_dump(raw), encoding="utf-8")
            extrinsic = load_handeye_yaml(MEASURED_HANDEYE)
            with self.assertRaisesRegex(CalibrationError, "different sessions"):
                verify_calibration_pairing(str(edited), extrinsic)


class RectifiedTransformTests(unittest.TestCase):
    def test_rectified_base_keeps_translation_and_rotates_orientation(self) -> None:
        extrinsic = load_handeye_yaml(MEASURED_HANDEYE)
        try:
            import cv2
        except ImportError:
            self.skipTest("OpenCV is required for the rectification rotation")
        stereo = load_stereo_yaml(MEASURED_STEREO)
        size = (stereo.image_width, stereo.image_height)
        R1, _, _, _, _, _, _ = cv2.stereoRectify(
            stereo.K_left, stereo.D_left, stereo.K_right, stereo.D_right, size,
            stereo.R_left_to_right, stereo.T_left_to_right_m,
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=0,
        )
        composed = rectified_base_transform(extrinsic, R1)
        # Rectification rotates about the camera centre: translation is unchanged.
        np.testing.assert_allclose(composed[:3, 3], extrinsic.translation_m, rtol=1e-12)
        np.testing.assert_allclose(
            composed[:3, :3], extrinsic.T_base_camera[:3, :3] @ R1.T, rtol=1e-9
        )
        det = float(np.linalg.det(composed[:3, :3]))
        self.assertAlmostEqual(det, 1.0, places=6)


@unittest.skipUnless(_HAS_LOAD_CALIBRATION, "load_calibration helper")
class LoadCalibrationTests(unittest.TestCase):
    def test_robot_config_resolves_to_the_measured_files(self) -> None:
        import shutil
        import sys
        import tempfile
        from pathlib import Path

        repo = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(repo))
        from core.config import load_yaml

        with tempfile.TemporaryDirectory() as tmp:
            # The shipped robot_g1d.yaml requires a site layer; in a fresh
            # clone that layer only exists as an example, so resolve the
            # config against a site copy filled with the LAB's measured paths.
            staged = Path(tmp) / "robot_g1d.yaml"
            shutil.copy(repo / "configs/robot_g1d.yaml", staged)
            site = staged.parent / "site_g1d.yaml"
            site.write_text(
                "calibration:\n"
                "  stereo_yaml: /home/yushu/.ros/g1d_camera_calibration/head_stereo.yaml\n"
                f"  handeye_yaml: {MEASURED_HANDEYE}\n"
                "  camera_frame: head_left_camera_optical_frame\n"
                "  base_frame: torso_link\n",
                encoding="utf-8",
            )
            text = staged.read_text(encoding="utf-8").replace(
                "  - site/g1d.yaml", f"  - {site.name}"
            )
            staged.write_text(text, encoding="utf-8")
            cfg = load_yaml(staged)
            bundle = load_calibration(cfg)
        self.assertEqual(bundle.base_frame, "torso_link")
        self.assertEqual(bundle.camera_frame, "head_left_camera_optical_frame")
        np.testing.assert_allclose(
            bundle.extrinsic.T_base_camera,
            load_handeye_yaml(MEASURED_HANDEYE).T_base_camera,
            rtol=1e-12,
        )


if __name__ == "__main__":
    unittest.main()
