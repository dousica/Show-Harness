"""The preflight checker: passes on a complete config, fails with guidance
when site identity is missing. Offline modes only (--no-vlm --no-hardware)."""
from __future__ import annotations

import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.check_setup import main

GOOD = """\
hardware: franka
robot:
  nuc_ip: "10.0.0.9"
  external_camera_serial: "000000000001"
  wrist_camera_serial: "000000000002"
enable_z_floor: true
z_floor_name: default
z_floors: {default: 0.14}
vlm_backend: local
vlm: {base_url: http://localhost:8000/v1}
vlm_backends:
  local: {provider: vllm, base_url: http://localhost:8000/v1, model: m,
          api_key_env: PREFLIGHT_TEST_API_KEY}
"""


G1D = """\
hardware: g1d
robot:
  side: left
  bridge_namespace: lzh
enable_z_floor: true
z_floor_m: 0.0
z_floor_calibrated: true
z_floor_margin_m: 0.003
primitives_config: __PRIMS__
vlm_backend: local
vlm: {base_url: http://localhost:8000/v1}
vlm_backends:
  local: {provider: vllm, base_url: http://localhost:8000/v1, model: m,
          api_key_env: PREFLIGHT_TEST_API_KEY}
"""

PRIMS = str(Path(__file__).resolve().parents[1] / "configs" / "primitives_g1d.yaml")


class CheckSetupTests(unittest.TestCase):
    def _run(self, cfg_text: str, *, domain: str = "42", overlay: str = "") -> tuple[int, str]:
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "robot.yaml"
            p.write_text(cfg_text)
            out = io.StringIO()
            env = {
                "ROS_DOMAIN_ID": domain,
                "AMENT_PREFIX_PATH": overlay or "third_party/g1d_ros2_lzh/install/lzh_bringup",
                "LZH_ROS_WS": "third_party/g1d_ros2_lzh",
            }
            with patch.dict(os.environ, env), contextlib.redirect_stdout(out):
                code = main(["--robot-config", str(p), "--no-vlm", "--no-hardware"])
        return code, out.getvalue()

    def test_complete_config_passes(self) -> None:
        code, out = self._run(GOOD)
        self.assertEqual(code, 0, out)
        self.assertIn("z floor", out)
        self.assertIn("vlm backend", out)

    def test_missing_identity_fails_with_site_hint(self) -> None:
        code, out = self._run(GOOD.replace('  nuc_ip: "10.0.0.9"\n', ""))
        self.assertEqual(code, 1, out)
        self.assertIn("site/franka.yaml.example", out)

    def test_placeholder_z_floor_fails(self) -> None:
        code, out = self._run(GOOD.replace("z_floors: {default: 0.14}",
                                           "z_floors: {default: 0.0}"))
        self.assertEqual(code, 1, out)
        self.assertIn("placeholder", out)

    def test_g1d_calibrated_zero_floor_passes(self) -> None:
        code, out = self._run(G1D.replace("__PRIMS__", PRIMS))
        self.assertEqual(code, 0, out)
        self.assertIn("z floor", out)
        self.assertNotIn("PLACEHOLDER", out)

    def test_g1d_uncalibrated_zero_floor_fails(self) -> None:
        code, out = self._run(
            G1D.replace("__PRIMS__", PRIMS).replace("z_floor_calibrated: true\n", "")
        )
        self.assertEqual(code, 1, out)
        self.assertIn("PLACEHOLDER", out)

    def test_g1d_preflight_rejects_missing_domain(self) -> None:
        code, out = self._run(G1D.replace("__PRIMS__", PRIMS), domain="")
        self.assertEqual(code, 1, out)
        self.assertIn("ROS_DOMAIN_ID must be 42", out)

    def test_g1d_preflight_rejects_other_domain(self) -> None:
        code, out = self._run(G1D.replace("__PRIMS__", PRIMS), domain="0")
        self.assertEqual(code, 1, out)
        self.assertIn("got '0'", out)

    def test_g1d_preflight_rejects_senior_overlay(self) -> None:
        code, out = self._run(
            G1D.replace("__PRIMS__", PRIMS),
            overlay="/home/yushu/g1d_ros2_ws/install/g1d_bringup",
        )
        self.assertEqual(code, 1, out)
        self.assertIn("forbidden overlay path", out)

    def test_no_vlm_does_not_require_an_api_key(self) -> None:
        code, out = self._run(G1D.replace("__PRIMS__", PRIMS))
        self.assertEqual(code, 0, out)
        self.assertIn("not needed for this check", out)


if __name__ == "__main__":
    unittest.main()
