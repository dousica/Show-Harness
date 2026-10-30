"""Tests for the G1-D vision API layer (camera frame -> validated JSON detection).

The API is exercised through a fake transport, so the suite needs no network,
no API key and no ROS. The ROS-facing helpers (image decoding) are covered with
plain message objects.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.g1d.camera_source import decode_image_message
from core.g1d.vision_api import (
    CustomHttpTransport,
    VisionApiClient,
    VisionApiError,
    build_detection_prompt,
    encode_png_base64,
    resize_max_side,
)


class FakeTransport:
    """Records what it was asked and returns a canned answer."""

    name = "fake"

    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.calls: list[dict] = []

    def complete(self, *, frame, prompt, model):
        self.calls.append(
            {
                "frame": frame,
                "prompt": prompt,
                "model": model,
            }
        )
        return self.answer, 0.25


def _frame(height: int = 48, width: int = 64) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, size=(height, width, 3), dtype=np.uint8)


# --- prompt -----------------------------------------------------------------


def test_prompt_states_the_bbox_contract():
    prompt = build_detection_prompt("red steel bearing")
    assert "red steel bearing" in prompt
    assert "[x_min, y_min, x_max, y_max]" in prompt
    assert "[0, 1]" in prompt


def test_prompt_rejects_an_empty_target():
    with pytest.raises(ValueError):
        build_detection_prompt("   ")


def test_prompt_excludes_motion_from_the_model():
    prompt = build_detection_prompt("bearing")
    assert "joint angles" in prompt
    assert "robot motion" in prompt


# --- parsing ----------------------------------------------------------------


def test_a_valid_detection_is_returned():
    answer = json.dumps(
        {"label": "bearing", "bbox_norm": [0.1, 0.2, 0.5, 0.6], "confidence": 0.9}
    )
    client = VisionApiClient(FakeTransport(answer), model="m")
    result = client.detect(_frame(), "bearing")
    assert result.found is True
    assert result.label == "bearing"
    assert result.confidence == pytest.approx(0.9)
    assert result.bbox_norm == (0.1, 0.2, 0.5, 0.6)
    assert result.detection is not None
    assert result.detection.bbox_norm == (0.1, 0.2, 0.5, 0.6)


def test_an_empty_label_means_not_found():
    answer = json.dumps(
        {"label": "", "bbox_norm": [0.0, 0.0, 0.0, 0.0], "confidence": 0.0}
    )
    client = VisionApiClient(FakeTransport(answer), model="m")
    result = client.detect(_frame(), "bearing")
    assert result.found is False
    assert result.detection is None


def test_zero_confidence_means_not_found():
    answer = json.dumps(
        {"label": "bearing", "bbox_norm": [0.1, 0.2, 0.5, 0.6], "confidence": 0.0}
    )
    client = VisionApiClient(FakeTransport(answer), model="m")
    assert client.detect(_frame(), "bearing").found is False


def test_unparsable_output_means_not_found():
    client = VisionApiClient(FakeTransport("the bearing is on the left"), model="m")
    result = client.detect(_frame(), "bearing")
    assert result.found is False
    assert result.raw_text == "the bearing is on the left"


def test_an_out_of_range_box_means_not_found():
    answer = json.dumps(
        {"label": "bearing", "bbox_norm": [0.1, 0.2, 1.4, 0.6], "confidence": 0.9}
    )
    client = VisionApiClient(FakeTransport(answer), model="m")
    assert client.detect(_frame(), "bearing").found is False


def test_a_confidence_below_the_threshold_means_not_found():
    answer = json.dumps(
        {"label": "bearing", "bbox_norm": [0.1, 0.2, 0.5, 0.6], "confidence": 0.2}
    )
    client = VisionApiClient(FakeTransport(answer), model="m", min_confidence=0.7)
    result = client.detect(_frame(), "bearing")
    assert result.found is False
    assert result.confidence == pytest.approx(0.2)


def test_the_result_serialises_to_json():
    answer = json.dumps(
        {"label": "bearing", "bbox_norm": [0.1, 0.2, 0.5, 0.6], "confidence": 0.9}
    )
    client = VisionApiClient(FakeTransport(answer), model="m")
    record = client.detect(_frame(), "bearing").to_dict()
    assert record["action_sent"] is False
    assert record["bbox_norm"] == [0.1, 0.2, 0.5, 0.6]
    assert json.loads(json.dumps(record))["found"] is True


def test_an_invalid_threshold_is_rejected():
    with pytest.raises(ValueError):
        VisionApiClient(FakeTransport("{}"), min_confidence=1.5)
    with pytest.raises(ValueError):
        VisionApiClient(FakeTransport("{}"), min_confidence=-0.1)


# --- image handling ---------------------------------------------------------


def test_a_float_frame_is_scaled_to_uint8():
    frame = np.full((4, 4, 3), 0.5, dtype=np.float32)
    assert encode_png_base64(frame)  # does not raise


def test_a_chw_frame_is_transposed_to_hwc():
    frame = np.zeros((3, 5, 7), dtype=np.uint8)
    resized = resize_max_side(frame, 0)
    assert resized.shape == (5, 7, 3)


def test_resizing_caps_the_longest_side():
    frame = _frame(height=480, width=640)
    resized = resize_max_side(frame, 320)
    assert max(resized.shape[:2]) == 320
    # The aspect ratio is preserved.
    assert resized.shape[1] / resized.shape[0] == pytest.approx(640 / 480, rel=0.02)


def test_resizing_is_a_no_op_when_the_frame_is_small():
    frame = _frame(height=48, width=64)
    assert resize_max_side(frame, 320).shape == frame.shape


def test_the_result_reports_the_size_the_model_saw():
    answer = json.dumps(
        {"label": "bearing", "bbox_norm": [0.1, 0.2, 0.5, 0.6], "confidence": 0.9}
    )
    client = VisionApiClient(FakeTransport(answer), model="m", max_image_side=32)
    result = client.detect(_frame(height=480, width=640), "bearing")
    assert max(result.image_size) == 32


# --- image decoding ---------------------------------------------------------


class _Stamp:
    sec = 1
    nanosec = 0


class _Header:
    stamp = _Stamp()
    frame_id = "head_left_camera_optical_frame"


class _Msg:
    def __init__(self, data: bytes, encoding: str = "bgr8") -> None:
        self.data = data
        self.encoding = encoding
        self.height = 2
        self.width = 2
        self.step = 6
        self.header = _Header()


def test_bgr8_is_converted_to_rgb():
    # One blue pixel (BGR) must decode to (0, 0, 255) in RGB.
    msg = _Msg(bytes([255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
    frame = decode_image_message(msg)
    assert frame.shape == (2, 2, 3)
    assert tuple(frame[0, 0]) == (0, 0, 255)


def test_rgb8_is_kept_as_is():
    msg = _Msg(bytes([255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), encoding="rgb8")
    assert tuple(decode_image_message(msg)[0, 0]) == (255, 0, 0)


def test_an_unknown_encoding_is_rejected():
    msg = _Msg(bytes(12), encoding="mono8")
    with pytest.raises(ValueError):
        decode_image_message(msg)


# --- custom transport -------------------------------------------------------


class _Response:
    def __init__(self, status_code: int, payload=None, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        if self._payload is None:
            raise ValueError("no JSON body")
        return self._payload


def test_the_custom_transport_renders_the_request_template(monkeypatch):
    captured: dict = {}

    def fake_request(method, url, json=None, headers=None, timeout=None):
        captured["method"] = method
        captured["url"] = url
        captured["body"] = json
        captured["headers"] = headers
        return _Response(
            200,
            {"data": {"result": {"label": "bearing", "bbox_norm": [0, 0, 1, 1],
                                "confidence": 0.8}}},
        )

    import requests

    monkeypatch.setattr(requests, "request", fake_request)
    transport = CustomHttpTransport(
        base_url="https://gateway.example/detect",
        request_template={"model": "{model}", "image": "{image_base64}"},
        response_path="data.result",
        api_key="secret",
    )
    text, latency = transport.complete(frame=_frame(), prompt="p", model="m")
    assert captured["method"] == "POST"
    assert captured["body"]["model"] == "m"
    assert captured["body"]["image"]
    assert captured["headers"]["Authorization"] == "Bearer secret"
    assert json.loads(text)["label"] == "bearing"
    assert latency >= 0.0


def test_the_custom_transport_reports_an_http_error(monkeypatch):
    import requests

    monkeypatch.setattr(
        requests,
        "request",
        lambda *a, **k: _Response(500, text="upstream exploded"),
    )
    transport = CustomHttpTransport(
        base_url="https://gateway.example/detect",
        request_template={"image": "{image_base64}"},
        response_path="result",
        max_retries=0,
    )
    with pytest.raises(VisionApiError):
        transport.complete(frame=_frame(), prompt="p", model="m")


def test_the_custom_transport_reports_a_missing_response_path(monkeypatch):
    import requests

    monkeypatch.setattr(
        requests, "request", lambda *a, **k: _Response(200, {"data": {}})
    )
    transport = CustomHttpTransport(
        base_url="https://gateway.example/detect",
        request_template={"image": "{image_base64}"},
        response_path="data.result",
        max_retries=0,
    )
    with pytest.raises(VisionApiError):
        transport.complete(frame=_frame(), prompt="p", model="m")


def test_the_custom_transport_rejects_an_unknown_placeholder(monkeypatch):
    import requests

    monkeypatch.setattr(
        requests, "request", lambda *a, **k: _Response(200, {"result": {}})
    )
    transport = CustomHttpTransport(
        base_url="https://gateway.example/detect",
        request_template={"image": "{nope}"},
        response_path="result",
        max_retries=0,
    )
    with pytest.raises(VisionApiError):
        transport.complete(frame=_frame(), prompt="p", model="m")


def test_the_custom_transport_requires_a_base_url():
    with pytest.raises(ValueError):
        CustomHttpTransport(
            base_url="",
            request_template={"image": "{image_base64}"},
            response_path="result",
        )
