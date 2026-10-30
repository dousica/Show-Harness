from types import SimpleNamespace

from core.g1d.ros_transport import _ensure_rclpy_initialized


def test_transport_initializes_rclpy_before_creating_nodes():
    calls = []
    context = SimpleNamespace(ok=lambda: False, init=lambda **kwargs: calls.append(kwargs))

    _ensure_rclpy_initialized({"rclpy": context})

    assert calls == [{"args": None}]


def test_transport_reuses_existing_rclpy_context():
    calls = []
    context = SimpleNamespace(ok=lambda: True, init=lambda **kwargs: calls.append(kwargs))

    _ensure_rclpy_initialized({"rclpy": context})

    assert calls == []
