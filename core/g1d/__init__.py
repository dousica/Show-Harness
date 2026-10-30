"""G1-D interfaces, one-arm profile, and offline perception geometry."""

from .camera_source import G1DCameraSource, decode_image_message
from .g1d_interface import G1DRobotInterface, G1DTransport, MockG1DTransport
from .single_arm import ArmSide, SingleArmProfile
from .vision_api import (
    DETECTION_SCHEMA,
    CustomHttpTransport,
    OpenAICompatTransport,
    VisionApiClient,
    VisionApiError,
    VisionResult,
    VisionTransport,
    build_detection_prompt,
    build_vision_client,
)
from .vision_geometry import (
    DepthEstimate,
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

__all__ = [
    "ArmSide",
    "CustomHttpTransport",
    "DETECTION_SCHEMA",
    "DepthEstimate",
    "G1DCameraSource",
    "G1DRobotInterface",
    "G1DTransport",
    "MockG1DTransport",
    "OpenAICompatTransport",
    "SingleArmProfile",
    "StereoCalibration",
    "StereoDepthProcessor",
    "VLMDetection",
    "VisionApiClient",
    "VisionApiError",
    "VisionResult",
    "VisionTransport",
    "build_detection_prompt",
    "build_vision_client",
    "decode_image_message",
    "depth_from_disparity",
    "detection_to_base_point",
    "estimate_box_depth",
    "pixel_to_camera_xyz",
    "transform_point_4x4",
    "translate_pose_in_tool_axis",
]
