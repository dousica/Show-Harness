"""G1-D vision API: one RGB frame in, one validated JSON detection out.

Scope
-----
This module is the API half of the G1-D perception path. It takes a single RGB
frame plus a target description and returns a validated
:class:`~core.g1d.vision_geometry.VLMDetection`. It does not subscribe to ROS, does
not read depth, and issues no robot command -- the harness decides what to do with
the answer.

Two transports
--------------
``openai_compat`` (default) reuses :class:`core.vlm.vlm_client.VLMClient`, so the
G1-D path inherits the harness's provider dialects (vLLM / OpenAI / Gemini), its
rate-limit retries and its API-key refresh instead of re-implementing them.

``custom`` posts a JSON body built from a field template and reads the answer out
of a dotted response path. It exists because the deployment gateway is not
guaranteed to speak the OpenAI dialect: point ``request_template`` at the real
field names and the same client drives it.

The JSON contract
-----------------
The model answers with ONE object::

    {"label": "cup", "bbox_norm": [0.25, 0.20, 0.70, 0.85], "confidence": 0.92}

``bbox_norm`` is ``[x_min, y_min, x_max, y_max]`` normalised to ``[0, 1]`` on the
FULL input image. The answer carries NO joint angles, NO free-form distances and
NO executable ROS command -- perception and motion stay decoupled, exactly like
the rest of the G1-D contract (see ``docs/g1d.md``).

"Not visible" is expressed in-band, not as an error: an empty ``label`` (or a
zero confidence) yields a :class:`VisionResult` with ``found=False``. Only a
transport failure raises.
"""

from __future__ import annotations

import base64
import io
import json
import time
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Protocol

import numpy as np
from PIL import Image

from core.g1d.vision_geometry import VLMDetection

__all__ = [
    "DETECTION_SCHEMA",
    "CustomHttpTransport",
    "OpenAICompatTransport",
    "VisionApiClient",
    "VisionApiError",
    "VisionResult",
    "VisionTransport",
    "build_detection_prompt",
    "build_vision_client",
]


#: JSON schema the model is constrained to on the ``openai_compat`` transport. The
#: custom transport sends it as prompt text instead (see ``build_detection_prompt``).
DETECTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "label": {"type": "string"},
        "bbox_norm": {
            "type": "array",
            "items": {"type": "number"},
            "minItems": 4,
            "maxItems": 4,
        },
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
    "required": ["label", "bbox_norm", "confidence"],
    "additionalProperties": False,
}


class VisionApiError(RuntimeError):
    """Raised when the vision API itself fails (network, HTTP status, bad config)."""


def _to_uint8_hwc(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 3:
        raise ValueError(f"Expected an HWC image, got shape {array.shape}")
    if array.shape[0] == 3 and array.shape[-1] != 3:
        array = np.transpose(array, (1, 2, 0))
    if np.issubdtype(array.dtype, np.floating):
        array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    elif array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def encode_png_base64(image: np.ndarray) -> str:
    """Encode an RGB frame as base64 PNG, without a data-URL prefix."""
    buffer = io.BytesIO()
    Image.fromarray(_to_uint8_hwc(image)).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def resize_max_side(image: np.ndarray, max_side: int) -> np.ndarray:
    """Downscale so the longest side is at most ``max_side`` pixels (0 disables).

    Vision gateways reject oversized bodies (commonly ~20 MB of base64), and a
    640x480 frame is already fine, so the default is to leave frames alone. The
    aspect ratio is preserved; the result is what the model actually saw, which is
    why the returned result reports the RESIZED size.
    """
    limit = int(max_side)
    array = _to_uint8_hwc(image)
    if limit <= 0:
        return array
    height, width = array.shape[:2]
    longest = max(height, width)
    if longest <= limit:
        return array
    scale = limit / float(longest)
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    resized = Image.fromarray(array).resize(new_size, Image.LANCZOS)
    return np.ascontiguousarray(np.asarray(resized, dtype=np.uint8))


def build_detection_prompt(target: str, *, extra_instruction: str = "") -> str:
    """The instruction sent with every detection request.

    The wording is a CONTRACT, not a suggestion: it is the same for the guided
    (JSON-schema) transport and for the custom transport, so a deployment gateway
    and the reference path produce comparable answers.
    """
    target_text = str(target or "").strip()
    if not target_text:
        raise ValueError("target description must not be empty")
    parts = [
        f"Find the object matching this description in the image: {target_text}.",
        "Return the tight visible bounding box as normalized [x_min, y_min, x_max, "
        "y_max] coordinates in [0, 1], measured on the full input image.",
        "If the object is not clearly visible, return an empty label, the bbox "
        "[0, 0, 0, 0] and confidence 0.",
        "Return only the required JSON object. Do not estimate metric depth, do not "
        "propose robot motion, and do not include joint angles or distances.",
    ]
    extra = str(extra_instruction or "").strip()
    if extra:
        parts.append(extra)
    return "\n".join(parts)


@dataclass(frozen=True)
class VisionResult:
    """The outcome of one detection request.

    ``found=False`` means the model answered "not visible" (or answered something
    unparsable) -- that is a normal outcome, not a failure. ``bbox_norm`` is
    ``None`` whenever nothing was detected.
    """

    found: bool
    label: str
    confidence: float
    bbox_norm: Optional[tuple[float, float, float, float]]
    raw_text: str
    latency_s: float
    image_size: tuple[int, int]
    model: str
    transport: str

    @property
    def detection(self) -> Optional[VLMDetection]:
        """The detection as the harness's canonical dataclass, or ``None``."""
        if not self.found or self.bbox_norm is None:
            return None
        return VLMDetection(
            label=self.label,
            bbox_norm=self.bbox_norm,
            confidence=self.confidence,
        )

    def to_dict(self) -> dict[str, Any]:
        """A JSON-serialisable record, one per line, as the CLI prints it."""
        return {
            "found": self.found,
            "label": self.label,
            "confidence": self.confidence,
            "bbox_norm": list(self.bbox_norm) if self.bbox_norm is not None else None,
            "image_size": [int(self.image_size[0]), int(self.image_size[1])],
            "model": self.model,
            "transport": self.transport,
            "vlm_latency_s": round(float(self.latency_s), 3),
            "action_sent": False,
        }


class VisionTransport(Protocol):
    """What a transport must provide: turn a prompt plus an image into raw text."""

    name: str

    def complete(
        self,
        *,
        frame: np.ndarray,
        prompt: str,
        model: str,
    ) -> tuple[str, float]:
        """Return ``(raw_text, latency_s)``; raise :class:`VisionApiError` on failure."""
        ...


class OpenAICompatTransport:
    """Talk to any OpenAI-compatible ``/chat/completions`` endpoint.

    Backed by :class:`core.vlm.vlm_client.VLMClient`, so provider quirks
    (``max_completion_tokens`` on OpenAI, ``reasoning_effort`` on Gemini, guided
    decoding on vLLM), retry-on-429 and API-key refresh are handled by the harness,
    not here.
    """

    name = "openai_compat"

    def __init__(self, client: Any, *, use_schema: bool = True) -> None:
        self._client = client
        self._use_schema = bool(use_schema)

    def complete(
        self,
        *,
        frame: np.ndarray,
        prompt: str,
        model: str,
    ) -> tuple[str, float]:
        del model  # the client already carries the configured model id
        try:
            response = self._client.complete_json(
                prompt,
                frame,
                schema=DETECTION_SCHEMA if self._use_schema else None,
                agentview_label=None,
                wrist_label=None,
            )
        except Exception as exc:  # noqa: BLE001 - re-raised as a vision API error
            raise VisionApiError(f"vision API call failed: {exc}") from exc
        payload = response.payload.get("json")
        raw_text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return raw_text, float(response.payload.get("latency_s", 0.0))


class CustomHttpTransport:
    """POST a templated JSON body to any HTTP API and read a dotted path out.

    Configured from ``configs/robot_g1d.yaml``::

        vision_api:
          transport: custom
          base_url: https://gateway.example/v1/detect
          method: POST
          headers:
            X-Tenant: g1d
          request_template:
            model: "{model}"
            image: "{image_base64}"
            prompt: "{prompt}"
          response_path: "data.result"

    ``request_template`` values are format strings with the placeholders
    ``{model}``, ``{image_base64}``, ``{image_data_url}`` and ``{prompt}``.
    ``response_path`` walks the decoded JSON with dots; a numeric segment indexes
    a list (``"data.detections.0"``).
    """

    name = "custom"

    def __init__(
        self,
        *,
        base_url: str,
        request_template: Mapping[str, str],
        response_path: str,
        method: str = "POST",
        headers: Optional[Mapping[str, str]] = None,
        api_key: str = "",
        api_key_header: str = "Authorization",
        api_key_prefix: str = "Bearer ",
        timeout_s: float = 120.0,
        max_retries: int = 3,
    ) -> None:
        if not str(base_url).strip():
            raise ValueError("custom transport requires a base_url")
        if not request_template:
            raise ValueError("custom transport requires a non-empty request_template")
        if not str(response_path).strip():
            raise ValueError("custom transport requires a response_path")
        self.base_url = str(base_url).rstrip("/")
        self.request_template = dict(request_template)
        self.response_path = str(response_path).strip()
        self.method = str(method or "POST").upper()
        self.headers = {str(k): str(v) for k, v in dict(headers or {}).items()}
        self.api_key = str(api_key or "")
        self.api_key_header = str(api_key_header or "Authorization")
        self.api_key_prefix = str(api_key_prefix)
        self.timeout_s = float(timeout_s)
        self.max_retries = max(0, int(max_retries))

    def _render(self, values: Mapping[str, str]) -> dict[str, str]:
        rendered: dict[str, str] = {}
        for key, template in self.request_template.items():
            try:
                rendered[str(key)] = str(template).format_map(values)
            except KeyError as exc:
                raise VisionApiError(
                    f"request_template field {key!r} references unknown placeholder "
                    f"{exc}; available: {sorted(values)}"
                ) from exc
        return rendered

    def complete(
        self,
        *,
        frame: np.ndarray,
        prompt: str,
        model: str,
    ) -> tuple[str, float]:
        import requests

        image_base64 = encode_png_base64(frame)
        values = {
            "model": model,
            "image_base64": image_base64,
            "image_data_url": f"data:image/png;base64,{image_base64}",
            "prompt": prompt,
        }
        body = self._render(values)
        headers = {"Content-Type": "application/json", **self.headers}
        if self.api_key:
            headers[self.api_key_header] = f"{self.api_key_prefix}{self.api_key}"
        started = time.monotonic()
        last = "unknown error"
        for attempt in range(self.max_retries + 1):
            try:
                response = requests.request(
                    self.method,
                    self.base_url,
                    json=body,
                    headers=headers,
                    timeout=self.timeout_s,
                )
            except requests.RequestException as exc:
                last = f"request error: {exc}"
            else:
                if response.status_code < 400:
                    try:
                        payload = response.json()
                    except ValueError as exc:
                        last = f"non-JSON response body: {exc}"
                    else:
                        extracted = _extract_path(payload, self.response_path)
                        if extracted is None:
                            last = (
                                f"response_path {self.response_path!r} not found in the "
                                f"response body"
                            )
                        else:
                            latency = time.monotonic() - started
                            text = (
                                extracted
                                if isinstance(extracted, str)
                                else json.dumps(extracted, ensure_ascii=False)
                            )
                            return text, latency
                else:
                    last = f"HTTP {response.status_code}: {response.text[:500]}"
                    if response.status_code in (400, 401, 403, 404, 422):
                        break  # a caller/auth error: retrying cannot fix it
            if attempt >= self.max_retries:
                break
            time.sleep(min(2.0 ** attempt, 8.0))
        raise VisionApiError(f"vision API call failed: {last}")


def _extract_path(payload: Any, path: str) -> Any:
    """Walk a dotted path through decoded JSON; numeric segments index lists."""
    current = payload
    for part in str(path).split("."):
        if current is None:
            return None
        if isinstance(current, dict):
            if part not in current:
                return None
            current = current[part]
        elif isinstance(current, list):
            try:
                current = current[int(part)]
            except (ValueError, IndexError):
                return None
        else:
            return None
    return current


class VisionApiClient:
    """Send frames to a vision API and turn the answer into a validated detection.

    The client owns everything that is not transport-specific: image encoding and
    optional downscaling, prompt construction, JSON parsing, and the
    "not visible" convention. The transport owns the wire format.
    """

    def __init__(
        self,
        transport: VisionTransport,
        *,
        model: str = "",
        min_confidence: float = 0.0,
        max_image_side: int = 0,
    ) -> None:
        threshold = float(min_confidence)
        if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("min_confidence must be finite and inside [0, 1]")
        self.transport = transport
        self.model = str(model or "")
        self.min_confidence = threshold
        self.max_image_side = int(max_image_side)

    def detect(
        self,
        image: np.ndarray,
        target: str,
        *,
        extra_instruction: str = "",
    ) -> VisionResult:
        """Run one detection request and return the validated outcome."""
        frame = resize_max_side(image, self.max_image_side)
        height, width = frame.shape[:2]
        prompt = build_detection_prompt(target, extra_instruction=extra_instruction)
        raw_text, latency = self.transport.complete(
            frame=frame,
            prompt=prompt,
            model=self.model,
        )
        return self._parse(raw_text, latency, (width, height))

    def _parse(
        self,
        raw_text: str,
        latency_s: float,
        image_size: tuple[int, int],
    ) -> VisionResult:
        def result(
            *,
            found: bool,
            label: str = "",
            confidence: float = 0.0,
            bbox: Optional[tuple[float, float, float, float]] = None,
        ) -> VisionResult:
            return VisionResult(
                found=found,
                label=label,
                confidence=confidence,
                bbox_norm=bbox,
                raw_text=raw_text,
                latency_s=latency_s,
                image_size=image_size,
                model=self.model,
                transport=self.transport.name,
            )

        text = str(raw_text or "").strip()
        if not text:
            return result(found=False)
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return result(found=False)
        if not isinstance(payload, dict):
            return result(found=False)

        label = payload.get("label")
        if not isinstance(label, str):
            label = ""
        label = label.strip()
        try:
            confidence = float(payload.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        if not np.isfinite(confidence):
            confidence = 0.0
        confidence = min(1.0, max(0.0, confidence))

        # The "not visible" answer: empty label, zero confidence, or an empty box.
        if not label or confidence <= 0.0:
            return result(found=False, label=label, confidence=confidence)
        try:
            detection = VLMDetection.from_json(payload)
        except ValueError:
            # A malformed box is not a usable detection; report it as "not found"
            # rather than crashing the caller, but keep the raw answer for the log.
            return result(found=False, label=label, confidence=confidence)
        if detection.confidence < self.min_confidence:
            return result(
                found=False,
                label=detection.label,
                confidence=detection.confidence,
            )
        return result(
            found=True,
            label=detection.label,
            confidence=detection.confidence,
            bbox=detection.bbox_norm,
        )


def build_vision_client(
    cfg: Mapping[str, Any],
    *,
    backend: Optional[str] = None,
    transport: Optional[str] = None,
) -> VisionApiClient:
    """Build a client from a robot config (the ``vlm`` / ``vision_api`` blocks).

    ``cfg`` is the parsed YAML of a ``configs/robot_*.yaml``. The ``openai_compat``
    transport reads the harness's standard ``vlm`` + ``vlm_backends`` blocks, so the
    API endpoint, model and credential are configured in exactly one place for the
    whole harness. The ``custom`` transport reads the ``vision_api`` block.
    """
    from core.config import (  # noqa: PLC0415
        make_api_key_refresher,
        resolve_api_key,
        resolve_vlm_config,
    )
    from core.vlm.vlm_client import VLMClient  # noqa: PLC0415

    vision_cfg = dict(cfg.get("vision_api") or {})
    selected = str(transport or vision_cfg.get("transport") or "openai_compat").lower()
    min_confidence = float(vision_cfg.get("min_confidence", 0.0))
    max_image_side = int(vision_cfg.get("max_image_side", 0))

    if selected in ("openai_compat", "openai", "vlm"):
        vlm_cfg = resolve_vlm_config(dict(cfg), backend=backend)
        client = VLMClient(
            base_url=vlm_cfg["base_url"],
            model=vlm_cfg["model"],
            api_key=vlm_cfg.get("api_key", "EMPTY"),
            timeout_s=float(vlm_cfg.get("timeout_s", 120)),
            max_tokens=int(vision_cfg.get("max_tokens", vlm_cfg.get("max_tokens", 2048))),
            temperature=float(vision_cfg.get("temperature", 0.0)),
            chat_template_kwargs=vlm_cfg.get("chat_template_kwargs", {}),
            provider=vlm_cfg.get("provider", "vllm"),
            api_dialect=vlm_cfg.get("api_dialect"),
            reasoning_effort=vlm_cfg.get("reasoning_effort"),
            max_retries=vlm_cfg.get("max_retries"),
            api_key_refresh=make_api_key_refresher(vlm_cfg),
        )
        return VisionApiClient(
            OpenAICompatTransport(client),
            model=str(vlm_cfg["model"]),
            min_confidence=min_confidence,
            max_image_side=max_image_side,
        )

    if selected == "custom":
        transport_impl = CustomHttpTransport(
            base_url=str(vision_cfg.get("base_url", "")),
            request_template=dict(vision_cfg.get("request_template") or {}),
            response_path=str(vision_cfg.get("response_path", "")),
            method=str(vision_cfg.get("method", "POST")),
            headers=dict(vision_cfg.get("headers") or {}),
            api_key=resolve_api_key(vision_cfg),
            api_key_header=str(vision_cfg.get("api_key_header", "Authorization")),
            api_key_prefix=str(vision_cfg.get("api_key_prefix", "Bearer ")),
            timeout_s=float(vision_cfg.get("timeout_s", 120)),
            max_retries=int(vision_cfg.get("max_retries", 3)),
        )
        return VisionApiClient(
            transport_impl,
            model=str(vision_cfg.get("model", "")),
            min_confidence=min_confidence,
            max_image_side=max_image_side,
        )

    raise VisionApiError(
        f"unknown vision_api.transport {selected!r}; "
        "expected 'openai_compat' or 'custom'"
    )
