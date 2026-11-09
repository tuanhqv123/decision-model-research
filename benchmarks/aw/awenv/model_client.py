"""HTTP client for the d1-2B decision server.

The contract is additive over the existing ``/v1/decide`` endpoint, so a server that
ignores ``elements`` and ``return_readouts`` still works for the visual track.

Latency is split into ``model_ms`` (server-reported) and ``wall_ms`` (client-observed),
because the two diverging is how GPU contention shows up.
"""
import base64
import io
import json
import os
import time
from typing import Any, Dict, List, Optional

DEFAULT_URL = os.environ.get("D1_URL", "http://127.0.0.1:8700")


def _read_token() -> str:
    if os.environ.get("D1_API_TOKEN"):
        return os.environ["D1_API_TOKEN"].strip()
    for path in ("~/.d1-api-token", "~/d1-api/.env"):
        expanded = os.path.expanduser(path)
        if not os.path.exists(expanded):
            continue
        text = open(expanded, "r", encoding="utf-8").read().strip()
        if "=" in text:
            for line in text.splitlines():
                if line.startswith("D1_API_TOKEN="):
                    return line.split("=", 1)[1].strip()
        return text
    raise RuntimeError("no D1 API token: set D1_API_TOKEN or ~/.d1-api-token")


def encode_png(pixels: Any) -> str:
    """Encode a HxWx3 uint8 array as base64 PNG without requiring PIL."""
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - PIL ships with the model image
        raise RuntimeError("Pillow is required to encode screenshots") from exc
    buf = io.BytesIO()
    Image.fromarray(pixels).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


class ModelClient:
    def __init__(self, base_url: str = DEFAULT_URL, token: Optional[str] = None, timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.token = token or _read_token()
        self.timeout = timeout

    @property
    def _headers(self) -> Dict[str, str]:
        return {"content-type": "application/json", "x-api-key": self.token}

    def health(self) -> Dict[str, Any]:
        import requests

        return requests.get("%s/health" % self.base_url, timeout=30).json()

    def decide(
        self,
        goal: str,
        history: Optional[List[str]] = None,
        pixels: Any = None,
        image_b64: Optional[str] = None,
        elements: Optional[List[Dict[str, Any]]] = None,
        track: str = "V",
        return_readouts: bool = False,
    ) -> Dict[str, Any]:
        import requests

        if image_b64 is None:
            if pixels is None:
                raise ValueError("decide() needs pixels or image_b64")
            image_b64 = encode_png(pixels)

        payload = {
            "goal": goal,
            "history": history or [],
            "image_b64": image_b64,
            "track": track,
        }
        if elements is not None:
            payload["elements"] = elements
        if return_readouts:
            payload["return_readouts"] = True

        started = time.time()
        response = requests.post(
            "%s/v1/decide" % self.base_url,
            data=json.dumps(payload),
            headers=self._headers,
            timeout=self.timeout,
        )
        response.raise_for_status()
        decision = response.json()
        decision["wall_ms"] = (time.time() - started) * 1000.0
        lat = decision.setdefault("latency_ms", {})
        if "model" not in lat:
            lat["model"] = decision["wall_ms"]
        return decision
