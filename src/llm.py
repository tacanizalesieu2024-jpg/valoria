"""One small wrapper around the LLM API, so the rest of the code never talks to it directly.
It uses the OpenAI Python library, which also works with other providers that offer an
OpenAI-compatible endpoint (default: Google Gemini, free tier). Provider = settings in .env.

Reliability: if the main model is overloaded or unavailable (errors 429/503/404), the
call is retried with the backup models in LLM_FALLBACK_MODELS."""
import base64
import json
import re

from src.config import LLM_API_KEY, LLM_BASE_URL, LLM_FALLBACK_MODELS, LLM_MODEL

_client = None
RETRYABLE = ("429", "500", "503", "404", "overloaded", "unavailable", "not found", "RESOURCE_EXHAUSTED")


def _get_client():
    global _client
    if _client is None:
        from openai import OpenAI
        _client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL or None, max_retries=1, timeout=60)
    return _client


def _call(**kwargs):
    """Try the main model, then each backup model."""
    last_error = None
    for model in [LLM_MODEL] + [m for m in LLM_FALLBACK_MODELS if m != LLM_MODEL]:
        try:
            return _get_client().chat.completions.create(model=model, **kwargs)
        except Exception as e:  # noqa: BLE001 - we re-raise below if nothing works
            if not any(s.lower() in str(e).lower() for s in RETRYABLE) and "response_format" not in str(e):
                raise
            last_error = e
    raise last_error


def _mime(img: bytes) -> str:
    """Tell the AI the real image type (screenshots are PNG, Idealista photos are often WEBP)."""
    if img[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if img[:4] == b"RIFF" and img[8:12] == b"WEBP":
        return "image/webp"
    return "image/jpeg"


def _parse_json(text: str) -> dict:
    """Read JSON even if the model wraps it in ```json ... ``` fences."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    return json.loads(text)


def ask_json(system: str, user: str, images: list[bytes] | None = None) -> dict:
    """Ask the LLM and get a Python dict back. Optional images for the vision step."""
    content = [{"type": "text", "text": user}]
    for img in images or []:
        b64 = base64.b64encode(img).decode()
        content.append({"type": "image_url", "image_url": {"url": f"data:{_mime(img)};base64,{b64}"}})
    messages = [{"role": "system", "content": system}, {"role": "user", "content": content}]
    try:
        resp = _call(temperature=0, response_format={"type": "json_object"}, messages=messages)
    except Exception as e:
        if "response_format" not in str(e):
            raise
        resp = _call(temperature=0, messages=messages)  # provider without JSON mode
    return _parse_json(resp.choices[0].message.content)


def ask_text(system: str, user: str) -> str:
    resp = _call(temperature=0.3,
                 messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
    return resp.choices[0].message.content
