"""One small wrapper around the LLM API, so the rest of the code never talks to it directly.
Switching provider (OpenAI -> Gemini, Claude, ...) only means changing this file."""
import base64
import json

from src.config import LLM_API_KEY, LLM_MODEL

_client = None


def _get_client():
    global _client
    if _client is None:
        from openai import OpenAI
        _client = OpenAI(api_key=LLM_API_KEY)
    return _client


def ask_json(system: str, user: str, images: list[bytes] | None = None) -> dict:
    """Ask the LLM and get a Python dict back (JSON mode). Optional images for the vision step."""
    content = [{"type": "text", "text": user}]
    for img in images or []:
        b64 = base64.b64encode(img).decode()
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    resp = _get_client().chat.completions.create(
        model=LLM_MODEL,
        temperature=0,  # same input -> same output: important for a reliable live demo
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": system}, {"role": "user", "content": content}],
    )
    return json.loads(resp.choices[0].message.content)


def ask_text(system: str, user: str) -> str:
    resp = _get_client().chat.completions.create(
        model=LLM_MODEL,
        temperature=0.3,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    return resp.choices[0].message.content
