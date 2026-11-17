"""One small wrapper around the LLM API, so the rest of the code never talks to it directly.
It uses the OpenAI Python library, which also works with other providers that offer an
OpenAI-compatible endpoint (default: Google Gemini, free tier). Provider = settings in .env."""
import base64
import json
import re

from src.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL

_client = None


def _get_client():
    global _client
    if _client is None:
        from openai import OpenAI
        _client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL or None)
    return _client


def _parse_json(text: str) -> dict:
    """Read JSON even if the model wraps it in ```json ... ``` fences."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    return json.loads(text)


def ask_json(system: str, user: str, images: list[bytes] | None = None) -> dict:
    """Ask the LLM and get a Python dict back. Optional images for the vision step."""
    content = [{"type": "text", "text": user}]
    for img in images or []:
        b64 = base64.b64encode(img).decode()
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    messages = [{"role": "system", "content": system}, {"role": "user", "content": content}]
    try:
        resp = _get_client().chat.completions.create(
            model=LLM_MODEL,
            temperature=0,  # same input -> same output: important for a reliable live demo
            response_format={"type": "json_object"},
            messages=messages,
        )
    except Exception as e:
        if "response_format" not in str(e):
            raise
        # Some providers don't accept response_format: ask again without it
        resp = _get_client().chat.completions.create(model=LLM_MODEL, temperature=0, messages=messages)
    return _parse_json(resp.choices[0].message.content)


def ask_text(system: str, user: str) -> str:
    resp = _get_client().chat.completions.create(
        model=LLM_MODEL,
        temperature=0.3,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    return resp.choices[0].message.content
