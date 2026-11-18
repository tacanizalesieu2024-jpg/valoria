"""Step 2 - Look at the photos: a vision LLM rates the flat's condition.

Bonus feature (Week 3). If it is not ready in time, the app simply skips it.
"""
import io

from src.config import USE_LLM


def shrink(img: bytes, max_side: int = 1024) -> bytes:
    """Resize a photo to at most 1024 px and save as JPEG: faster and cheaper to send to the AI."""
    try:
        from PIL import Image
        im = Image.open(io.BytesIO(img)).convert("RGB")
        im.thumbnail((max_side, max_side))
        out = io.BytesIO()
        im.save(out, format="JPEG", quality=85)
        return out.getvalue()
    except Exception:  # noqa: BLE001 - if anything fails, send the original
        return img

SYSTEM_PROMPT = """You are an experienced property appraiser in Madrid.
Look at the photos of one flat and return ONLY a JSON object:
- condition: "new", "good" or "to_renovate"
- kitchen_score, bathroom_score, finishes_score: 1 (very poor) to 5 (excellent), or null if not visible
- notes: one short sentence in English explaining your rating
Judge only what is visible. Do not invent rooms that are not in the photos."""


def assess_photos(images: list[bytes]) -> dict | None:
    if not images:
        return None
    if not USE_LLM:
        return {"condition": None, "notes": "Photo analysis needs an API key (mock mode)."}
    from src.llm import ask_json
    small = [shrink(img) for img in images[:4]]
    return ask_json(SYSTEM_PROMPT, "Rate the condition of this flat.", images=small)
