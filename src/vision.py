"""Step 2 - Look at the photos: a vision LLM rates the flat's condition.

Bonus feature (Week 3). If it is not ready in time, the app simply skips it.
"""
from src.config import USE_LLM

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
    return ask_json(SYSTEM_PROMPT, "Rate the condition of this flat.", images=images[:4])
