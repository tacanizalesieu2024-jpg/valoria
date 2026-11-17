"""Step 1 - Read the listing: LLM turns free text into the model's features (JSON).

Without an API key, a simple keyword parser is used instead (mock mode), so the
app always runs. Comparing the two is a nice point for the documentation:
the keyword parser misses anything not written in the exact expected words.
"""
import re

from src.config import USE_LLM

BOOL_FIELDS = [
    "has_terrace", "has_lift", "has_air_conditioning", "has_parking", "has_storage_room",
    "has_built_in_wardrobe", "has_pool", "has_doorman", "has_garden", "is_duplex",
    "is_studio", "is_top_floor", "is_exterior",
    "orientation_south", "orientation_north", "orientation_east", "orientation_west",
]

SYSTEM_PROMPT = f"""You are a real estate data assistant for Madrid.
Read a property listing (Spanish or English) and return ONLY a JSON object with these keys:
- area_m2 (number), rooms (int), baths (int), floor (int, 0 = ground floor), year_built (int or null)
- {", ".join(BOOL_FIELDS)}: 1 if the listing clearly says so, else 0
- condition: one of "new", "good", "to_renovate", "unknown"
- address: street and neighbourhood if mentioned, else null
- asking_price: number in euros if mentioned, else null
- highlights: list of up to 4 short selling points in English
Never guess numbers that are not in the text: use null instead."""


def extract_features(listing_text: str) -> dict:
    if USE_LLM:
        return _clean(_llm_extract(listing_text))
    return _clean(_keyword_extract(listing_text))


def _llm_extract(text: str) -> dict:
    from src.llm import ask_json
    return ask_json(SYSTEM_PROMPT, f"Listing:\n{text}")


def _clean(d: dict) -> dict:
    for f in BOOL_FIELDS:
        d[f] = 1 if d.get(f) in (1, True, "1", "true") else 0
    d.setdefault("condition", "unknown")
    d.setdefault("highlights", [])
    return d


# ---------- Mock mode: keyword rules (Spanish + English) ----------
KEYWORDS = {
    "has_terrace": ["terraza", "terrace", "balcón", "balcon", "balcony"],
    "has_lift": ["ascensor", "lift", "elevator"],
    "has_air_conditioning": ["aire acondicionado", "air conditioning", "a/c"],
    "has_parking": ["garaje", "parking", "plaza de aparcamiento"],
    "has_storage_room": ["trastero", "storage room"],
    "has_built_in_wardrobe": ["armarios empotrados", "armario empotrado", "built-in wardrobe"],
    "has_pool": ["piscina", "pool"],
    "has_doorman": ["portero", "conserje", "doorman", "concierge"],
    "has_garden": ["jardín", "jardin", "garden"],
    "is_duplex": ["dúplex", "duplex"],
    "is_studio": ["estudio", "studio"],
    "is_top_floor": ["ático", "atico", "penthouse", "top floor"],
    "is_exterior": ["exterior"],
    "orientation_south": ["orientación sur", "orientacion sur", "south facing", "south-facing"],
    "orientation_north": ["orientación norte", "orientacion norte", "north facing"],
    "orientation_east": ["orientación este", "orientacion este", "east facing"],
    "orientation_west": ["orientación oeste", "orientacion oeste", "west facing"],
}


def _number(pattern, text, cast=float):
    m = re.search(pattern, text)
    return cast(m.group(1).replace(".", "").replace(",", ".")) if m else None


def _keyword_extract(text: str) -> dict:
    t = text.lower()
    d = {f: int(any(k in t for k in words)) for f, words in KEYWORDS.items()}
    d["area_m2"] = _number(r"(\d+[.,]?\d*)\s*(?:m2|m²|metros)", t)
    d["rooms"] = _number(r"(\d+)\s*(?:habitaciones|habitación|dormitorios|bedrooms?|rooms?)", t, int)
    d["baths"] = _number(r"(\d+)\s*(?:baños|baño|bathrooms?)", t, int)
    d["floor"] = _number(r"(\d+)\s*[ºª]?\s*(?:planta|floor)", t, int)
    if d["floor"] is None:
        words = {"bajo": 0, "ground": 0, "primer": 1, "first": 1, "segund": 2, "second": 2,
                 "tercer": 3, "third": 3, "cuart": 4, "fourth": 4, "quint": 5, "fifth": 5,
                 "sext": 6, "sixth": 6, "séptim": 7, "seventh": 7, "octav": 8, "eighth": 8}
        m = re.search(r"(bajo|ground|primer|first|segund|second|tercer|third|cuart|fourth|quint|fifth|"
                      r"sext|sixth|séptim|seventh|octav|eighth)\w*\s+(?:planta|floor)", t)
        d["floor"] = words[m.group(1)] if m else None
    d["year_built"] = _number(r"(?:construido en|built in|año|finca de|building from|edificio de)\s*(\d{4})", t, int)
    m = re.search(r"(\d{1,3}(?:[.,]\d{3})+|\d{5,})\s*(?:€|eur)", t)
    d["asking_price"] = float(re.sub(r"[.,]", "", m.group(1))) if m else None  # 520.000 € or 520,000 €
    if any(k in t for k in ["obra nueva", "new build", "a estrenar"]):
        d["condition"] = "new"
    elif any(k in t for k in ["a reformar", "para reformar", "needs renovation", "to renovate"]):
        d["condition"] = "to_renovate"
    elif any(k in t for k in ["reformado", "buen estado", "renovated", "good condition"]):
        d["condition"] = "good"
    else:
        d["condition"] = "unknown"
    d["address"] = None
    d["highlights"] = []
    return d
