"""The end-to-end flow: listing text + photos + location -> valuation report.
The app calls `run()`. You can also test from the terminal:
    python -m src.pipeline data/demo_listings/chamberi_example/listing.txt

If the LLM fails (no internet, provider overloaded), each GenAI step falls back to
offline mode and a warning is returned, so the live demo never crashes.
"""
import sys

from src import extract, report, vision
from src.features import distances
from src.valuation import fill_missing, find_comparables, value_flat

CONDITION_FLAGS = {"new": (1, 0), "to_renovate": (0, 1), "good": (0, 0), "unknown": (0, 0)}


def _safe(step_name, llm_fn, offline_fn, warnings):
    try:
        return llm_fn()
    except Exception as e:  # noqa: BLE001
        warnings.append(f"{step_name}: AI unavailable ({type(e).__name__}: {str(e)[:200]}), used offline mode.")
        return offline_fn()


def run(listing_text: str, latitude: float, longitude: float,
        images: list[bytes] | None = None, asking_price: float | None = None,
        on_step=None) -> dict:
    """on_step(message) is called before each step, so the app can show live progress."""
    step = on_step or (lambda message: None)
    warnings = []
    step("Reading the listing with Gemini…")
    features = _safe("Reading the listing",                         # 1. read the text
                     lambda: extract.extract_features(listing_text),
                     lambda: extract._clean(extract._keyword_extract(listing_text)), warnings)
    if images:
        step(f"Looking at {len(images)} photo(s)…")
    photo = _safe("Photo analysis",                                  # 2. look at photos
                  lambda: vision.assess_photos(images or []), lambda: None, warnings)
    if features["condition"] == "unknown" and photo and photo.get("condition"):
        features["condition"] = photo["condition"]                  #    photos fill the gap
    features["condition_new"], features["condition_to_renovate"] = CONDITION_FLAGS.get(
        features["condition"], (0, 0))
    features.update(latitude=latitude, longitude=longitude, **distances(latitude, longitude))
    if asking_price:
        features["asking_price"] = asking_price

    features = fill_missing(features)                              #    gaps filled from similar flats
    step("Valuing with the random forest model…")
    valuation = value_flat(features)                                # 3. ML model sets price
    step("Finding comparable flats nearby…")
    comps = find_comparables(features)                              # 4. similar flats
    step("Writing the report…")
    text = _safe("Writing the report",                              # 5. LLM explains
                 lambda: report.write_report(features, valuation, comps, photo),
                 lambda: "### Valuation report (offline)\n\n" + report._facts(features, valuation, comps, photo),
                 warnings)
    return {"features": features, "photo": photo, "valuation": valuation,
            "comparables": comps, "report": text, "warnings": warnings}


if __name__ == "__main__":
    text = open(sys.argv[1], encoding="utf-8").read()
    result = run(text, latitude=40.4300, longitude=-3.7030)  # example point in Chamberí
    print("\n".join(result["warnings"]))
    print(result["report"])
