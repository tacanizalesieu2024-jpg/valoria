"""The end-to-end flow: listing text + photos + location -> valuation report.
The app calls `run()`. You can also test from the terminal:
    python -m src.pipeline data/demo_listings/example_listing.txt
"""
import sys

from src.extract import extract_features
from src.features import distances
from src.report import write_report
from src.valuation import find_comparables, value_flat
from src.vision import assess_photos

CONDITION_FLAGS = {"new": (1, 0), "to_renovate": (0, 1), "good": (0, 0), "unknown": (0, 0)}


def run(listing_text: str, latitude: float, longitude: float,
        images: list[bytes] | None = None, asking_price: float | None = None) -> dict:
    features = extract_features(listing_text)                       # 1. read the text
    photo = assess_photos(images or [])                             # 2. look at photos
    if features["condition"] == "unknown" and photo and photo.get("condition"):
        features["condition"] = photo["condition"]                  #    photos fill the gap
    features["condition_new"], features["condition_to_renovate"] = CONDITION_FLAGS.get(
        features["condition"], (0, 0))
    features.update(latitude=latitude, longitude=longitude, **distances(latitude, longitude))
    if asking_price:
        features["asking_price"] = asking_price

    valuation = value_flat(features)                                # 3. ML model sets price
    comps = find_comparables(features)                              # 4. similar flats
    report = write_report(features, valuation, comps, photo)        # 5. LLM explains
    return {"features": features, "photo": photo, "valuation": valuation,
            "comparables": comps, "report": report}


if __name__ == "__main__":
    text = open(sys.argv[1], encoding="utf-8").read()
    result = run(text, latitude=40.4300, longitude=-3.7030)  # example point in Chamberí
    print(result["report"])
