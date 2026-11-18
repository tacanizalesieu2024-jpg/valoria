"""Step 3 - Value it: the ML model sets the price (the LLM never does).
Step 4 - Find comparables: a similarity search for the most similar flats in the dataset.
(Also: missing details are filled in from similar flats in the dataset.)

Price range: a random forest is many trees; each tree gives its own estimate.
The spread between them (10th to 90th percentile) is our "confidence range".
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from src.config import MARKET_ADJUSTMENT, MODEL_PATH
from src.features import listing_to_row

_bundle = None


def _load():
    global _bundle
    if _bundle is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError("No trained model yet. Run:  python -m src.train_model")
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


def value_flat(features: dict) -> dict:
    b = _load()
    X = listing_to_row(features)
    per_tree = np.exp([tree.predict(X.values)[0] for tree in b["model"].estimators_])
    estimate = float(np.exp(b["model"].predict(X)[0])) * MARKET_ADJUSTMENT
    low, high = (np.percentile(per_tree, [10, 90]) * MARKET_ADJUSTMENT).tolist()
    return {"estimate": estimate, "low": low, "high": high,
            "per_m2": estimate / max(features.get("area_m2") or 1, 1)}


# Similarity search: which columns count, and how much a difference "costs".
# 1 unit ~ 1 km of distance, 20 m2 of size, 1 bedroom, or half a yes/no feature (lift, exterior, condition).
SIM_COLS = ["latitude", "longitude", "area_m2", "rooms", "has_lift", "is_exterior",
            "condition_new", "condition_to_renovate"]
SIM_SCALE = np.array([0.009, 0.012, 20, 1, 0.5, 0.5, 0.5, 0.5])
_nn = None


def _neighbours(features: dict, k: int, cols=SIM_COLS, scale=SIM_SCALE) -> pd.DataFrame:
    global _nn
    data = _load()["train_data"]
    if _nn is None or _nn[0] != tuple(cols):
        _nn = (tuple(cols), NearestNeighbors().fit(data[cols].values / scale))
    defaults = {"area_m2": 80, "rooms": 2}
    query = np.array([[features.get(c) if features.get(c) is not None else defaults.get(c, 0)
                       for c in cols]], dtype=float) / scale
    _, idx = _nn[1].kneighbors(query, n_neighbors=k)
    return data.iloc[idx[0]]


def fill_missing(features: dict, k: int = 15) -> dict:
    """Details the listing doesn't mention (bathrooms, year built, floor) are filled with the
    median of the 15 most similar flats in OUR dataset, instead of a fixed guess.
    Returns the updated features; the filled keys are listed in features['estimated']."""
    missing = [c for c in ("baths", "year_built", "floor") if features.get(c) in (None, "", 0)
               and not (c == "floor" and features.get(c) == 0)]
    if not missing:
        features["estimated"] = []
        return features
    similar = _neighbours(features, k, cols=SIM_COLS[:4], scale=SIM_SCALE[:4])
    for c in missing:
        features[c] = int(round(similar[c].median()))
    features["estimated"] = missing
    return features


def find_comparables(features: dict, k: int = 5) -> pd.DataFrame:
    """Similarity search over our dataset: the k flats closest in location, size, bedrooms,
    lift, exterior/interior and condition. (A nearest-neighbour search, not an LLM/RAG step.)"""
    comps = _neighbours(features, k).copy()
    comps["price"] = comps["price"] * MARKET_ADJUSTMENT
    comps["price_per_m2"] = comps["price"] / comps["area_m2"]
    return comps[["area_m2", "rooms", "baths", "has_lift", "has_terrace",
                  "price", "price_per_m2", "latitude", "longitude"]]


READABLE = {
    "area_m2": "Size (m²)", "latitude": "Location (north–south)", "longitude": "Location (east–west)",
    "baths": "Bathrooms", "has_lift": "Lift", "dist_castellana_km": "Distance to Castellana",
    "dist_center_km": "Distance to centre", "dist_metro_km": "Distance to metro", "rooms": "Bedrooms",
    "year_built": "Year built", "floor": "Floor", "has_doorman": "Doorman", "has_terrace": "Terrace",
    "has_parking": "Parking", "has_pool": "Pool", "is_exterior": "Exterior", "condition_new": "New build",
    "condition_to_renovate": "Needs renovation", "has_air_conditioning": "Air conditioning",
}


def price_drivers(top: int = 8) -> pd.DataFrame:
    """What the random forest relies on most (feature importance), with readable names."""
    b = _load()
    imp = pd.Series(b["model"].feature_importances_, index=b["features"]).sort_values(ascending=False).head(top)
    return pd.DataFrame({"driver": [READABLE.get(f, f.replace("_", " ").capitalize()) for f in imp.index],
                         "importance": imp.values})
