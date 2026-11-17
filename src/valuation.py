"""Step 3 - Value it: the ML model sets the price (the LLM never does).
Step 4 - Find comparables: the most similar flats in the dataset.

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


def find_comparables(features: dict, k: int = 5) -> pd.DataFrame:
    """Nearest flats by location and size. First version of the 'comparables' step;
    can be upgraded to an embeddings/vector-database search (RAG) in Week 3."""
    data = _load()["train_data"]
    cols = ["latitude", "longitude", "area_m2", "rooms"]
    # Scale so 1 unit ~ similar importance: ~1 km in lat/lon, 20 m2, 1 room
    scale = np.array([0.009, 0.012, 20, 1])
    nn = NearestNeighbors(n_neighbors=k).fit(data[cols].values / scale)
    query = np.array([[features["latitude"], features["longitude"],
                       features.get("area_m2") or 80, features.get("rooms") or 2]]) / scale
    _, idx = nn.kneighbors(query)
    comps = data.iloc[idx[0]].copy()
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
