"""Feature engineering shared by training and the app.

The SAME function (`build_features`) is used for the training data and for a
new listing, so the model always sees columns in the same format.
"""
import json

import numpy as np
import pandas as pd

from src.config import POIS_PATH

# Features the price model uses. Every one of them can be filled in for a NEW
# listing: by the LLM (from the text), the vision model (condition) or the
# address (location). That is why cadastral columns are left out.
FEATURES = [
    "area_m2", "rooms", "baths", "floor", "year_built",
    "has_terrace", "has_lift", "has_air_conditioning", "has_parking",
    "has_storage_room", "has_built_in_wardrobe", "has_pool", "has_doorman",
    "has_garden", "is_duplex", "is_studio", "is_top_floor", "is_exterior",
    "orientation_south", "orientation_north", "orientation_east", "orientation_west",
    "condition_new", "condition_to_renovate",  # baseline = second hand, good condition
    "latitude", "longitude", "dist_center_km", "dist_metro_km", "dist_castellana_km",
]

# Raw idealista18 column -> our readable name
RENAME = {
    "CONSTRUCTEDAREA": "area_m2", "ROOMNUMBER": "rooms", "BATHNUMBER": "baths",
    "FLOORCLEAN": "floor", "HASTERRACE": "has_terrace", "HASLIFT": "has_lift",
    "HASAIRCONDITIONING": "has_air_conditioning", "HASPARKINGSPACE": "has_parking",
    "HASBOXROOM": "has_storage_room", "HASWARDROBE": "has_built_in_wardrobe",
    "HASSWIMMINGPOOL": "has_pool", "HASDOORMAN": "has_doorman", "HASGARDEN": "has_garden",
    "ISDUPLEX": "is_duplex", "ISSTUDIO": "is_studio", "ISINTOPFLOOR": "is_top_floor",
    "HASSOUTHORIENTATION": "orientation_south", "HASNORTHORIENTATION": "orientation_north",
    "HASEASTORIENTATION": "orientation_east", "HASWESTORIENTATION": "orientation_west",
    "BUILTTYPEID_1": "condition_new", "BUILTTYPEID_2": "condition_to_renovate",
    "LATITUDE": "latitude", "LONGITUDE": "longitude", "PRICE": "price",
    "DISTANCE_TO_CITY_CENTER": "dist_center_km", "DISTANCE_TO_METRO": "dist_metro_km",
    "DISTANCE_TO_CASTELLANA": "dist_castellana_km",
}

_POIS = json.loads(POIS_PATH.read_text())


def _haversine_km(lat, lon, lats, lons):
    """Distance in km from one point to an array of points."""
    lat, lon, lats, lons = map(np.radians, (lat, lon, np.asarray(lats), np.asarray(lons)))
    a = np.sin((lats - lat) / 2) ** 2 + np.cos(lat) * np.cos(lats) * np.sin((lons - lon) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def distances(lat: float, lon: float) -> dict:
    """Distances used by the model, computed from coordinates (same POIs as the dataset)."""
    c_lon, c_lat = _POIS["City_Center"][0][0], _POIS["City_Center"][1][0]
    m_lon, m_lat = _POIS["Metro"]
    k_lon, k_lat = _POIS["Castellana"]
    return {
        "dist_center_km": float(_haversine_km(lat, lon, [c_lat], [c_lon])[0]),
        "dist_metro_km": float(_haversine_km(lat, lon, m_lat, m_lon).min()),
        "dist_castellana_km": float(_haversine_km(lat, lon, k_lat, k_lon).min()),
    }


def clean_training_data(raw: pd.DataFrame) -> pd.DataFrame:
    """idealista18 raw table -> one clean row per flat with our feature names."""
    df = raw.sort_values("PERIOD").drop_duplicates("ASSETID", keep="last")  # same flat in several quarters
    df = df.assign(
        year_built=df["CADCONSTRUCTIONYEAR"].fillna(df["CONSTRUCTIONYEAR"]),
        is_exterior=(df["FLATLOCATIONID"] == 1).astype(int),
    ).rename(columns=RENAME)
    df["floor"] = df["floor"].fillna(df["floor"].median())
    df["year_built"] = df["year_built"].fillna(df["year_built"].median())
    # Remove extreme outliers that would distort the model
    df = df[(df["area_m2"].between(20, 500)) & (df["price"].between(30_000, 5_000_000))]
    return df[["ASSETID"] + FEATURES + ["price"]].reset_index(drop=True)


def listing_to_row(features: dict) -> pd.DataFrame:
    """A new listing (dict from the LLM + location) -> one-row DataFrame for the model."""
    row = {f: features.get(f) for f in FEATURES}
    row.update(distances(row["latitude"], row["longitude"]))
    defaults = {"floor": 1, "year_built": 1975, "baths": 1}
    for f in FEATURES:
        if row[f] is None:
            row[f] = defaults.get(f, 0)
    return pd.DataFrame([row])[FEATURES].astype(float)
