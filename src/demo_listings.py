"""Demo listings: real listings saved once, loaded with one click in the app.

Folder format (one folder per listing) inside data/demo_listings/:

    data/demo_listings/
        chamberi_example/
            listing.txt      <- the listing text (required)
            info.json        <- optional: {"neighbourhood": "Chamberí", "address": "", "asking_price": 520000,
                                          "latitude": 40.43, "longitude": -3.70}  (exact spot, no internet needed)
            photo1.jpg       <- optional: 2-4 photos (jpg/jpeg/png/webp)
            photo2.jpg

The folder name becomes the label in the app ("chamberi_example" -> "Chamberi example").
"""
import json
from pathlib import Path

from src.config import ROOT

DEMO_DIR = ROOT / "data" / "demo_listings"
PHOTO_TYPES = {".jpg", ".jpeg", ".png", ".webp"}


def list_demos() -> dict[str, Path]:
    """Label -> folder, for every folder that has a listing.txt."""
    if not DEMO_DIR.exists():
        return {}
    folders = sorted(p for p in DEMO_DIR.iterdir() if p.is_dir() and (p / "listing.txt").exists())
    return {p.name.replace("_", " ").replace("-", " ").capitalize(): p for p in folders}


def load_demo(folder: Path) -> dict:
    """Read one demo listing: text, optional info and photos (as bytes)."""
    info_file = folder / "info.json"
    info = json.loads(info_file.read_text(encoding="utf-8")) if info_file.exists() else {}
    photos = sorted(p for p in folder.iterdir() if p.suffix.lower() in PHOTO_TYPES)
    return {
        "text": (folder / "listing.txt").read_text(encoding="utf-8").strip(),
        "neighbourhood": info.get("neighbourhood"),
        "address": info.get("address", ""),
        "asking_price": info.get("asking_price") or 0,
        "coords": (info["latitude"], info["longitude"]) if "latitude" in info and "longitude" in info else None,
        "photos": [p.read_bytes() for p in photos[:4]],
        "photo_names": [p.name for p in photos[:4]],
    }
