"""Central settings for ValorIA. Change values here, not inside the other files."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

DATA_PATH = ROOT / "data" / "madrid_sale.csv.gz"
POIS_PATH = ROOT / "data" / "madrid_pois.json"
MODEL_PATH = ROOT / "models" / "price_model.joblib"

# --- LLM settings (read from the .env file) ---
LLM_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")  # must support images for the photo step
USE_LLM = bool(LLM_API_KEY)  # no key -> the app runs in offline "mock" mode

# --- Market adjustment ---
# The training data is from 2018. Multiply predictions by this factor to bring
# them to today's prices. ASSUMPTION: 1.0 = no adjustment. The business lead
# should replace it with a sourced figure (e.g. an official Madrid house price
# index, 2018 -> today) and cite the source in the documentation.
MARKET_ADJUSTMENT = float(os.getenv("MARKET_ADJUSTMENT", "1.0"))
