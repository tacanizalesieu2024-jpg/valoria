"""ValorIA demo app.   Run:  streamlit run app.py"""
import pandas as pd
import streamlit as st

from src.config import MARKET_ADJUSTMENT, USE_LLM
from src.pipeline import run
from src.report import eur

# Approximate neighbourhood centres, used when no exact address is geocoded
NEIGHBOURHOODS = {
    "Sol (centre)": (40.4168, -3.7038), "Chamberí": (40.4340, -3.7040),
    "Salamanca": (40.4300, -3.6780), "Retiro": (40.4110, -3.6760),
    "Chamartín": (40.4580, -3.6770), "Malasaña": (40.4260, -3.7050),
    "Lavapiés": (40.4085, -3.7010), "Argüelles": (40.4300, -3.7180),
    "Tetuán": (40.4600, -3.6980), "Arganzuela": (40.3980, -3.6990),
    "Carabanchel": (40.3830, -3.7280), "Puente de Vallecas": (40.3910, -3.6580),
}

st.set_page_config(page_title="ValorIA", page_icon="🏠", layout="wide")
st.title("ValorIA – AI property valuation for Madrid")
st.caption("Paste a listing, add photos, and get an explained pre-valuation in seconds.")

with st.sidebar:
    st.subheader("Status")
    st.write("LLM: **connected**" if USE_LLM else "LLM: **mock mode** (no API key in .env)")
    st.write(f"Market adjustment vs. 2018 data: ×{MARKET_ADJUSTMENT}")

left, right = st.columns([1, 1])
with left:
    text = st.text_area("1. Listing text (Spanish or English)", height=220)
    photos = st.file_uploader("2. Photos (optional)", type=["jpg", "jpeg", "png"],
                              accept_multiple_files=True)
with right:
    address = st.text_input("3. Address (optional, e.g. 'Calle de Fuencarral 100, Madrid')")
    hood = st.selectbox("…or pick a neighbourhood", list(NEIGHBOURHOODS))
    asking = st.number_input("Asking price in € (optional)", min_value=0, step=5000)
    go = st.button("Value this flat", type="primary", use_container_width=True)


def locate(address: str, hood: str):
    if address:
        try:
            from geopy.geocoders import Nominatim
            loc = Nominatim(user_agent="valoria-student-project").geocode(address, timeout=10)
            if loc:
                return loc.latitude, loc.longitude, "address"
        except Exception:
            pass
        st.warning("Address not found, using the neighbourhood instead.")
    return (*NEIGHBOURHOODS[hood], "neighbourhood")


if go:
    if not text.strip():
        st.error("Paste a listing first.")
        st.stop()
    lat, lon, source = locate(address, hood)
    with st.spinner("Reading the listing, checking photos and valuing…"):
        r = run(text, lat, lon, images=[p.getvalue() for p in photos or []],
                asking_price=asking or None)
    v, f = r["valuation"], r["features"]

    st.divider()
    c1, c2, c3 = st.columns(3)
    c1.metric("Estimated value", eur(v["estimate"]))
    c2.metric("Range (10th–90th pct.)", f"{eur(v['low'])} – {eur(v['high'])}")
    if f.get("asking_price"):
        diff = f["asking_price"] / v["estimate"] - 1
        c3.metric("Asking price vs. estimate", eur(f["asking_price"]), f"{diff:+.0%}",
                  delta_color="inverse")
    else:
        c3.metric("Price per m²", eur(v["per_m2"]))

    tab_report, tab_features, tab_comps = st.tabs(["Report", "What the AI extracted", "Comparables"])
    with tab_report:
        st.markdown(r["report"])
        st.download_button("Download report", r["report"], file_name="valoria_report.md")
    with tab_features:
        st.caption(f"Location from: {source}")
        shown = {k: val for k, val in f.items() if val not in (0, None, [], "")}
        st.json(shown)
        if r["photo"]:
            st.subheader("Photo assessment")
            st.json(r["photo"])
    with tab_comps:
        comps = r["comparables"]
        st.dataframe(comps.drop(columns=["latitude", "longitude"]).round(0), use_container_width=True)
        points = pd.concat([comps[["latitude", "longitude"]],
                            pd.DataFrame({"latitude": [lat], "longitude": [lon]})])
        st.map(points)
