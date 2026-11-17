"""ValorIA demo app.   Run:  streamlit run app.py"""
import altair as alt
import pandas as pd
import pydeck as pdk
import streamlit as st

from src import ui
from src.config import LLM_MODEL, MARKET_ADJUSTMENT, USE_LLM
from src.demo_listings import list_demos, load_demo
from src.pipeline import run
from src.valuation import price_drivers

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
st.markdown(ui.CSS, unsafe_allow_html=True)
st.markdown(ui.header(), unsafe_allow_html=True)

for key, default in {"listing_text": "", "hood": "Sol (centre)", "address": "", "asking": 0,
                     "demo_photos": [], "demo_choice": "—"}.items():
    st.session_state.setdefault(key, default)

DEMOS = list_demos()


def load_selected_demo():
    """Fill the form with the chosen demo listing (text, location, price, photos)."""
    choice = st.session_state.demo_choice
    if choice == "—":
        st.session_state.demo_photos = []
        return
    d = load_demo(DEMOS[choice])
    st.session_state.listing_text = d["text"]
    st.session_state.address = d["address"]
    st.session_state.asking = int(d["asking_price"])
    st.session_state.demo_photos = d["photos"]
    if d["neighbourhood"] in NEIGHBOURHOODS:
        st.session_state.hood = d["neighbourhood"]
    st.session_state.pop("result", None)


# ---------------- Sidebar ----------------
with st.sidebar:
    st.subheader("Demo listings")
    st.selectbox("Load a demo listing", ["—"] + list(DEMOS), key="demo_choice", on_change=load_selected_demo)
    st.caption(f"{len(DEMOS)} saved in data/demo_listings")
    st.divider()
    st.subheader("Status")
    st.write(f"AI: **connected** · `{LLM_MODEL}`" if USE_LLM else "AI: **offline mode** (no API key in .env)")
    st.write(f"Prices: 2018 data × {MARKET_ADJUSTMENT}")
    st.divider()
    st.caption("**How it works:** Gemini reads the listing and photos → a random forest trained on "
               "75,000 Madrid flats sets the price → Gemini explains it. The AI never invents the price.")

# ---------------- Input ----------------
left, right = st.columns([3, 2], gap="large")
with left:
    st.text_area("Listing text (Spanish or English)", key="listing_text", height=190,
                 placeholder="Paste a listing, or pick one in 'Load a demo listing' on the left")
    uploads = st.file_uploader("Photos (optional: kitchen, bathroom, living room)",
                               type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    if st.session_state.demo_photos and not uploads:
        st.caption(f"Using {len(st.session_state.demo_photos)} photo(s) from the demo listing")
        st.image(st.session_state.demo_photos, width=110)
with right:
    st.text_input("Address (optional)", key="address", placeholder="e.g. Calle de Fuencarral 100, Madrid")
    st.selectbox("…or neighbourhood", list(NEIGHBOURHOODS), key="hood")
    st.number_input("Asking price in € (optional)", min_value=0, step=5000, key="asking")
    go = st.button("Value this flat", type="primary", use_container_width=True)


def locate(address: str, hood: str):
    if address:
        try:
            from geopy.geocoders import Nominatim
            loc = Nominatim(user_agent="valoria-student-project").geocode(address, timeout=10)
            if loc:
                return loc.latitude, loc.longitude, "address"
        except Exception:  # noqa: BLE001
            pass
        st.warning("Address not found, using the neighbourhood instead.")
    return (*NEIGHBOURHOODS[hood], "neighbourhood")


if go:
    text = st.session_state.listing_text
    if not text.strip():
        st.error("Paste a listing first, or load a demo listing from the sidebar.")
        st.stop()
    lat, lon, source = locate(st.session_state.address, st.session_state.hood)
    images = [p.getvalue() for p in uploads] if uploads else st.session_state.demo_photos
    with st.status("Valuing the flat…", expanded=True) as status:
        r = run(text, lat, lon, images=images, asking_price=st.session_state.asking or None,
                on_step=lambda message: st.write(message))
        status.update(label="Valuation ready", state="complete", expanded=False)
    # Keep the result, so clicking tabs or buttons doesn't erase it
    st.session_state.result = {**r, "lat": lat, "lon": lon, "images": images, "source": source}

# ---------------- Results ----------------
if "result" in st.session_state:
    r = st.session_state.result
    v, f = r["valuation"], r["features"]
    for w in r["warnings"]:
        st.warning(w, icon="⚠️")

    st.markdown(ui.price_card(v, f.get("asking_price")), unsafe_allow_html=True)
    st.markdown(ui.facts_row(f), unsafe_allow_html=True)
    st.markdown(ui.chips(f), unsafe_allow_html=True)

    t_report, t_photos, t_comps, t_model, t_raw = st.tabs(
        ["📝 Report", "📷 Photos", "📍 Comparables", "📊 Price drivers", "🔎 Extracted data"])

    with t_report:
        st.markdown(r["report"])
        st.download_button("Download report", r["report"], file_name="valoria_report.md")

    with t_photos:
        if r["images"]:
            st.image(r["images"], width=230)
            p = r["photo"] or {}
            if p.get("condition"):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Condition", str(p["condition"]).replace("_", " ").capitalize())
                for col, key, label in [(c2, "kitchen_score", "Kitchen"), (c3, "bathroom_score", "Bathroom"),
                                        (c4, "finishes_score", "Finishes")]:
                    col.metric(label, f"{p[key]}/5" if p.get(key) else "–")
            if p.get("notes"):
                st.caption(f"AI appraiser's note: {p['notes']}")
        else:
            st.info("No photos for this listing. Add 2–4 photos (kitchen, bathroom, living room) "
                    "and Gemini rates the flat's condition from them.")

    with t_comps:
        comps = r["comparables"].reset_index(drop=True)
        points = pd.concat([
            pd.DataFrame({"latitude": comps["latitude"], "longitude": comps["longitude"], "kind": "Similar flat",
                          "label": [f"{ui.eur(p)} · {a:.0f} m²" for p, a in zip(comps["price"], comps["area_m2"])],
                          "color": [[29, 78, 216, 210]] * len(comps)}),
            pd.DataFrame([{"latitude": r["lat"], "longitude": r["lon"], "kind": "This flat",
                           "label": f"Estimate {ui.eur(v['estimate'])}", "color": [14, 124, 102, 255]}]),
        ])
        layer = pdk.Layer("ScatterplotLayer", data=points, get_position="[longitude, latitude]",
                          get_fill_color="color", get_radius=40, radius_min_pixels=8, pickable=True)
        st.pydeck_chart(pdk.Deck(layers=[layer], map_style="light",
                                 initial_view_state=pdk.ViewState(latitude=r["lat"], longitude=r["lon"], zoom=14),
                                 tooltip={"text": "{kind}\n{label}"}))
        st.caption(f"Green = this flat (location from {r['source']}). Blue = the 5 most similar flats by "
                   "location and size in the training data.")
        table = pd.DataFrame({
            "Size": comps["area_m2"].map(lambda x: f"{x:.0f} m²"),
            "Bedrooms": comps["rooms"].astype(int), "Bathrooms": comps["baths"].astype(int),
            "Lift": comps["has_lift"].map({1: "✓", 0: ""}), "Terrace": comps["has_terrace"].map({1: "✓", 0: ""}),
            "Price": comps["price"].map(ui.eur), "€/m²": comps["price_per_m2"].map(ui.eur)})
        st.dataframe(table, hide_index=True, use_container_width=True)

    with t_model:
        drivers = price_drivers()
        chart = alt.Chart(drivers).mark_bar(color=ui.BRAND, cornerRadiusEnd=4).encode(
            x=alt.X("importance:Q", title="Share of the model's decisions", axis=alt.Axis(format="%")),
            y=alt.Y("driver:N", sort="-x", title=None),
            tooltip=[alt.Tooltip("driver:N", title="Driver"), alt.Tooltip("importance:Q", format=".1%")],
        ).properties(height=300)
        st.altair_chart(chart, use_container_width=True)
        st.caption("What the random forest relies on most across all 75,000 flats (feature importance). "
                   "Size and location dominate; features the AI reads from the text fine-tune the price.")

    with t_raw:
        st.caption("Everything the AI extracted from the listing, as structured data for the model.")
        st.json({k: val for k, val in f.items() if val not in (0, None, [], "")})
