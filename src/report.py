"""Step 5 - Write the report: the LLM explains the model's numbers in plain language.
It receives the numbers; it is told never to change or invent them."""
from src.config import USE_LLM

SYSTEM_PROMPT = """You are ValorIA, an AI assistant for real estate agents in Madrid.
Write a short valuation report (max 250 words, Markdown) for a client.
Always write in ENGLISH, even if the listing is in Spanish (keep Spanish place names).

Strict rules:
- Use ONLY the numbers given to you; never change or invent figures.
- The price comes from a machine learning model. In section 2, explain the price ONLY with
  the factors listed under "Model inputs" and "Main price drivers". Never say the price is
  higher or lower because of renovation dates, decor, light, views or anything else the model
  does not use.
- EVERYTHING listed under "Model inputs" (size, bedrooms, bathrooms, floor, year, condition and
  every feature such as lift, terrace, parking, top floor, exterior) IS used by the model; the
  "Main price drivers" are simply the strongest of them. Location and neighbourhood are also used,
  through the exact coordinates and distances. Never describe any of these as "not reflected".
- Only the photo observations are NOT used by the model. You may mention them as extra context,
  clearly separated (e.g. "Not reflected in the estimate: ..."). Do not repeat model inputs there.
- Copy all numbers exactly as written, including their format (e.g. 1.585.151 €).
- If some details were estimated from similar flats, say so in one short sentence.

Structure: 1) Estimated value and range, 2) What drives the price, 3) Comparison with similar
flats, 4) Verdict on the asking price if one is given. End with: "This is an automated
pre-valuation, not an official appraisal (tasación)." """

DRIVER_NAMES = "size (m²), bathrooms, location, lift, distance to Paseo de la Castellana, distance to the centre"


def eur(x):
    return f"{x:,.0f} €".replace(",", ".")


def write_report(features: dict, valuation: dict, comps, photo: dict | None) -> str:
    facts = _facts(features, valuation, comps, photo)
    if USE_LLM:
        from src.llm import ask_text
        return ask_text(SYSTEM_PROMPT, facts)
    return "### Valuation report (mock mode, no LLM)\n\n" + facts


def _facts(f, v, comps, photo) -> str:
    yes = [label for key, label in [("has_lift", "lift"), ("has_terrace", "terrace"), ("is_exterior", "exterior"),
                                    ("has_parking", "parking"), ("has_air_conditioning", "air conditioning"),
                                    ("has_doorman", "doorman"), ("has_pool", "pool"), ("is_top_floor", "top floor")]
           if f.get(key)]
    lines = [
        f"- Estimated value: **{eur(v['estimate'])}** (range {eur(v['low'])} – {eur(v['high'])}), "
        f"{eur(v['per_m2'])}/m²",
        f"- Model inputs (all used by the model): {f.get('area_m2')} m², {f.get('rooms')} bedrooms, {f.get('baths')} bathrooms, "
        f"floor {f.get('floor')}, built {f.get('year_built')}, condition: {f.get('condition')}, "
        f"features: {', '.join(yes) or 'none listed'}, {f.get('dist_center_km', 0):.1f} km from the centre",
        f"- Main price drivers in the model (most important first): {DRIVER_NAMES}",
        f"- Similar flats in our dataset (similarity search): median {eur(comps['price_per_m2'].median())}/m² "
        f"(n={len(comps)})",
    ]
    if f.get("estimated"):
        lines.append("- Not in the listing, estimated from similar flats in our dataset: "
                     + ", ".join(f"{k.replace('_', ' ')} = {f.get(k)}" for k in f["estimated"]))
    # Listing highlights are NOT passed on: they often repeat model inputs (parking, terrace...)
    # and the AI then wrongly calls them "not reflected". They are shown as chips in the app instead.
    if photo and photo.get("notes"):
        lines.append(f"- Photo assessment (context only): {photo['notes']}")
    if f.get("asking_price"):
        diff = f["asking_price"] / v["estimate"] - 1
        lines.append(f"- Asking price: {eur(f['asking_price'])} ({diff:+.0%} vs. estimate)")
    return "\n".join(lines)
