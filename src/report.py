"""Step 5 - Write the report: the LLM explains the model's numbers in plain language.
It receives the numbers; it is told never to change or invent them."""
from src.config import USE_LLM

SYSTEM_PROMPT = """You are ValorIA, an AI assistant for real estate agents in Madrid.
Write a short valuation report (max 250 words, Markdown) for a client.
Always write in ENGLISH, even if the listing is in Spanish (keep Spanish place names).
Use ONLY the numbers given to you; never change or invent figures.
Structure: 1) Estimated value and range, 2) What drives the price (location, size,
condition, features), 3) Comparison with similar flats, 4) Verdict on the asking
price if one is given. End with: "This is an automated pre-valuation, not an
official appraisal (tasación)." """


def eur(x):
    return f"{x:,.0f} €".replace(",", ".")


def write_report(features: dict, valuation: dict, comps, photo: dict | None) -> str:
    facts = _facts(features, valuation, comps, photo)
    if USE_LLM:
        from src.llm import ask_text
        return ask_text(SYSTEM_PROMPT, facts)
    return "### Valuation report (mock mode, no LLM)\n\n" + facts


def _facts(f, v, comps, photo) -> str:
    lines = [
        f"- Estimated value: **{eur(v['estimate'])}** (range {eur(v['low'])} – {eur(v['high'])}), "
        f"{eur(v['per_m2'])}/m²",
        f"- Flat: {f.get('area_m2')} m², {f.get('rooms')} rooms, {f.get('baths')} baths, "
        f"floor {f.get('floor')}, condition: {f.get('condition')}",
        f"- Distance to city centre: {f.get('dist_center_km', 0):.1f} km",
        f"- Similar flats nearby: median {eur(comps['price_per_m2'].median())}/m² "
        f"(n={len(comps)})",
    ]
    if f.get("highlights"):
        lines.append("- Highlights: " + "; ".join(f["highlights"]))
    if photo and photo.get("notes"):
        lines.append(f"- Photo assessment: {photo['notes']}")
    if f.get("asking_price"):
        diff = f["asking_price"] / v["estimate"] - 1
        lines.append(f"- Asking price: {eur(f['asking_price'])} ({diff:+.0%} vs. estimate)")
    return "\n".join(lines)
