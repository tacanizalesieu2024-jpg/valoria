"""Visual building blocks for the demo app (plain HTML/CSS/SVG, no extra libraries).
The Madrid skyline is an original drawing in SVG, so it works offline and has no copyright issues."""
from html import escape

BRAND = "#0E7C66"       # teal: ValorIA
WARM = "#E07A5F"        # terracotta: Madrid rooftops
INK = "#1B2733"
MUTED = "#6B7280"


def _html(s: str) -> str:
    """Remove blank lines and indentation, so Streamlit's Markdown never breaks the HTML."""
    return "".join(line.strip() for line in s.splitlines() if line.strip())


CSS = _html(f"""
<style>
.block-container {{padding-top: 3.2rem; max-width: 1200px;}}
.vx-card {{background:white; border:1px solid #ECE6DD; border-radius:18px; padding:20px 24px;
          margin:8px 0 14px 0; box-shadow:0 2px 12px rgba(27,39,51,0.05);}}
.vx-label {{color:{MUTED}; font-size:0.8rem; text-transform:uppercase; letter-spacing:0.8px; font-weight:600;}}
.vx-price {{font-size:2.8rem; font-weight:800; color:{INK}; line-height:1.15; letter-spacing:-1px;}}
.vx-small {{color:{MUTED}; font-size:0.95rem;}}
.vx-badge {{display:inline-block; padding:7px 16px; border-radius:999px; font-weight:700; font-size:0.95rem;}}
.vx-chip {{display:inline-block; padding:6px 13px; margin:3px 5px 3px 0; border-radius:999px;
          background:white; border:1px solid #E5DED3; font-size:0.9rem; color:{INK};}}
.vx-chip-hl {{background:{BRAND}12; border-color:{BRAND}66; color:{BRAND}; font-weight:600;}}
.vx-chip-off {{color:#A8A29E; text-decoration:line-through; border-style:dashed; background:transparent;}}
.vx-facts {{display:grid; grid-template-columns:repeat(6,1fr); gap:10px;}}
.vx-fact {{text-align:center; padding:10px 4px; border-radius:14px; background:#FAF7F2;}}
.vx-fact i {{display:block; font-style:normal; font-size:1.4rem; margin-bottom:2px;}}
.vx-fact b {{display:block; font-size:1.3rem; color:{INK};}}
.vx-fact span {{color:{MUTED}; font-size:0.8rem;}}
.vx-pill {{display:inline-block; padding:5px 12px; margin:0 6px 6px 0; border-radius:999px; font-size:0.85rem;
          background:rgba(255,255,255,0.18); border:1px solid rgba(255,255,255,0.35); color:white;
          backdrop-filter:blur(4px);}}
</style>
""")

# Original stylised skyline: Palacio Real · Edificio España · Metrópolis · Puerta de Alcalá ·
# KIO towers · Cuatro Torres. Coordinates: 1200 x 170, ground at y=170.
SKYLINE = """
<svg viewBox="0 0 1200 170" preserveAspectRatio="xMidYMax slice" style="position:absolute; left:0; bottom:0; width:100%; height:50%;">
<g fill="#2B3A4A" opacity="0.45">
<rect x="0" y="120" width="40" height="50"/><rect x="200" y="110" width="30" height="60"/>
<rect x="300" y="100" width="45" height="70"/><rect x="345" y="125" width="30" height="45"/>
<rect x="470" y="105" width="40" height="65"/><rect x="515" y="118" width="35" height="52"/>
<rect x="720" y="112" width="45" height="58"/><rect x="905" y="100" width="40" height="70"/>
<rect x="1150" y="95" width="50" height="75"/>
</g>
<g fill="#15202C">
<rect x="30" y="135" width="175" height="35"/><rect x="92" y="120" width="52" height="50"/>
<path d="M104 120 Q118 96 132 120 Z"/><rect x="116" y="92" width="4" height="8"/>
<rect x="232" y="72" width="62" height="98"/><rect x="246" y="60" width="34" height="14"/>
<rect x="258" y="50" width="10" height="12"/>
<rect x="380" y="98" width="72" height="72"/><path d="M380 98 Q416 52 452 98 Z"/>
<rect x="412" y="58" width="8" height="12"/><circle cx="416" cy="52" r="5"/><rect x="414" y="40" width="4" height="10"/>
<path fill-rule="evenodd" d="M560 170 V126 H712 V170 Z M603 170 V150 A8 8 0 0 1 619 150 V170 Z M628 170 V148 A8 8 0 0 1 644 148 V170 Z M653 170 V150 A8 8 0 0 1 669 150 V170 Z M572 170 V150 H584 V170 Z M688 170 V150 H700 V170 Z"/>
<rect x="598" y="116" width="76" height="11"/><path d="M606 116 L636 98 L666 116 Z"/>
<polygon points="776,170 796,170 828,64 808,64"/><polygon points="884,170 904,170 872,64 852,64"/>
<rect x="955" y="28" width="30" height="142"/>
<path fill-rule="evenodd" d="M1000 170 V14 H1042 V170 Z M1012 150 V28 H1030 V150 Z"/>
<rect x="1060" y="40" width="32" height="130"/><ellipse cx="1076" cy="40" rx="16" ry="9"/>
<rect x="1108" y="8" width="30" height="162"/><rect x="1112" y="2" width="22" height="8"/>
</g>
</svg>"""


def eur(x: float) -> str:
    return f"{x:,.0f} €".replace(",", ".")


def header() -> str:
    return _html(f"""
<div style="position:relative; height:285px; border-radius:22px; overflow:hidden; margin-bottom:22px;
            background:linear-gradient(180deg, #1E3A5F 0%, #7A4E6E 45%, {WARM} 75%, #F2C38B 100%);">
  <div style="position:absolute; left:57%; top:46%; width:120px; height:120px; border-radius:50%;
              background:radial-gradient(circle, #FBE3A6 0%, #F6C26B 55%, rgba(246,194,107,0) 72%);"></div>
  {SKYLINE}
  <div style="position:absolute; left:30px; top:24px; right:30px;">
    <div style="display:flex; align-items:center; gap:12px;">
      <div style="width:44px; height:44px; border-radius:12px; background:white; color:{BRAND};
                  display:flex; align-items:center; justify-content:center; font-size:24px; font-weight:800;">⌂</div>
      <div style="font-size:2.3rem; font-weight:800; color:white; letter-spacing:-0.5px;
                  text-shadow:0 2px 10px rgba(0,0,0,0.25);">ValorIA</div>
    </div>
    <div style="color:#FFF4E8; font-size:1.05rem; margin:6px 0 12px 2px; max-width:780px;
                text-shadow:0 1px 6px rgba(0,0,0,0.3);">
      AI pre-valuation for Madrid flats: reads the listing, looks at the photos, explains the price.</div>
    <span class="vx-pill">🏙️ 75,000 Madrid flats</span><span class="vx-pill">🌲 Random forest model</span><span class="vx-pill">✨ Gemini AI</span>
  </div>
</div>""")


def verdict(asking: float | None, estimate: float) -> tuple[str, str]:
    """(text, colour) for the asking price vs. our estimate. ±5% counts as fair."""
    if not asking:
        return "No asking price given", MUTED
    diff = asking / estimate - 1
    if diff > 0.05:
        return f"Above market · {diff:+.0%}", "#C2410C"
    if diff < -0.05:
        return f"Below market · {diff:+.0%} · possible deal", "#1D4ED8"
    return f"Fair price · {diff:+.0%}", BRAND


def price_card(v: dict, asking: float | None) -> str:
    """Big estimate + a gauge showing the range, the estimate and the asking price."""
    low, est, high = v["low"], v["estimate"], v["high"]
    lo = min(low, asking or low) * 0.93
    hi = max(high, asking or high) * 1.07
    pos = lambda x: max(0.0, min(100.0, (x - lo) / (hi - lo) * 100))  # noqa: E731
    text, colour = verdict(asking, est)
    asking_marker = ""
    if asking:
        asking_marker = f"""
      <div style="position:absolute; left:{pos(asking):.1f}%; top:-6px; height:40px; border-left:3px dashed {INK};"></div>
      <div style="position:absolute; left:{pos(asking):.1f}%; top:40px; transform:translateX(-50%);
                  font-size:0.8rem; white-space:nowrap; color:{INK};"><b>Asking</b> {eur(asking)}</div>"""
    return _html(f"""
<div class="vx-card" style="border-left:6px solid {colour};">
  <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:10px;">
    <div>
      <div class="vx-label">Estimated value</div>
      <div class="vx-price">{eur(est)}</div>
      <div class="vx-small">{eur(v['per_m2'])}/m² · likely range {eur(low)} – {eur(high)}</div>
    </div>
    <span class="vx-badge" style="background:{colour}1A; color:{colour}; border:1px solid {colour}55;">{escape(text)}</span>
  </div>
  <div style="position:relative; height:{70 if asking else 34}px; margin:26px 8px 4px 8px;">
    <div style="position:absolute; left:0; right:0; top:12px; height:8px; border-radius:4px;
                background:linear-gradient(90deg, #DCEBE6, #F3E3D8);"></div>
    <div style="position:absolute; left:{pos(low):.1f}%; width:{pos(high) - pos(low):.1f}%; top:8px; height:16px;
                border-radius:8px; background:linear-gradient(90deg, {BRAND}40, {BRAND}70); border:1px solid {BRAND}88;"></div>
    <div style="position:absolute; left:{pos(est):.1f}%; top:4px; width:24px; height:24px; margin-left:-12px;
                border-radius:50%; background:{BRAND}; border:3px solid white; box-shadow:0 1px 5px rgba(0,0,0,0.25);"></div>
    <div style="position:absolute; left:{pos(est):.1f}%; top:-18px; transform:translateX(-50%);
                font-size:0.8rem; font-weight:700; color:{BRAND}; white-space:nowrap;">Our estimate</div>
    {asking_marker}
  </div>
</div>""")


FACTS = [("area_m2", "m²", "📐"), ("rooms", "bedrooms", "🛏️"), ("baths", "bathrooms", "🛁"),
         ("floor", "floor", "🏢"), ("year_built", "built", "🏗️"), ("condition", "condition", "✨")]

CHIPS = [("has_lift", "Lift"), ("has_terrace", "Terrace"), ("is_exterior", "Exterior"),
         ("has_air_conditioning", "Air con"), ("has_parking", "Parking"), ("has_storage_room", "Storage room"),
         ("has_built_in_wardrobe", "Built-in wardrobes"), ("has_doorman", "Doorman"), ("has_pool", "Pool"),
         ("has_garden", "Garden"), ("orientation_south", "South facing"), ("is_top_floor", "Top floor")]


def facts_row(f: dict) -> str:
    cells = []
    for key, label, icon in FACTS:
        val = f.get(key)
        if val in (None, "", "unknown"):
            val = "–"
        elif isinstance(val, float) and val.is_integer():
            val = int(val)
        if key == "condition" and val != "–":
            val = str(val).replace("_", " ").capitalize()
        cells.append(f'<div class="vx-fact"><i>{icon}</i><b>{escape(str(val))}</b><span>{label}</span></div>')
    return _html('<div class="vx-card"><div class="vx-facts">' + "".join(cells) + "</div></div>")


def chips(f: dict) -> str:
    hl = [str(h) for h in (f.get("highlights") or [])[:4]]
    said = " ".join(hl).lower()
    on = [f'<span class="vx-chip">✓ {label}</span>' for key, label in CHIPS
          if f.get(key) and label.lower() not in said]  # don't repeat what a highlight already says
    off = [f'<span class="vx-chip vx-chip-off">{label}</span>' for key, label in CHIPS if not f.get(key)][:4]
    highlights = "".join(f'<span class="vx-chip vx-chip-hl">★ {escape(h)}</span>' for h in hl)
    return _html(f'<div style="margin:4px 0 14px 0;">{highlights}{"".join(on)}{"".join(off)}</div>')
