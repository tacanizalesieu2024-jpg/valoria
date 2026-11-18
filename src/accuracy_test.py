"""GenAI accuracy test: how well does Gemini read a listing, compared with simple keyword rules?
Run:  python -m src.accuracy_test

Input: data/accuracy_test/accuracy_listings.xlsx
  - listing_text = ONLY the description written by the agent ("Comentario del anunciante")
  - the other columns = the TRUE facts from the listing's facts box ("Características"),
    which the AI never sees. Leave a cell empty if the facts box doesn't say it.

For every listing and field, both methods extract the value from the text only, and we
check it against the facts box. Output: docs/accuracy_results.md / .csv and a chart.
"""
import time

import pandas as pd

from src.config import ROOT, USE_LLM
from src.extract import _clean, _keyword_extract, _llm_extract

INPUT = ROOT / "data" / "accuracy_test" / "accuracy_listings.xlsx"
OUT = ROOT / "docs"
OUT_SUFFIX = ""
NUMERIC = ["area_m2", "rooms", "baths", "floor"]
BOOLEAN = ["has_lift", "has_terrace", "is_exterior", "has_parking", "has_air_conditioning"]
FIELDS = NUMERIC + BOOLEAN + ["condition"]
LABELS = {"area_m2": "Size (m²)", "rooms": "Bedrooms", "baths": "Bathrooms", "floor": "Floor",
          "has_lift": "Lift", "has_terrace": "Terrace", "is_exterior": "Exterior", "has_parking": "Parking",
          "has_air_conditioning": "Air conditioning", "condition": "Condition"}
YES = {"1", "yes", "y", "si", "sí", "true", "x"}
NO = {"0", "no", "n", "false"}


def _truth(value, field):
    """Read a cell from the template. Returns None when the facts box didn't say."""
    if pd.isna(value) or str(value).strip() == "":
        return None
    v = str(value).strip().lower()
    if field in BOOLEAN:
        return 1 if v in YES else 0 if v in NO else None
    if field == "condition":
        return {"buen estado": "good", "a reformar": "to_renovate", "obra nueva": "new"}.get(v, v)
    if field == "floor" and v in ("bajo", "ground", "0"):
        return 0
    try:
        return float(v.replace(",", "."))
    except ValueError:
        return None


def _correct(pred, truth, field):
    if pred is None or pred == "":
        return False
    if field == "area_m2":
        return abs(float(pred) - truth) <= max(2, 0.03 * truth)  # ±3% or ±2 m²
    if field in NUMERIC:
        return float(pred) == truth
    if field == "condition":
        return str(pred) == truth
    return int(pred) == truth


def run_test():
    df = pd.read_excel(INPUT, sheet_name="Listings")
    df = df[df["listing_text"].notna() & (df["listing_text"].astype(str).str.strip() != "")]
    if df.empty:
        raise SystemExit("No listings yet: fill in data/accuracy_test/accuracy_listings.xlsx")
    if not USE_LLM:
        raise SystemExit("No API key in .env: the LLM side of the test needs Gemini.")
    rows = []
    for i, r in df.reset_index(drop=True).iterrows():
        text = str(r["listing_text"])
        offline = _clean(_keyword_extract(text))
        llm = {}
        for attempt in range(3):  # a timeout is a network problem, not a reading mistake: retry
            try:
                llm = _clean(_llm_extract(text))
                break
            except Exception as e:  # noqa: BLE001
                print(f"  listing {i + 1}: LLM failed ({type(e).__name__}), retry {attempt + 1}/2")
                time.sleep(10)
        else:
            print(f"  listing {i + 1}: LLM failed 3 times, counted as wrong")
        for field in FIELDS:
            truth = _truth(r.get(field), field)
            if truth is None:
                continue
            rows.append({"listing": r.get("id", i + 1), "field": field, "truth": truth,
                         "llm": llm.get(field), "offline": offline.get(field),
                         "llm_correct": _correct(llm.get(field), truth, field),
                         "offline_correct": _correct(offline.get(field), truth, field)})
        print(f"  listing {i + 1}/{len(df)} done")
        time.sleep(4)  # stay inside the free-tier rate limit
    return pd.DataFrame(rows), len(df)


def summarise(detail: pd.DataFrame) -> pd.DataFrame:
    per_field = detail.groupby("field").agg(checks=("truth", "size"), llm=("llm_correct", "mean"),
                                            offline=("offline_correct", "mean"))
    per_field = per_field.reindex([f for f in FIELDS if f in per_field.index])
    total = pd.DataFrame({"checks": [len(detail)], "llm": [detail["llm_correct"].mean()],
                          "offline": [detail["offline_correct"].mean()]}, index=["ALL FIELDS"])
    return pd.concat([per_field, total])


def chart(summary: pd.DataFrame):
    import matplotlib.pyplot as plt
    import numpy as np
    teal, terracotta, ink, muted = "#00957A", "#C9573A", "#1B2733", "#6B7280"
    s = summary.drop(index="ALL FIELDS")
    names = [LABELS[f] for f in s.index]
    y = np.arange(len(s))
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(s) + 2.4), dpi=200)
    ax.barh(y - 0.19, s["llm"] * 100, height=0.36, color=teal, label="Gemini (LLM)")
    ax.barh(y + 0.19, s["offline"] * 100, height=0.36, color=terracotta, label="Keyword rules (offline)")
    for i, (a, b) in enumerate(zip(s["llm"], s["offline"])):
        ax.text(a * 100 + 1, i - 0.19, f"{a:.0%}", va="center", fontsize=8, color=ink)
        ax.text(b * 100 + 1, i + 0.19, f"{b:.0%}", va="center", fontsize=8, color=ink)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlim(0, 112)
    ax.set_xlabel("Fields read correctly from the description (%)", color=muted, fontsize=10)
    ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2,
              labelcolor=ink)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#D9DDE1")
    ax.tick_params(colors=muted, labelsize=9)
    tot = summary.loc["ALL FIELDS"]
    ax.set_title(f"Gemini reads {tot['llm']:.0%} of listing details correctly vs. {tot['offline']:.0%} "
                 "for keyword rules", loc="left", fontsize=12, fontweight="bold", color=ink, pad=22)
    ax.text(0, 1.02, f"{int(tot['checks'])} checks against the listings' facts boxes", transform=ax.transAxes,
            fontsize=9, color=muted)
    fig.tight_layout()
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "figures" / f"accuracy_llm_vs_offline{OUT_SUFFIX}.png", facecolor="white")
    plt.close(fig)


def main():
    import sys
    tag = sys.argv[1] if len(sys.argv) > 1 else ""   # e.g. "v2" -> files named accuracy_results_v2.csv
    global OUT_SUFFIX
    OUT_SUFFIX = f"_{tag}" if tag else ""
    print(f"Reading {INPUT.name}")
    detail, n = run_test()
    summary = summarise(detail)
    detail.to_csv(OUT / f"accuracy_detail{OUT_SUFFIX}.csv", index=False)
    summary.to_csv(OUT / f"accuracy_results{OUT_SUFFIX}.csv")
    table = ["| Field | Checks | Gemini (LLM) | Keyword rules |", "|---|---|---|---|"]
    for f, r in summary.iterrows():
        name = "**All fields**" if f == "ALL FIELDS" else LABELS[f]
        table.append(f"| {name} | {int(r['checks'])} | {r['llm']:.0%} | {r['offline']:.0%} |")
    md = (f"# GenAI accuracy test\n\n{n} real listings. Each method read ONLY the agent's description; "
          f"answers were checked against the listing's facts box, which neither method saw.\n\n"
          + "\n".join(table) + "\n\nChart: `figures/accuracy_llm_vs_offline.png` · "
          "Every single check: `accuracy_detail.csv`\n")
    (OUT / f"ACCURACY_RESULTS{OUT_SUFFIX}.md").write_text(md, encoding="utf-8")
    chart(summary)
    print("\n" + "\n".join(table))
    print(f"\nSaved docs/ACCURACY_RESULTS{OUT_SUFFIX}.md, accuracy_results{OUT_SUFFIX}.csv, accuracy_detail{OUT_SUFFIX}.csv and the chart")


if __name__ == "__main__":
    main()
