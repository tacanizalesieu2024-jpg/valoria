# GenAI accuracy test: can the LLM read a listing?

**Question.** Our price model needs structured facts (m², bedrooms, floor, lift…). Agents write free text. How well does Gemini turn that text into facts, compared with simple keyword rules (our offline fallback)?

**Method.** 15 real Madrid listings (Idealista, Nov 2026; 8 English, 7 Spanish descriptions). Each method read ONLY the agent's description. Answers were checked against the listing's facts box ("Características"), which neither method saw. Facts the box didn't state were skipped. 121 checks in total. Run: `python -m src.accuracy_test v2`

## Results

| Field | Checks | Gemini v1 | **Gemini v2** | Keyword rules |
|---|---|---|---|---|
| Size (m²) | 15 | 67% | **80%** | 67% |
| Bedrooms | 15 | 80% | **93%** | 40% |
| Bathrooms | 15 | 73% | **100%** | 33% |
| Floor | 15 | 47% | **60%** | 40% |
| Lift | 15 | 67% | **80%** | 80% |
| Exterior | 15 | 67% | **80%** | 73% |
| Air conditioning | 12 | 42% | **50%** | 50% |
| Condition | 15 | 33% | **93%** | 47% |
| Terrace / Parking | 2 + 2 | 100% / 50% | **100% / 100%** | 100% / 50% |
| **All fields** | **121** | **60%** | **81%** | **55%** |

Charts: `figures/accuracy_llm_vs_offline_v1.png`, `figures/accuracy_llm_vs_offline_v2.png` · every check: `accuracy_detail_v1.csv`, `accuracy_detail_v2.csv`

## What changed between v1 and v2
1. **Condition definition.** v1 called renovated flats "new". In Idealista and in our training data, "new" means a new-build development; a renovated flat is "good". This was also a valuation bug (it switched on the model's new-build flag). The prompt now defines the three categories.
2. **Bathroom counting.** Idealista counts a guest toilet ("aseo") as a bathroom. The prompt now says so.
3. **Timeouts.** In v1, 2 listings timed out and were scored as all wrong. v2 retries and falls back to a backup model.

**Caveat:** the v2 prompt was improved after looking at these same 15 listings, so v2 is partly tuned on its own test set. A fully fair number needs a fresh set of listings.

## Where Gemini v2 is still "wrong" (23 checks)
Reviewing each one: almost none are reading mistakes.
- **Fact not in the description** (most cases): floor, air conditioning and lift are often only in the facts box; three descriptions never state the m².
- **The listing contradicts itself:** one text says 3 bedrooms (facts box: 4); one says "interior" (facts box: exterior).
- **Judgement calls:** a text says the flat "requires updating" while the facts box says "good condition".

**Conclusion.** When the information is in the text, the LLM extracts it reliably and handles what keyword rules cannot: counting ("two en-suite bathrooms and a guest toilet" = 3), implied facts ("a spacious en-suite bedroom" = 1 bedroom), both languages. Its limit is what the agent writes, which is why ValorIA fills missing details from similar flats in our dataset and marks them as *estimated*.
