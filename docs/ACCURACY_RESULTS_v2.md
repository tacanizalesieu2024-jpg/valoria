# GenAI accuracy test

15 real listings. Each method read ONLY the agent's description; answers were checked against the listing's facts box, which neither method saw.

| Field | Checks | Gemini (LLM) | Keyword rules |
|---|---|---|---|
| Size (m²) | 15 | 80% | 67% |
| Bedrooms | 15 | 93% | 40% |
| Bathrooms | 15 | 100% | 33% |
| Floor | 15 | 60% | 40% |
| Lift | 15 | 80% | 80% |
| Terrace | 2 | 100% | 100% |
| Exterior | 15 | 80% | 73% |
| Parking | 2 | 100% | 50% |
| Air conditioning | 12 | 50% | 50% |
| Condition | 15 | 93% | 47% |
| **All fields** | 121 | 81% | 55% |

Chart: `figures/accuracy_llm_vs_offline.png` · Every single check: `accuracy_detail.csv`
