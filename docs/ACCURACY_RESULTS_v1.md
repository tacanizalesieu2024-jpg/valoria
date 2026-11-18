# GenAI accuracy test

15 real listings. Each method read ONLY the agent's description; answers were checked against the listing's facts box, which neither method saw.

| Field | Checks | Gemini (LLM) | Keyword rules |
|---|---|---|---|
| Size (m²) | 15 | 67% | 67% |
| Bedrooms | 15 | 80% | 40% |
| Bathrooms | 15 | 73% | 33% |
| Floor | 15 | 47% | 40% |
| Lift | 15 | 67% | 80% |
| Terrace | 2 | 100% | 100% |
| Exterior | 15 | 67% | 73% |
| Parking | 2 | 50% | 50% |
| Air conditioning | 12 | 42% | 50% |
| Condition | 15 | 33% | 47% |
| **All fields** | 121 | 60% | 55% |

Chart: `figures/accuracy_llm_vs_offline.png` · Every single check: `accuracy_detail.csv`
