# Tasks and deadlines

**Final presentation: Fri 4 Dec 2026.** Code freeze: Fri 27 Nov.
The same list, with a Status column, is in `docs/ValorIA_group_plan.xlsx`.

## Milestones

| Date | Milestone | Done when |
|---|---|---|
| Thu 19 Nov | Everyone set up | Roles confirmed, repo access, app runs on every laptop, API key ready, demo listings collected |
| Sun 22 Nov | Core working with real LLM | Extraction + report work with the API key; model improved; business case facts in |
| Fri 27 Nov | Code freeze | Full demo works end to end. After this only bug fixes, docs and slides |
| Mon 30 Nov | Docs + slides draft | Every role has written its documentation section; full deck drafted |
| Wed 02 Dec | Rehearsal + backup | Timed rehearsal; backup demo video recorded |
| Fri 04 Dec | Final presentation | 10 minutes + live demo |

## 1. Business lead

- [ ] **Thu 19 Nov**: Final company name, mission, vision and target customer
- [ ] **Sun 22 Nov**: Market facts: Madrid agencies, what Idealista/Fotocasa valuations offer; sourced 2018→today price change for MARKET_ADJUSTMENT
- [ ] **Thu 26 Nov**: Slide outline for the 10-minute presentation
- [ ] **Mon 30 Nov**: Full slide deck (business case, product, results, demo slot)
- [ ] **Wed 02 Dec**: Organise the two rehearsals

## 2. Data & price model

- [ ] **Thu 19 Nov**: Run the setup; explore the data (price distribution, prices by area, missing values)
- [ ] **Sun 22 Nov**: Improve the model: tune the random forest, try gradient boosting, keep the best
- [ ] **Fri 27 Nov**: Charts for the slides: actual vs. predicted, feature importance
- [ ] **Mon 30 Nov**: Write the documentation section on data cleaning and the model

## 3. LLM pipeline

- [ ] **Thu 19 Nov**: Get the API key, set a spending limit, add it to .env and share it privately
- [ ] **Sun 22 Nov**: Test extraction on 10 real listings and improve the prompt
- [ ] **Fri 27 Nov**: Measure accuracy: % of fields correct, LLM vs. mock mode
- [ ] **Mon 30 Nov**: Polish the report prompt; write the LLM documentation section

## 4. Vision & comparables

- [ ] **Thu 19 Nov**: Collect 10-20 real Madrid listings (text + 3-4 photos each) into data/demo_listings
- [ ] **Tue 24 Nov**: Test the photo rating: same result when run twice?
- [ ] **Fri 27 Nov**: Optional: comparables with embeddings + Chroma (RAG). Cut it if not working by this date
- [ ] **Mon 30 Nov**: Write the documentation section on vision and comparables

## 5. App & integration

- [ ] **Thu 19 Nov**: Invite everyone to the repo; check everyone can run the app
- [ ] **Sun 22 Nov**: Run the app with the real API key and fix what breaks
- [ ] **Fri 27 Nov**: CODE FREEZE: full flow works end to end; pick 2 demo cases (fair price + overpriced)
- [ ] **Tue 01 Dec**: Final README with run instructions; record a backup video of the demo

## Everyone

- [ ] **Wed 02 Dec**: Rehearsal 1 (full presentation + live demo, timed to 10 min)
- [ ] **Thu 03 Dec**: Final rehearsal
- [ ] **Fri 04 Dec**: FINAL PRESENTATION
- [ ] **Before deadline**: Fill in the anonymous peer evaluation (missing it = 0/10 individual grade)

Setup for everyone: follow `README.md` and get the app running.
