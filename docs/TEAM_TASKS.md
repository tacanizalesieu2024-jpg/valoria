# First tasks per role

Everyone: follow the Setup in `README.md` and get the app running in mock mode.

**Deadline: final presentation on Dec 4.** Code freeze: Nov 27.

## 1. Business lead
- [ ] Final company name, mission and vision (draft in the group proposal doc)
- [ ] Find a sourced Madrid house price change from 2018 to today, and set `MARKET_ADJUSTMENT`
- [ ] Market facts for the business case: number of agencies in Madrid, what competitors (Idealista and Fotocasa valuations) offer
- [ ] Slide outline for the 10-minute presentation

## 2. Data & price model (`src/features.py`, `src/train_model.py`)
- [ ] Explore the dataset: price distribution, prices by area, missing values
- [ ] Try to improve the random forest (tune `n_estimators`, `min_samples_leaf`; try gradient boosting)
- [ ] Charts for the slides: actual vs. predicted, feature importance
- [ ] Document the cleaning decisions (duplicates, outliers, log price)

## 3. LLM pipeline (`src/llm.py`, `src/extract.py`, `src/report.py`)
- [ ] Get the API key, **set a spending limit**, and share it privately (never on GitHub)
- [ ] Test extraction on 10 real listings: which fields does it get wrong? Improve the prompt
- [ ] Measure it: % of fields correct for the LLM vs. mock (keyword) mode. Good for the documentation
- [ ] Improve the report prompt (tone, length, Spanish or English option)

## 4. Vision & comparables (`src/vision.py`, `src/valuation.py`)
- [ ] Collect 10-20 real Madrid listings (text + 3-4 photos each) into `data/demo_listings/`
- [ ] Test the photo rating: is it consistent if you run it twice?
- [ ] Optional upgrade: comparables via embeddings + a vector database (Chroma or FAISS) = RAG

## 5. App & integration (`app.py`)
- [ ] Create the GitHub repo, push this code, add everyone as collaborators
- [ ] Run the app with the API key and fix anything that breaks
- [ ] Prepare 2 demo cases (a fairly priced flat and an overpriced one)
- [ ] Record a backup video of the demo in case the internet fails
