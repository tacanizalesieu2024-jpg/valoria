# ValorIA – AI property valuation for Madrid

ValorIA reads a property listing (text + photos), turns it into data with GenAI,
values it with a machine learning model trained on ~75,000 Madrid flats, and
writes an explained valuation report for real estate agents.

**Key design choice:** the ML model calculates the price, and the LLM only reads
the listing and explains the result. LLMs are unreliable with precise numbers.

## How it works

| Step | File | What it does | Technique |
|---|---|---|---|
| 1. Read the listing | `src/extract.py` | Listing text → structured features (JSON) | LLM extraction |
| 2. Look at the photos | `src/vision.py` | Photos → condition rating | Vision LLM |
| 3. Value it | `src/valuation.py` | Features → price + range | Random forest (Session 8) |
| 4. Find comparables | `src/valuation.py` | 5 most similar flats nearby | Nearest neighbours (→ RAG upgrade) |
| 5. Write the report | `src/report.py` | Numbers → client-ready explanation | LLM generation |

`app.py` is the Streamlit demo: a guided 5-step flow (details → photos → AI analysis → valuation → report) with source labels on every fact and model-measured price drivers. `app_classic.py` is the earlier single-page version.

## Setup (about 10 minutes, everyone does this once)

You need Python 3.10 or newer.

```bash
# 1. Get the code (after one person has pushed it to GitHub)
git clone <your-repo-url>
cd valoria

# 2. Create a virtual environment and install the libraries
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Add the API key (free: aistudio.google.com -> Get API key)
cp .env.example .env               # Windows: copy .env.example .env
#    then open .env and paste the key after LLM_API_KEY=

# 4. Train the price model (about 20-60 seconds)
python -m src.train_model

# 5. Run the app
streamlit run app.py
```

**Which LLM?** By default the project uses Google Gemini through its free tier
(OpenAI-compatible endpoint, settings in `.env`). Note: on the free tier, Google may
use what you send to improve its products, so only send public listing data.
Switching to OpenAI (paid) only needs a different key, base URL and model in `.env`.

**No API key yet?** Everything still runs: the app switches to *mock mode*,
using simple keyword rules instead of the LLM. This mode is also the backup if the
internet fails during the live demo.

## First results (random forest vs. baseline)

Running `python -m src.train_model` on a 20% test set gives:

| Model | Mean absolute error | Mean % error |
|---|---|---|
| Linear regression (baseline) | ~101,600 € | 25.3% |
| Random forest | ~46,800 € | 13.0% |

The strongest price drivers are size, location (latitude, distance to Castellana
and the centre), number of bathrooms and having a lift.

## Data

- **Training data:** `data/madrid_sale.csv.gz`, Madrid sale listings from the
  [idealista18](https://github.com/paezha/idealista18) dataset (Idealista, 2018;
  94,815 rows, 75,607 unique flats after cleaning). License: Open Database
  License (ODbL 1.0). Please keep this attribution in the documentation.
- **Points of interest:** `data/madrid_pois.json` (city centre, 240 metro
  entrances, Paseo de la Castellana) from the same package, used to compute distances.
- **Demo listings:** put the 10-20 real listings you collect in
  `data/demo_listings/` (one `.txt` per listing, plus its photos).

**Known limitation:** prices are from 2018. `MARKET_ADJUSTMENT` in `.env` scales them
to today. Set it from a sourced price index and cite the source.

## Project structure

```
valoria/
├── app.py                  Streamlit demo
├── requirements.txt
├── .env.example            copy to .env, add your key (never commit .env)
├── data/
│   ├── madrid_sale.csv.gz  training data (idealista18)
│   ├── madrid_pois.json    centre, metro, Castellana coordinates
│   └── demo_listings/      listings for the live demo
├── docs/TEAM_TASKS.md      first tasks per role
├── models/                 created by train_model.py (not on GitHub)
└── src/
    ├── config.py           settings
    ├── features.py         feature list + distance calculations
    ├── train_model.py      trains and compares the models
    ├── llm.py              the only file that talks to the LLM API
    ├── extract.py          step 1
    ├── vision.py           step 2
    ├── valuation.py        steps 3 and 4
    ├── report.py           step 5
    └── pipeline.py         runs all steps
```
