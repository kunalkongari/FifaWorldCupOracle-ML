# ⚽ FifaWorldCupOracle-ML (FIFA 2026 World Cup Predictor)

> **An end-to-end Machine Learning pipeline** — from raw web-scraped data to a Monte Carlo tournament simulator — built to predict the FIFA 2026 World Cup using ELO ratings, feature engineering, XGBoost, and probabilistic simulation.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/model-XGBoost-orange.svg)](https://xgboost.readthedocs.io/)
[![scikit-learn](https://img.shields.io/badge/sklearn-1.3+-yellow.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🌐 Live Dashboard
👉 [https://kunalkongari.github.io/FifaWorldCupOracle-ML/](https://kunalkongari.github.io/FifaWorldCupOracle-ML/)

## 📌 Table of Contents

1. [Project Overview](#-project-overview)
2. [Why This Project?](#-why-this-project)
3. [Pipeline Architecture](#-pipeline-architecture)
4. [Project Structure](#-project-structure)
5. [Technical Deep Dive](#-technical-deep-dive)
   - [01 · Web Scraping](#01--web-scraping)
   - [02 · Data Cleaning & EDA](#02--data-cleaning--eda)
   - [03 · ELO Ratings, Feature Engineering & XGBoost](#03--elo-ratings-feature-engineering--xgboost)
   - [04 · Monte Carlo Simulation](#04--monte-carlo-simulation)
   - [05 · Interactive Dashboard](#05--interactive-dashboard)
6. [Design Decisions & Methodology](#-design-decisions--methodology)
7. [Results](#-results)
8. [What Makes This an ML Project (Not Just a Formula)](#-what-makes-this-an-ml-project-not-just-a-formula)
9. [Limitations & Future Work](#-limitations--future-work)
10. [Quick Start](#-quick-start)
11. [Requirements](#-requirements)

---

## 🔭 Project Overview

This project builds a **complete, production-style ML pipeline** to predict the 2026 FIFA World Cup. It spans every phase of a real-world data science workflow:

| Phase | What Happens |
|---|---|
| **Data Collection** | Scrape 928 match results from Wikipedia (1930–2022) + full 2026 fixture |
| **Data Engineering** | Clean, validate, and engineer 9 features from raw match history |
| **Team Strength Modelling** | Compute dynamic ELO ratings that evolve across 22 World Cups |
| **ML Classification** | Train an XGBoost classifier to predict Home Win / Draw / Away Win |
| **Hybrid Prediction** | Blend XGBoost (70%) + Poisson model (30%) for stable probabilities |
| **Stochastic Simulation** | Run 10,000 Monte Carlo simulations of the full tournament bracket |
| **Visualisation & Dashboard** | 8+ charts + an interactive HTML dashboard of all results |

**Key result:** France leads with a **22.4% win probability**, followed by Netherlands (19.4%) and Brazil (8.4%), based on 10,000 simulations.

---

## 🎯 Why This Project?

### The ML Career Angle

Sports prediction is an ideal ML portfolio domain because it has all the properties that make machine learning genuinely hard and genuinely useful:

- **Noisy labels** — football outcomes are inherently stochastic; even the best model cannot achieve high accuracy, and understanding *why* is part of the skill.
- **Temporal structure** — you cannot use future data to predict the past. This forces proper train/test thinking and careful feature construction to avoid data leakage.
- **Imbalanced classes** — draws are rarer than wins, requiring class-weight balancing.
- **Domain knowledge integration** — raw match results alone are weak signals. Meaningful features (ELO ratings, rolling form, head-to-head) require thought.
- **End-to-end ownership** — the project covers scraping, cleaning, feature engineering, modelling, evaluation, and delivery, which mirrors real ML engineering work.

### What This Project Demonstrates

This is not a tutorial project that starts from a pre-cleaned Kaggle CSV. Every component was built from scratch:

- **Custom web scraper** targeting Wikipedia's football markup
- **Custom ELO implementation** with goal-difference multipliers and yearly decay
- **Manual feature engineering** with strict temporal ordering (no leakage)
- **Proper model evaluation** with 5-fold stratified cross-validation, a confusion matrix, and a learning curve
- **Monte Carlo simulation** as the output layer — because probabilities matter more than point predictions in tournament contexts

---

## 🏗️ Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    FIFA 2026 PREDICTOR PIPELINE                 │
└─────────────────────────────────────────────────────────────────┘

  ┌─────────────────┐
  │  01_web_scraping│  Wikipedia → raw CSVs (928 historical matches
  │  .ipynb         │             + 104 fixture rows + group tables)
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │  02_data_       │  Clean scores, fix team names, remove walkovers,
  │  cleaning.ipynb │  validate types, EDA (4 charts)
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐   ┌──────────────────────────────────────────┐
  │  03_elo_and_    │   │  ELO Ratings (dynamic, per-match update) │
  │  model.ipynb    │──►│  + 9 engineered features                 │
  │                 │   │  + XGBoost classifier (5-fold CV)        │
  └────────┬────────┘   │  + Hybrid XGBoost+Poisson predictor     │
           │            └──────────────────────────────────────────┘
           │  model.pkl / elo_dict.pkl / team_stats.pkl
           ▼
  ┌─────────────────┐
  │  04_simulate    │  10,000 × full 2026 bracket simulation
  │  .ipynb         │  → win% / final% / semi% / QF% per team
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │  05_dashboard   │  Interactive HTML dashboard (no server needed)
  │  .html          │
  └─────────────────┘
```

---

## 📁 Project Structure

```
FIFA2026_Predictor/
│
├── 01_web_scraping.ipynb       # Stage 1: Wikipedia scraper (BeautifulSoup)
├── 02_data_cleaning.ipynb      # Stage 2: Data cleaning + EDA visualisations
├── 03_elo_and_model.ipynb      # Stage 3: ELO ratings + XGBoost model
├── 04_simulate.ipynb           # Stage 4: Monte Carlo tournament simulation
├── 05_dashboard.html           # Stage 5: Interactive results dashboard
│
├── data/
│   ├── raw_historical_data.csv     # 928 raw match results (1930–2022)
│   ├── raw_fixture_2026.csv        # 104 raw 2026 fixture rows
│   ├── historical_data.csv         # Cleaned historical data
│   ├── fixture_2026.csv            # Cleaned 2026 fixture
│   └── dict_table_2026             # 12 group tables (pickle)
│
├── outputs/                    # All generated artefacts
│   ├── model.pkl               # Trained XGBoost classifier
│   ├── elo_dict.pkl            # Final ELO ratings (per team)
│   ├── team_stats.pkl          # Rolling form stats (per team)
│   ├── elo_ratings.csv         # Ranked ELO table
│   ├── feature_importances.csv # Feature importance scores
│   ├── monte_carlo_results.csv # Win/Final/Semi/QF% per team
│   ├── mc_results.json         # Same data in JSON format
│   └── *.png                   # 8 charts (ELO, features, simulation)
│
└── requirements.txt
```

> **Note:** Pre-scraped and pre-cleaned data is included in `data/`. You only need to run `01_web_scraping.ipynb` if you want to re-scrape fresh data from Wikipedia.

---

## 🔬 Technical Deep Dive

### 01 · Web Scraping

**Notebook:** `01_web_scraping.ipynb`

**Goal:** Collect all historical World Cup match results and the 2026 fixture from Wikipedia without any pre-built dataset.

#### How It Works

Wikipedia's World Cup pages use a consistent CSS class (`footballbox`) for each match result. The scraper:

1. Iterates over all 22 World Cup years (1930–2022)
2. Sends a polite HTTP `GET` with a `User-Agent` header (Wikipedia policy compliance)
3. Parses the HTML tree with `BeautifulSoup` and `lxml`, targeting `fhome` / `faway` / `fscore` classes
4. Adds a 0.5-second delay between requests to avoid hammering Wikipedia's servers

```python
YEARS = [1930, 1934, ..., 2022]

def get_matches(year: int) -> pd.DataFrame:
    url = f"https://en.wikipedia.org/wiki/{year}_FIFA_World_Cup"
    soup = BeautifulSoup(requests.get(url, headers=HEADERS).text, "lxml")
    for match in soup.find_all("div", class_="footballbox"):
        home.append(match.find("th", class_="fhome").get_text(strip=True))
        # ...
```

For the **2026 fixture**, a Wayback Machine snapshot was used (`web.archive.org`) rather than the live Wikipedia page. This guarantees **reproducibility** — the fixture locked in at the draw stage will not change as the page gets edited over time.

The 12 group tables (Groups A–L) are scraped using `pd.read_html()` on the same page, then stored as a Python dictionary of DataFrames in a pickle file (`dict_table_2026`).

#### Why BeautifulSoup + Wikipedia?

- **No API key or signup needed** — lowering the barrier to reproduce the project
- **Structured HTML** — Wikipedia's football pages are consistent enough to parse reliably
- **Transparent sourcing** — every match result can be verified by visiting the corresponding Wikipedia page

**Output:** `data/raw_historical_data.csv` (928 rows), `data/raw_fixture_2026.csv` (104 rows), `data/dict_table_2026`

---

### 02 · Data Cleaning & EDA

**Notebook:** `02_data_cleaning.ipynb`

**Goal:** Transform raw scraped strings into a clean, typed DataFrame ready for feature engineering, and produce exploratory charts that validate the data.

#### Cleaning Steps

| Step | Issue | Fix |
|---|---|---|
| Score parsing | Raw score like `"3–1 (a.e.t.)"` | Regex to strip non-digit/dash chars, then `str.split()` |
| Whitespace | Leading/trailing spaces in team names | `.str.strip()` |
| Walkover removal | Sweden vs. Austria 1938 (Austria withdrew) | Mask + drop — it is not a real match result |
| Type enforcement | Goals and year as strings | `.astype(int)` for `HomeGoals`, `AwayGoals`, `Year` |
| Null rows | Irregular Wikipedia table layouts may yield blank rows | `dropna()` before type conversion |

#### EDA Visualisations Produced

1. **Matches per World Cup year** — validates the scraper captured the right count for each tournament (group stage expansions in 1998 and 2026 are visible)
2. **Average goals per match over time** — shows the historical decline from early high-scoring tournaments toward the modern defensive era (~2.6 goals/match)
3. **Outcome distribution** — pie chart showing Home Win ≈ 54.6%, Draw ≈ 22.2%, Away Win ≈ 23.2%
4. **Top 15 goal-scoring nations (all-time)** — horizontal bar chart, Germany and Brazil lead

#### Why Do EDA Before Modelling?

Exploratory analysis is not cosmetic — it catches problems before they propagate into the model. The outcome distribution chart (54.6% home wins) directly informed the decision to use **class-weight balancing** in the XGBoost classifier, since a naive model would be biased toward predicting home wins.

**Output:** `data/historical_data.csv`, `data/fixture_2026.csv`, 4 saved PNGs

---

### 03 · ELO Ratings, Feature Engineering & XGBoost

**Notebook:** `03_elo_and_model.ipynb`

This is the core ML notebook. It has five distinct sections.

---

#### 3.1 · ELO Rating System

**What is ELO?**

ELO is a self-correcting, zero-sum strength rating system originally designed for chess. It has one critical property that makes it ideal for football: a team's rating is updated *after every match*, so it reflects current form rather than static historical averages.

**Update Rule:**

```
new_ELO = old_ELO + K × GD_multiplier × (actual_result − expected_result)
```

| Parameter | Value | Rationale |
|---|---|---|
| `K` (K-factor) | 32 | Controls the speed of learning; higher K = faster adaptation |
| `HOME_ADV` | +50 | Applied to the home team's ELO before computing expected score; reflects World Cup host advantage |
| `INITIAL_ELO` | 1500 | Universal starting point for all teams |
| `DECAY_RATE` | 0.85 | Shrinks each team's gap from 1500 by 15% each year — ensures past dominance fades |

**Goal-Difference Multiplier:**

```python
def gd_multiplier(gd: int) -> float:
    if gd <= 1:   return 1.00
    elif gd == 2: return 1.50
    elif gd == 3: return 1.75
    else:         return 2.00
```

A 5-0 win against a weak team tells us more about a team's strength than a 1-0 win against the same opponent. The multiplier encodes this — bigger wins produce bigger ELO shifts, but with diminishing returns to prevent absurd outliers.

**Why ELO Instead of Average Goals?**

Average goals per match ignores *who* you played against. A team that has beaten strong opponents has a very different true strength from a team with the same average goals that played only weak sides. ELO accounts for opponent quality at every update step.

---

#### 3.2 · Feature Engineering

Nine features are computed for every match. All are calculated strictly from **data available before that match** — ensuring zero data leakage.

| Feature | Type | Description |
|---|---|---|
| `HomeELO` | Continuous | Home team's ELO rating before kick-off |
| `AwayELO` | Continuous | Away team's ELO rating before kick-off |
| `ELO_diff` | Continuous | `HomeELO − AwayELO` — the core relative strength signal |
| `HomeFormScored` | Continuous | Home team's avg goals scored in their last 15 matches |
| `HomeFormConceded` | Continuous | Home team's avg goals conceded in their last 15 matches |
| `AwayFormScored` | Continuous | Away team's avg goals scored in their last 15 matches |
| `AwayFormConceded` | Continuous | Away team's avg goals conceded in their last 15 matches |
| `H2H_HomeWinRate` | Continuous | Historical home win rate in this specific head-to-head matchup |
| `H2H_DrawRate` | Continuous | Historical draw rate in this specific matchup |

**Why 15-match rolling window?**

World Cup qualifying and friendly schedules mean national teams play far fewer matches than club sides. A 15-match window captures approximately 18–24 months of competitive history — long enough to be statistically stable, short enough to reflect recent form rather than a team's performances from five years ago.

**On Data Leakage**

Data leakage in sports ML is a very common mistake. It occurs when features computed at prediction time accidentally incorporate information from the match being predicted. This project avoids it by computing all rolling statistics using only matches *prior to* the current row in chronological order — the feature builder processes matches in temporal sequence and updates the team history *after* each row.

---

#### 3.3 · XGBoost Classifier

**Why XGBoost?**

XGBoost (Extreme Gradient Boosting) is a tree-ensemble method that builds models sequentially, with each tree correcting the residual errors of the previous one. Key advantages for this problem:

- **Handles mixed feature scales** — no need to standardise ELO values alongside binary-like H2H rates
- **Captures non-linear interactions** — e.g., a high `ELO_diff` might only predict a win if the home team also has good recent form
- **Native multi-class support** — directly outputs probabilities for all three outcomes (Home Win / Draw / Away Win)
- **Robust to small datasets** — 928 training samples is modest; XGBoost's regularisation (min_child_weight, gamma) reduces overfitting

**Hyperparameters (chosen by reasoning, not search):**

```python
model = xgb.XGBClassifier(
    n_estimators    = 300,    # enough trees to converge without overfitting
    max_depth       = 3,      # shallow trees reduce variance on small data
    learning_rate   = 0.04,   # slow learning + more trees = stable convergence
    subsample       = 0.8,    # row subsampling adds regularisation
    colsample_bytree= 0.8,    # feature subsampling same
    min_child_weight= 3,      # minimum samples per leaf — prevents tiny noisy splits
    gamma           = 0.1,    # minimum loss reduction to make a further partition
    random_state    = 42,
)
```

**Class Imbalance Handling:**

The three classes (Home Win, Draw, Away Win) are not equally frequent — draws are rarer. If not corrected, the model simply learns to never predict draws. The fix:

```python
sample_weights = compute_sample_weight(class_weight='balanced', y=y)
model.fit(X, y, sample_weight=sample_weights)
```

This reweights each training sample so that the model penalises errors on rarer classes more heavily, producing a more calibrated classifier.

**Evaluation:**

| Metric | Value |
|---|---|
| 5-fold Stratified CV Accuracy | **51.5%** |
| Random baseline | 33.3% |
| Lift over random | **+18.2 percentage points** |

A learning curve and confusion matrix are generated to diagnose bias vs. variance. The learning curve confirms the model is not in high-variance territory (train and CV scores converge), and the confusion matrix shows the model does learn to predict draws — not just home wins.

**Why 51.5% is a Good Result**

Football is the world's most-studied sports prediction problem, and professional betting markets with billions of dollars of information integration achieve ~54–55% accuracy on match outcomes. A from-scratch model trained only on World Cup history reaching 51.5% against a 33.3% baseline is a meaningful result. The ceiling is low by design, not by error.

---

#### 3.4 · Hybrid Prediction: XGBoost + Poisson

For the actual match predictions used in simulation, a **hybrid model** is used:

```
Final probability = 0.70 × XGBoost + 0.30 × Poisson
```

**Why Blend?**

XGBoost predicts the most likely outcome well, but its raw probabilities can be extreme — e.g., 95% home win when the data contains very few similar matchups. The Poisson model anchors these predictions.

**The Poisson Component**

The Poisson distribution models the number of goals each team is expected to score. Expected goals are derived from each team's recent attacking and defensive form:

```
λ_home = HomeFormScored × AwayFormConceded
λ_away = AwayFormScored × HomeFormConceded

P(home scores x goals) = Poisson(λ_home, x)
```

By summing over all possible scorelines (0-0, 0-1, 1-0, ..., 8-8), the Poisson model produces smooth, bounded probabilities. Its 30% weight corrects the edges without dominating the ML signal.

**Output artefacts:** `outputs/model.pkl`, `outputs/elo_dict.pkl`, `outputs/team_stats.pkl`

---

### 04 · Monte Carlo Simulation

**Notebook:** `04_simulate.ipynb`

**Goal:** Rather than predicting a single deterministic bracket, simulate the entire tournament **10,000 times**, sampling match outcomes from the hybrid model's probabilities. This gives a full probability distribution over every team's path.

#### Why Monte Carlo?

A single-path prediction (e.g., "France beats England in the final") is almost certainly wrong. The correct question is: *across all possible worlds where France plays the 2026 World Cup, in what fraction do they win?* Monte Carlo simulation answers exactly this.

Each simulation:

1. Runs all 48 group-stage matches, sampling outcomes from pre-computed probabilities
2. Builds group tables by sorting on points → goal difference → goals scored
3. Generates the Round of 32, Quarter-final, Semi-final, and Final brackets according to official 2026 FIFA seeding rules
4. Tracks which team wins the tournament

After 10,000 runs, for each team:

```
Win %  = (times they won) / 10,000 × 100
Final %  = (times they appeared in the final) / 10,000 × 100
Semi %   = (times they reached the semi-finals) / 10,000 × 100
QF %     = (times they reached the quarter-finals) / 10,000 × 100
```

**Performance Optimisation — Probability Cache**

Computing hybrid probabilities for each matchup inside each simulation would make 10,000 runs prohibitively slow. Instead, all `N × (N-1)` team pair probabilities are pre-computed once and stored in a dictionary. Each simulation simply looks up the cache — reducing per-simulation overhead to pure random sampling.

```python
prob_cache = {}
for a in all_teams:
    for b in all_teams:
        get_probs(a, b, prob_cache)  # computed once

for i in range(10_000):
    simulate_once(prob_cache)  # only dict lookups inside
```

#### Simulation Visualisations

1. **Win probability bar chart** — top 15 teams ranked by tournament win %
2. **Stacked progression chart** — QF / Semi / Final / Win % side by side for top 12 teams
3. **Group heatmap** — QF probability for every team in every group, as a colour-coded matrix

**Output:** `outputs/monte_carlo_results.csv`, `outputs/mc_results.json`, 3 visualisation PNGs

---

### 05 · Interactive Dashboard

**File:** `05_dashboard.html`

A self-contained HTML file (no server required — open in any browser) that presents:

- Overall win probabilities per team
- Progression breakdown (QF / Semi / Final / Win)
- Group-by-group heatmap

Built with plain HTML + JavaScript — no Python runtime needed to view results after the pipeline has run.

---

## 🧠 Design Decisions & Methodology

### Why Not Deep Learning?

Neural networks require large datasets to outperform gradient-boosted trees. With 928 training samples (and far fewer unique team pairings), a deep model would severely overfit. XGBoost with regularisation is the standard choice for tabular data at this scale — it is the algorithm that wins Kaggle tabular competitions repeatedly for the same reason.

### Why Not a Simple Regression?

Predicting the exact score (e.g., 2-1) and deriving the outcome would be appealing, but introduces compounding errors. Predicting the *outcome class* directly (Home Win / Draw / Away Win) is simpler, more robust, and produces probabilities that are directly consumable by the simulation layer.

### Why Historical World Cup Data Only?

This is a deliberate methodological constraint. Qualification matches, friendlies, and club-level data introduce selection bias and context differences — a team might field a weakened squad in a friendly that they never would at a World Cup. Using only World Cup matches ensures all 928 training examples are from the same distributional context as the matches being predicted.

### Why 10,000 Simulations?

The law of large numbers guarantees that with 10,000 samples, the Monte Carlo estimate of each team's win probability converges to within approximately ±0.5 percentage points of the true model probability. Going higher (100,000 simulations) gives only marginal precision gains with significant runtime cost. 10,000 is the professional standard for sports tournament simulation.

### Reproducibility

All random operations use `np.random.seed(42)`. The 2026 fixture was pinned to a Wayback Machine snapshot to prevent drift. The full pipeline can be re-run from scratch — including re-scraping Wikipedia — and produce identical results.

---

## 📊 Results

Based on 10,000 simulations of the full 2026 FIFA World Cup:

| Rank | Team | 🏆 Win % | 🥈 Final % | 🥉 Semi % | 🔝 QF % |
|---|---|---|---|---|---|
| 1 | 🇫🇷 France | **22.4%** | 30.6% | 40.6% | 59.7% |
| 2 | 🇳🇱 Netherlands | **19.4%** | 29.4% | 38.3% | 65.8% |
| 3 | 🇧🇷 Brazil | **8.4%** | 15.9% | 26.4% | 42.1% |
| 4 | 🇦🇷 Argentina | **7.5%** | 15.3% | 27.9% | 44.6% |
| 5 | 🇨🇴 Colombia | **6.3%** | 14.6% | 29.1% | 49.0% |
| 6 | 🇩🇪 Germany | **5.5%** | 10.5% | 16.5% | 35.6% |
| 7 | 🇧🇪 Belgium | **4.5%** | 9.3% | 27.3% | 46.6% |
| 8 | 🏴󠁧󠁢󠁥󠁮󠁧󠁿 England | **3.9%** | 8.3% | 17.4% | 34.3% |
| 9 | 🇵🇹 Portugal | **3.5%** | 9.3% | 22.6% | 40.8% |
| 10 | 🇭🇷 Croatia | **3.3%** | 8.4% | 18.5% | 34.0% |

**Notable observations:**

- France and Netherlands together account for >40% of simulated wins — reflecting their strong ELO ratings and favourable group draws
- Colombia's 49% QF rate (rank 5 overall in QF%) is driven by a weaker group draw, not raw ELO strength
- Argentina's QF rate (44.6%) exceeds Germany's (35.6%) despite Germany's higher win %, illustrating how bracket path matters

---

## 🔍 What Makes This an ML Project (Not Just a Formula)

Many sports prediction tools are purely formulaic (e.g., "take average goals, apply Poisson"). This project makes several choices that differentiate it as a machine learning project:

| Concern | Formulaic Approach | This Project |
|---|---|---|
| Team strength | Static average goals | Dynamic ELO updated after every match |
| Prediction method | Poisson formula only | XGBoost trained on 9 features, Poisson used as a prior |
| Feature set | Goals scored/conceded | + ELO ratings + rolling form + head-to-head history |
| Class imbalance | Ignored | `compute_sample_weight('balanced')` |
| Evaluation | None | 5-fold stratified CV, confusion matrix, learning curve |
| Uncertainty | Point prediction | Full probability distributions via Monte Carlo |
| Overfitting guard | None | max_depth=3, min_child_weight, gamma, subsample |
| Reproducibility | Assumed | Fixed seeds, pinned data snapshot |

---

## ⚠️ Limitations & Future Work

### Current Limitations

**Data size:** 928 matches is small by ML standards. World Cup data is inherently limited — the tournament happens every 4 years. A future extension could incorporate qualifying matches, though domain shift would need to be handled carefully.

**Player-level information:** The model knows nothing about individual players — injuries, suspensions, or a team playing without their top striker. Squad-level features (average player ELO from club competitions, injury reports) would improve predictions.

**Draw prediction:** Draws are the hardest outcome to predict in football. Even world-class models trained on millions of club matches struggle to reach >40% accuracy on draws. This model is no exception — the confusion matrix shows draws are frequently misclassified.

**2026 format novelty:** The 2026 tournament introduces a new 48-team format with 12 groups of 4 and a Round of 32. There is no historical data on this format, so the bracket simulation relies on structural assumptions. The group-stage dynamics (point thresholds to qualify, third-place qualification rules) may differ from historical patterns.

### Future Extensions

- **Hyperparameter search** — GridSearchCV or Optuna to tune n_estimators, max_depth, and learning rate systematically
- **Player-level features** — integrate FIFA ratings or Transfermarkt valuations as additional signals
- **Calibration** — apply `CalibratedClassifierCV` to improve the quality of probability estimates
- **Ensemble models** — stack XGBoost with a logistic regression or a Random Forest for ensemble benefit
- **Qualifying data integration** — cautiously add qualifying results as auxiliary training data with a context flag
- **REST API** — wrap the `predict_match()` function in a FastAPI endpoint for real-time queries

---

## 🚀 Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/your-username/fifa2026-predictor
cd fifa2026-predictor

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run notebooks in order
#    (pre-scraped data is already in data/ — skip notebook 01 if desired)
jupyter notebook

# 4. Open the dashboard (no server needed)
open 05_dashboard.html        # macOS
xdg-open 05_dashboard.html   # Linux
start 05_dashboard.html       # Windows
```

> **Tip:** `data/` already contains pre-scraped and cleaned data. You only need to run `01_web_scraping.ipynb` if you want to re-scrape fresh data from Wikipedia.

### Run Order

| Step | Notebook | Skip if? |
|---|---|---|
| 1 | `01_web_scraping.ipynb` | `data/` already populated |
| 2 | `02_data_cleaning.ipynb` | `data/historical_data.csv` exists |
| 3 | `03_elo_and_model.ipynb` | `outputs/model.pkl` exists |
| 4 | `04_simulate.ipynb` | `outputs/monte_carlo_results.csv` exists |
| — | `05_dashboard.html` | Open directly in any browser |

---

## 📦 Requirements

```
pandas>=2.0
numpy>=1.24
scipy>=1.10
scikit-learn>=1.3
xgboost>=2.0
matplotlib>=3.7
beautifulsoup4>=4.12
requests>=2.31
lxml>=4.9
nbformat>=5.9
```

Install all dependencies:

```bash
pip install -r requirements.txt
```

**Python version:** 3.10 or higher recommended.

---

## 📄 License

MIT — free to use, modify, and distribute. Attribution appreciated.

---

## 🙏 Acknowledgements

- Match data sourced from [Wikipedia](https://en.wikipedia.org/) under the Creative Commons Attribution-ShareAlike License
- ELO methodology adapted from the [World Football Elo Ratings](https://www.eloratings.net/) system
- 2026 fixture structure based on the official [FIFA 2026 World Cup](https://www.fifa.com/fifaplus/en/tournaments/mens/worldcup/canadamexicousa2026) draw

---

*Built as part of an ML portfolio project. All predictions are probabilistic estimates based on historical data — not guarantees of tournament outcomes.*
