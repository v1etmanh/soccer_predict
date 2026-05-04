# Architecture

## Directory Structure

```
pipeline_dish/
│
├── src/                          # All source modules
│   ├── config.py                 # Central config — loads .env, defines all constants
│   │
│   ├── data/
│   │   ├── scraper.py            # Downloads CSV from football-data.co.uk
│   │   └── database.py           # SQLAlchemy ORM — all table definitions + session factory
│   │
│   ├── signals/
│   │   ├── line_movement.py      # ★ CORE ENGINE — sharp money signal detection
│   │   └── live_signals.py       # Live odds fetcher (scaffolded, not active)
│   │
│   ├── betting/
│   │   ├── kelly.py              # Kelly criterion + BankrollManager (used by old ML pipeline)
│   │   ├── backtest.py           # Walk-forward backtest for ML model (deprecated)
│   │   └── backtest_kelly_lm.py  # ★ Kelly sizing for line movement signals
│   │
│   ├── features/
│   │   └── features.py           # Rolling stats, Elo, H2H (used by deprecated ML pipeline)
│   │
│   ├── models/
│   │   └── trainer.py            # XGBoost + LightGBM ensemble (deprecated)
│   │
│   └── dashboard/
│       └── app.py                # Flask dashboard (scaffolded)
│
├── data/
│   ├── pipeline.db               # SQLite database (main data store)
│   ├── processed/
│   │   ├── kelly_lm_results.csv  # Output of Kelly backtest
│   │   └── backtest_results.csv  # Output of old ML backtest
│   └── raw/                      # Raw CSV downloads (cache)
│
├── models/
│   ├── saved/                    # Trained XGB + LGB + meta models (.pkl)
│   └── calibrated/               # Isotonic-calibrated probability models (.pkl)
│
├── logs/                         # Loguru rotating log files
│
├── OVERVIEW.md                   # ← You came from here
├── ARCHITECTURE.md               # ← This file
├── WORKFLOW.md                   # How to run everything
│
├── run_backtest_lm.py            # Quick flat-unit backtest (line movement)
├── run_kelly_lm.py               # Full Kelly bankroll simulation
├── test_by_season.py             # Consistency check — ROI per season
├── test_thresholds.py            # Sweep movement thresholds 1%→5%
├── test_no_home.py               # Side analysis (H vs D vs A)
├── diagnose_2425.py              # Deep-dive on any problem season
├── update_closing_odds.py        # Backfill closing odds for existing DB rows
├── migrate_closing_odds.py       # One-time DB column migration
└── run_pipeline.py               # Legacy ML pipeline runner
```

---

## Database Schema

All tables are in `data/pipeline.db` (SQLite).

### `matches` — Primary data table

```
id            INTEGER  PK
league        TEXT     e.g. "soccer_epl"
season        TEXT     e.g. "2024-25"
date          DATETIME
home_team     TEXT
away_team     TEXT
home_goals    INTEGER
away_goals    INTEGER
result        TEXT     "H" / "D" / "A"

odds_h        REAL     Bet365 opening odds — Home
odds_d        REAL     Bet365 opening odds — Draw
odds_a        REAL     Bet365 opening odds — Away

close_h       REAL     Bet365 closing odds — Home  ← added by migration
close_d       REAL     Bet365 closing odds — Draw
close_a       REAL     Bet365 closing odds — Away

home_shots         INTEGER
away_shots         INTEGER
home_shots_target  INTEGER
away_shots_target  INTEGER
home_corners       INTEGER
away_corners       INTEGER
```

### Other tables (used by deprecated ML pipeline)

- `match_features` — Pre-computed rolling stats, Elo, H2H per match
- `predictions` — Model output: prob_home/draw/away, EV, Kelly
- `betting_records` — Historical bet ledger with P&L
- `live_odds` — Snapshot of live odds from The Odds API

---

## Signal Engine — How It Works

**File:** `src/signals/line_movement.py`

### Step 1 — Remove vig from both opening and closing odds

```python
def remove_vig(h, d, a):
    ih, id_, ia = 1/h, 1/d, 1/a
    total = ih + id_ + ia
    return ih/total, id_/total, ia/total  # true probabilities
```

### Step 2 — Compute movement per outcome

```
movement = close_implied_prob - open_implied_prob
```

Positive movement = market thinks this outcome is MORE likely at close than open = sharp money moved it.

### Step 3 — Apply filters

```python
MOVEMENT_THRESHOLD = 0.03    # ≥3% shift required
MIN_CLOSE_PROB     = 0.20    # ignore extreme longshots
MAX_CLOSE_PROB     = 0.80    # ignore near-certainties
MIN_OPEN_ODDS      = 1.30    # minimum opening price
ALLOWED_SIDES      = {"D", "A"}  # Home excluded — empirically unstable
```

### Step 4 — Select strongest signal per match

If multiple sides pass the filter, take the one with highest movement.

### Step 5 — Kelly stake sizing

```python
f_full = (b * close_prob - (1 - close_prob)) / b   # full Kelly
f_frac = f_full * 0.25                              # quarter Kelly
stake  = min(f_frac, 0.03) * bankroll              # hard cap 3%
```

`close_prob` is used as the "true probability" estimate because closing odds represent the sharpest available market consensus.

---

## Config Constants

All tunable parameters live in `src/config.py` and `src/signals/line_movement.py`:

| Constant | Value | Location | Meaning |
|----------|-------|----------|---------|
| `MOVEMENT_THRESHOLD` | 0.03 | line_movement.py | Min prob shift to trigger signal |
| `ALLOWED_SIDES` | {D, A} | line_movement.py | Home excluded |
| `MIN_CLOSE_PROB` | 0.20 | line_movement.py | Filter extreme odds |
| `MAX_CLOSE_PROB` | 0.80 | line_movement.py | Filter near-certainties |
| `KELLY_FRACTION` | 0.25 | backtest_kelly_lm.py | Fractional Kelly multiplier |
| `MAX_BET_PCT` | 0.03 | backtest_kelly_lm.py | Hard cap per bet (3% bankroll) |
| `MIN_BET_PCT` | 0.005 | backtest_kelly_lm.py | Ignore tiny Kelly bets |
| `INITIAL_BANKROLL` | 1000 | backtest_kelly_lm.py | Starting bankroll for simulation |

---

## Deprecated Components

These exist in the codebase but are **not part of the active strategy**:

| File | Status | Why deprecated |
|------|--------|----------------|
| `src/models/trainer.py` | Deprecated | ML model ROI was negative after Kelly sizing |
| `src/betting/backtest.py` | Deprecated | ML pipeline backtest — bankroll went to near zero |
| `src/features/features.py` | Deprecated | Features computed for ML model only |
| `models/saved/*.pkl` | Deprecated | Trained model artifacts, kept for reference |

The active strategy requires **no ML models** — only odds data.
