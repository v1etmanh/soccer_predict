# pipeline_dish — Project Overview

## What is this?

**pipeline_dish** is a football betting edge-detection system built on a single core insight:

> *Closing odds are the most accurate probability estimate in the market. When the line moves significantly from open to close, sharp (professional) money caused that move — and betting in the same direction as sharp money generates positive ROI.*

This is called **Line Movement Analysis** or "following sharp money." The system does not predict match outcomes from scratch. Instead, it reads the market's own signal of where smart money went, then bets there using the opening odds (before the line moved) to capture the value.

---

## Core Thesis

```
Opening odds  →  Closing odds
      ↓                ↓
 Public money      Sharp money
  pushes line       corrects line

If close_implied_prob > open_implied_prob by ≥ 3%:
  → Sharps loaded this outcome
  → Bet it at the opening price (before squeeze)
  → Edge confirmed: ROI +12.4% over 5 seasons
```

The closing line is treated as ground truth. The opening line is where we get paid.

---

## Backtest Results (5 seasons, EPL + Bundesliga)

| Metric | Value |
|--------|-------|
| Seasons tested | 2020-21 → 2024-25 |
| Total bets | 332 |
| Win rate | 43.7% |
| Flat-unit ROI | **+12.4%** |
| Kelly ROI | **+75%** ($1,000 → $1,749) |
| Max drawdown | -16.7% |
| Seasons profitable | 4 / 5 |

**Key finding:** Home signals are unstable across seasons. Only Away (A) and Draw (D) signals are used. This was discovered empirically by testing each side independently across all 5 seasons.

---

## What This Is NOT

- Not a score prediction model
- Not a machine learning classifier (there is one in `src/models/` but it is deprecated in favor of line movement)
- Not a real-time trading system (not yet — live signal module is scaffolded but incomplete)

---

## Data Sources

| Source | What | How |
|--------|------|-----|
| [football-data.co.uk](https://football-data.co.uk) | Historical results + Bet365 opening & closing odds | HTTP scrape → CSV |
| The Odds API (`api.the-odds-api.com`) | Live odds for upcoming matches | REST API (key in `.env`) |

---

## Leagues Supported

| League key | Name | FD code |
|------------|------|---------|
| `soccer_epl` | English Premier League | E0 |
| `soccer_germany_bundesliga` | German Bundesliga | D1 |

EPL has stronger and more consistent ROI (+17.6% flat). Bundesliga is included but noisier (+5.5%).

---

## Tech Stack

| Layer | Tech |
|-------|------|
| Language | Python 3.13 |
| Database | SQLite via SQLAlchemy ORM |
| Data scraping | `requests` + `pandas` |
| Signal engine | Pure pandas (no ML needed) |
| Logging | `loguru` |
| Dashboard | Flask (scaffolded, not primary interface) |

---

## Who Should Read Next

- **To understand the full file structure** → read `ARCHITECTURE.md`
- **To run or modify the system** → read `WORKFLOW.md`
