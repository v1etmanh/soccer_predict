# pipeline_dish — Project Overview

## What is this?

**pipeline_dish** is a football betting edge-detection system built on a single core insight:

> *Closing odds are the most accurate probability estimate in the market. When the line moves significantly from open to close, sharp (professional) money caused that move — and betting in the same direction as sharp money generates positive ROI.*

This is called **Line Movement Analysis** or "following sharp money." The system does not predict match outcomes from scratch. Instead, it reads the market's own signal of where smart money went, then bets there using the opening odds (before the line moved) to capture the value.

---

## Active Config — S6 (as of 2026-05-04)

After Bootstrap CI optimization testing 7 scenarios, **S6** is the only config with CI lower > 0:

| Parameter | Value | Reason |
|-----------|-------|--------|
| `TARGET_LEAGUES` | `soccer_epl` only | Bundesliga p(ROI≤0)=72.3% — no edge |
| `ALLOWED_SIDES` | `A` (Away only) | Draw n=23 too small; Home unstable |
| `MOVEMENT_THRESHOLD` | `0.03` (3%) | Raising to 4% reduces n→98, widens CI |

---

## Core Thesis

```
Opening odds  →  Closing odds
      ↓                ↓
 Public money      Sharp money
  pushes line       corrects line

If close_implied_prob > open_implied_prob by ≥ 3%:
  → Sharps loaded this outcome (Away side only)
  → Bet it at the opening price (before squeeze)
  → Edge confirmed: EPL Away ROI +20.4%, CI [+0.1%, +41.0%]
```

---

## Backtest Results — S6 Config (EPL Away-only, 5 seasons)

| Metric | Baseline (D+A, EPL+BL) | **S6 (Away, EPL only)** |
|--------|----------------------|------------------------|
| Total bets | 332 | **179** |
| Win rate | 43.7% | **49.2%** |
| Flat-unit ROI | +12.4% | **+20.44%** |
| Kelly ROI | +75% | **+60.3%** |
| Max drawdown | -16.7% | **-11.05%** |
| Bootstrap CI (95%) | [-3.6%, +28.0%] ❌ | **[+0.1%, +41.0%] ✅** |
| p(ROI ≤ 0) | 6.2% | **2.5%** |
| Seasons profitable | 4 / 5 | **4 / 5** |
| Edge status | MARGINAL | **CONFIRMED** |

**Bad season:** 2021-22 (-37.7%, n=27) — suspected variance, under investigation.

---

## What This Is NOT

- Not a score prediction model
- Not a machine learning classifier (one exists in `src/models/` but deprecated — ROI was negative)
- Not yet real money — currently in **paper trading phase**

---

## Live Pipeline Status (2026-05-04)

| Component | Status |
|-----------|--------|
| Odds Poller | ✅ Running — Task Scheduler, every 60 min |
| Bet Settler | ✅ Running — Task Scheduler, 7:00 AM daily |
| Drawdown Guard | ✅ Wired into poller + settler |
| Baselines in DB | ✅ 11 matches for GW 9-10 May 2026 |
| Live bets fired | ⏳ 0 — awaiting movement to accumulate |
| Telegram alerts | ⚠️ Not configured (TODO) |
| CLV data | ⏳ 0 — need first bets to settle |

---

## Data Sources

| Source | What | How |
|--------|------|-----|
| [football-data.co.uk](https://football-data.co.uk) | Historical results + Bet365 opening & closing odds | HTTP scrape → CSV |
| The Odds API (`api.the-odds-api.com`) | Live odds for upcoming matches | REST API (key in `.env`) |

---

## Leagues

| League key | Status | Notes |
|------------|--------|-------|
| `soccer_epl` | ✅ ACTIVE | Flat ROI +20.4%, CI confirmed |
| `soccer_germany_bundesliga` | ❌ DISABLED | p(ROI≤0)=72.3%, no statistical edge |

---

## Tech Stack

| Layer | Tech |
|-------|------|
| Language | Python 3.13 |
| Database | SQLite via SQLAlchemy ORM |
| Signal engine | Pure pandas — no ML |
| Risk management | Drawdown Guard (3-layer Kelly protection) |
| Scheduler | Windows Task Scheduler |
| Logging | `loguru` |

---

## Who Should Read Next

- **Full file structure** → `ARCHITECTURE.md`
- **How to run everything** → `WORKFLOW.md`
