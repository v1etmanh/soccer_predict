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
│   │   │                         #   ALLOWED_SIDES = {"A"} (Away-only, S6 config)
│   │   │                         #   Tự filter theo TARGET_LEAGUES từ .env
│   │   ├── odds_poller.py        # ★ LIVE POLLER — poll The Odds API mỗi 60 phút
│   │   │                         #   Dùng BankrollState (Drawdown Guard) để size stake
│   │   ├── bet_settler.py        # ★ SETTLER — settle bets + tính CLV sau trận
│   │   ├── live_signals.py       # Fetch live odds từ The Odds API
│   │   └── telegram_alert.py     # Telegram alerts (cần config BOT_TOKEN)
│   │
│   ├── betting/
│   │   ├── backtest_kelly_lm.py  # ★ Kelly sizing cho line movement backtest
│   │   ├── kelly.py              # Kelly criterion cũ (dùng bởi deprecated ML pipeline)
│   │   └── backtest.py           # Walk-forward backtest ML (deprecated)
│   │
│   ├── features/
│   │   └── features.py           # Rolling stats, Elo, H2H (deprecated — ML only)
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
│   │   ├── kelly_lm_results.csv      # Output of Kelly backtest
│   │   ├── bootstrap_ci_results.json # Bootstrap CI kết quả mới nhất
│   │   └── strategy_comparison.json  # 7-scenario comparison từ optimize_strategy.py
│   └── raw/                      # Raw CSV downloads (cache)
│
├── models/
│   ├── saved/                    # Trained XGB + LGB models (deprecated)
│   └── calibrated/               # Calibrated models (deprecated)
│
├── logs/                         # Loguru rotating log files
│   ├── odds_poller.log           # Live poller activity
│   ├── bet_settler.log           # Settlement activity
│   ├── bootstrap_ci.log
│   └── ...
│
├── OVERVIEW.md                   # Project overview + current status
├── ARCHITECTURE.md               # ← This file
├── WORKFLOW.md                   # How to run everything
│
├── run_backtest_lm.py            # Quick flat-unit backtest (line movement)
├── run_kelly_lm.py               # Full Kelly bankroll simulation
├── run_poller.py                 # ★ Live odds poller entry point
├── run_settler.py                # ★ Bet settler entry point
├── bootstrap_ci.py               # Bootstrap CI — xác nhận edge
├── optimize_strategy.py          # ★ 7-scenario CI comparison — tìm config tốt nhất
├── drawdown_guard.py             # ★ Drawdown-aware Kelly sizing (BankrollState)
├── clv_tracker.py                # CLV tracking — đánh giá execution quality
├── test_by_season.py             # ROI per season consistency check
├── test_thresholds.py            # Sweep movement thresholds
├── test_no_home.py               # Side analysis (deprecated — đã loại Home)
├── diagnose_2425.py              # Deep-dive season diagnosis
├── validate_edge.py              # Opening vs closing odds ROI comparison
├── update_closing_odds.py        # Backfill closing odds
├── migrate_closing_odds.py       # One-time DB column migration
├── check_db.py                   # Quick DB state check (snapshots, live bets)
└── run_pipeline.py               # Legacy ML pipeline (deprecated)
```

---

## Database Schema

### `matches` — Historical data (3430 rows: EPL + BL, 2020-21 → 2024-25)

```
id, league, season, date, home_team, away_team
home_goals, away_goals, result          -- H / D / A
odds_h, odds_d, odds_a                  -- Bet365 opening odds
close_h, close_d, close_a              -- Bet365 closing odds ← key for signal
pin_h/d/a, pin_close_h/d/a             -- Pinnacle odds (columns exist, not populated)
home/away_shots, corners, etc.         -- Match stats
```

### `odds_snapshots` — Live poll history

```
match_key       -- "Arsenal__Chelsea__2026-05-10" (unique per match)
league, home_team, away_team, match_date
snapshot_at     -- khi nào poll
odds_h/d/a      -- odds tại thời điểm poll
is_baseline     -- True = snapshot đầu tiên (baseline để đo movement)
```

**Current state (2026-05-04):** 11 baselines cho GW 9-10 May, 0 non-baseline.

### `live_bets` — Paper trading ledger

```
match_key, league, home_team, away_team, match_date
signal_side, movement, baseline_odds, bet_odds
stake, bankroll_before
-- Điền sau settle:
result, won, profit, bankroll_after
close_odds, clv          -- CLV = close_implied_prob - bet_implied_prob
settled, is_paper        -- is_paper=True trong paper trading phase
```

**Current state (2026-05-04):** 0 live bets — chờ signal fire.

---

## Signal Engine — How It Works

**File:** `src/signals/line_movement.py`

### Step 1 — Load matches (filter theo TARGET_LEAGUES)

```python
# Tự đọc TARGET_LEAGUES từ .env — chỉ load EPL
q = q.filter(Match.league.in_(TARGET_LEAGUES))
```

### Step 2 — Remove vig

```python
def remove_vig(h, d, a):
    ih, id_, ia = 1/h, 1/d, 1/a
    total = ih + id_ + ia
    return ih/total, id_/total, ia/total
```

### Step 3 — Compute movement

```
movement = close_implied_prob - open_implied_prob
```

### Step 4 — Apply filters (S6 config)

```python
MOVEMENT_THRESHOLD = 0.03
MIN_CLOSE_PROB     = 0.20
MAX_CLOSE_PROB     = 0.80
MIN_OPEN_ODDS      = 1.30
ALLOWED_SIDES      = {"A"}   # Away-only (S6)
```

### Step 5 — Kelly stake via Drawdown Guard

```python
guard = get_guard_state(session)          # reconstruct từ DB history
stake, debug = guard.kelly_stake(close_prob, open_odds)
# quarter Kelly * smooth_scale * tier_mult * streak_mult, cap 3%
```

---

## Config Constants (S6 Active)

| Constant | Value | File |
|----------|-------|------|
| `TARGET_LEAGUES` | `soccer_epl` | `.env` |
| `ALLOWED_SIDES` | `A` | `.env` → `line_movement.py` |
| `MOVEMENT_THRESHOLD` | `0.03` | `line_movement.py` |
| `KELLY_FRACTION` | `0.25` | `backtest_kelly_lm.py` |
| `MAX_BET_PCT` | `0.03` | `config.py` |
| `DRAWDOWN_EXPONENT` | `2.0` | `drawdown_guard.py` |
| `STOP_LOSS_STREAK` | `4` | `drawdown_guard.py` |
| `POLL_INTERVAL_MINUTES` | `60` | `config.py` |

---

## Deprecated Components

| File | Status | Why |
|------|--------|-----|
| `src/models/trainer.py` | Deprecated | ML ROI âm sau Kelly sizing |
| `src/betting/backtest.py` | Deprecated | ML backtest — bankroll → near zero |
| `src/features/features.py` | Deprecated | Features cho ML đã bỏ |
| `models/saved/*.pkl` | Deprecated | Trained model artifacts |
| `test_no_home.py` | Superseded | Đã loại Home, dùng optimize_strategy.py thay |

Active strategy **không cần ML** — chỉ cần odds data.
