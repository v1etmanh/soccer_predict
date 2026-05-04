# Workflow — How to Run Everything

## Prerequisites

```bash
pip install sqlalchemy loguru requests pandas python-dotenv xgboost lightgbm scikit-learn flask
```

Create `.env` in project root:

```env
ODDS_API_KEY=your_key_here
DB_PATH=data/pipeline.db
TARGET_LEAGUES=soccer_epl,soccer_germany_bundesliga
INITIAL_BANKROLL=1000
```

---

## First-Time Setup (run once)

### 1. Initialize database + scrape historical data

```powershell
# Creates all DB tables and downloads 5 seasons of data
python -c "import sys; sys.path.insert(0,'src'); from data.database import init_db; init_db()"
python -c "import sys; sys.path.insert(0,'src'); from data.scraper import run_scraper; run_scraper()"
```

Expected output: ~3430 matches across EPL + Bundesliga (2020-21 to 2024-25).

### 2. Add closing odds columns (one-time migration)

```powershell
python migrate_closing_odds.py
```

### 3. Backfill closing odds for all existing matches

```powershell
python update_closing_odds.py
```

Expected output: `3430 trận có closing odds mới`

---

## Daily Testing Commands

These are the 3 commands you will use regularly:

### Quick signal check (< 1 second)

```powershell
python run_backtest_lm.py
```

Shows: total bets, win rate, ROI, breakdown by league. Use this to verify the engine is working after any code change.

**Expected output:**
```
[LineMovement] Loaded 3430 trận, 332 có sharp signal
[Backtest] 332 bets | Win: 43.7% | ROI: 12.42% | Profit: 41.23u
```

### Season-by-season consistency check

```powershell
python test_by_season.py
```

Shows ROI per season + league x season matrix. Use this to detect if edge is holding or degrading. **This is the most important health check** — if 3+ seasons flip to LOSS, the strategy needs re-evaluation.

**Healthy output pattern:**
```
2020-21   PROFIT
2021-22   PROFIT
2022-23   LOSS or ~break-even   ← acceptable
2023-24   PROFIT
2024-25   PROFIT
```

### Kelly bankroll simulation

```powershell
python run_kelly_lm.py
```

Shows: final bankroll, total profit in $, max drawdown, avg stake. Also saves full bet-by-bet log to `data/processed/kelly_lm_results.csv`.

**Expected output:**
```
[Kelly Backtest] 270 bets | Win: 38.9% | ROI: 74.95% | Bankroll: 1000.0 -> 1749.53 | MaxDD: -16.71%
```

---

## Tuning Commands

### Sweep movement thresholds

```powershell
python test_thresholds.py
```

Tests thresholds from 1% to 5%. Use when deciding whether to tighten or loosen the signal filter. Current setting is 3%.

| Threshold | Signal count | ROI |
|-----------|-------------|-----|
| 3% | 332 | +12.4% ← current |
| 4% | 183 | +13.7% |
| 5% | 94 | +21.9% (low volume) |

### Diagnose a specific season

```powershell
python diagnose_2425.py
```

Currently configured for 2024-25. Edit the season filter inside to diagnose any season. Shows: movement size, ROI by side (H/D/A), cumulative profit by month.

### Test without Home signals

```powershell
python test_no_home.py
```

Compares All signals vs No-Home vs Away-only. Use when re-evaluating whether to re-include Home signals. As of the last test, Home signals hurt ROI in 2024-25 and are excluded.

---

## Updating Data (ongoing)

To refresh data at the start of a new season or mid-season:

```powershell
# Re-scrape new matches
python -c "import sys; sys.path.insert(0,'src'); from data.scraper import run_scraper; run_scraper()"

# Backfill any new closing odds
python update_closing_odds.py

# Verify
python run_backtest_lm.py
```

---

## Understanding a Signal (Manual Walkthrough)

Given a match: **Arsenal vs Chelsea**, odds open at H:2.10 / D:3.40 / A:3.60

After removing vig:
```
open_prob_H = 0.452,  open_prob_D = 0.280,  open_prob_A = 0.268
```

Closing odds shift to H:2.30 / D:3.20 / A:3.20

After removing vig:
```
close_prob_H = 0.411,  close_prob_D = 0.296,  close_prob_A = 0.293
```

Movement:
```
move_H = 0.411 - 0.452 = -0.041  (sharp money AGAINST Home)
move_D = 0.296 - 0.280 = +0.016  (weak movement, below 3% threshold)
move_A = 0.293 - 0.268 = +0.025  (below threshold)
```

No signal fires — movement on D and A is under 3%, and H moved negatively.

---

If instead closing odds were H:2.50 / D:3.10 / A:2.80:
```
close_prob_A = 0.340
move_A = 0.340 - 0.268 = +0.072  → ABOVE 3% threshold ✓
```

Signal fires: **BET AWAY at opening odds 3.60**

Kelly stake: `f = (2.60 * 0.340 - 0.660) / 2.60 = 0.087` → quarter Kelly = 2.2% of bankroll → capped at 3% → stake = $30 on $1,000 bankroll.

---

## What To Do If ROI Drops

Run this diagnostic sequence in order:

```powershell
# 1. Check if a specific side is causing losses
python test_no_home.py

# 2. Check if a specific season/league is the culprit
python test_by_season.py

# 3. Check if threshold needs adjustment
python test_thresholds.py

# 4. Deep-dive into the problem season
python diagnose_2425.py   # edit season inside file
```

Most likely causes and fixes:

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| Home signals suddenly bad | Market learned home bias | Already excluded, verify ALLOWED_SIDES |
| Single league dragging ROI | Bundesliga noise | Restrict TARGET_LEAGUES to soccer_epl in .env |
| All seasons degrading | Threshold too loose | Raise MOVEMENT_THRESHOLD to 0.04 |
| Volume too low | Threshold too tight | Lower MOVEMENT_THRESHOLD to 0.025 |
| Bankroll crashing | Kelly too aggressive | Lower MAX_BET_PCT from 0.03 to 0.02 |

---

## File Output Reference

| File | Generated by | Content |
|------|-------------|---------|
| `data/processed/kelly_lm_results.csv` | `run_kelly_lm.py` | Full bet log with stakes, profit, running bankroll |
| `data/processed/backtest_results.csv` | `src/betting/backtest.py` | Legacy ML backtest (ignore) |
| `logs/pipeline.log` | `run_pipeline.py` | Legacy ML pipeline log |
| `logs/update_closing.log` | `update_closing_odds.py` | Closing odds update log |
| `data/processed/live_bets_pnl.csv` | `run_settler.py --export` | Live bet P&L log với CLV |
| `logs/odds_poller.log` | `run_poller.py` | Poller cycle log |
| `logs/bet_settler.log` | `run_settler.py` | Settlement log |

---

## Phase 2 — Live Odds Poller (Intraday Movement)

**Mục đích:** Detect sharp money signal trong tuần (không cần chờ closing)
bằng cách so sánh odds hiện tại với baseline (odds khi match mới publish).

```powershell
# Test 1 vòng poll (không lưu DB, không gửi Telegram)
python run_poller.py --once --dry-run

# Chạy 1 vòng thật (lưu DB + Telegram)
python run_poller.py --once

# Chạy liên tục (production — để chạy nền hoặc Task Scheduler)
python run_poller.py
```

**Logic:** `baseline_odds` (snapshot đầu tiên) vs `current_odds` (lần poll này).
Movement ≥ 3% → signal fire ngay tại `current_odds` (không phải opening/closing).

---

## Phase 3 — Telegram Alerts

**Setup (1 lần):**
1. Chat với @BotFather trên Telegram → `/newbot` → lấy `BOT_TOKEN`
2. Nhắn tin cho bot → vào URL: `https://api.telegram.org/bot<TOKEN>/getUpdates`
3. Tìm `"chat":{"id": 12345}` → đó là `CHAT_ID`
4. Thêm vào `.env`:
   ```
   TELEGRAM_BOT_TOKEN=1234567890:ABCdef...
   TELEGRAM_CHAT_ID=987654321
   ```

**Test gửi alerts:**
```powershell
python src/signals/telegram_alert.py
```

**Các loại alert:**
- `⚡ SHARP MONEY SIGNAL` — khi signal fire, kèm stake/odds/movement
- `✅ WIN` / `❌ LOSS` — khi bet được settle sau trận
- `📊 Daily P&L` — tóm tắt ngày lúc 7:00 SA

---

## Phase 4 — Live Bet Settler + P&L Tracking

**Settle bets sau khi có kết quả trận:**
```powershell
# Settle tất cả pending bets + in P&L report
python run_settler.py

# Preview không lưu DB
python run_settler.py --dry-run

# Chỉ in P&L report (không settle)
python run_settler.py --report

# Daily summary (settle + Telegram + export CSV)
python run_settler.py --daily

# Export CSV
python run_settler.py --export
```

**CLV (Closing Line Value)** — metric quan trọng nhất:
- `CLV > 0`: đặt tốt hơn closing market → edge execution tốt
- `CLV < 0`: đặt tệ hơn closing → cần poll sớm hơn hoặc act faster
- Nếu avg CLV > 0 sau 50+ bets → hệ thống hoạt động đúng

---

## Statistical Validation (QUAN TRỌNG — chạy trước khi live)

### Bootstrap Confidence Interval — xác nhận edge có thật không

```powershell
python bootstrap_ci.py
```

Trả lời câu hỏi: ROI 12.4% là edge thật hay chỉ là may mắn với n=332?

**Expected output nếu edge thật:**
```
✅ All signals   ROI  +12.4%  95% CI [+3.2%, +21.8%]  p(ROI≤0)=0.003  n=332
✅ Edge CONFIRMED: 95% CI hoàn toàn dương → edge thật
```

**Nếu CI cắt qua 0 (ví dụ [-2%, +18%]):**
```
⚠️ Edge MARGINAL: không đủ confidence — cần thêm data hoặc raise threshold
```

Results được lưu vào `data/processed/bootstrap_ci_results.json`.

---

### Drawdown Guard — Kelly có bảo vệ drawdown

```powershell
# So sánh Plain Kelly vs Guarded Kelly trên historical data
python drawdown_guard.py --compare

# Unit test BankrollState logic
python drawdown_guard.py --test

# Chạy guarded backtest và xem summary
python drawdown_guard.py
```

Drawdown Guard hoạt động theo 3 lớp:
- **Smooth scaling**: Kelly tự giảm khi bankroll dưới peak (liên tục)
- **Tier system**: > -10% DD → bet 60%, > -15% → bet 30%, > -20% → pause
- **Streak protection**: thua 4 lần liên tiếp → giảm stake 50%

Tích hợp vào live trading: thay `kelly_stake()` bằng `BankrollState.kelly_stake()` trong `odds_poller.py`.

---

### CLV Tracker — proper Closing Line Value

```powershell
# In CLV report từ settled bets
python clv_tracker.py --report

# Test logic tính CLV
python clv_tracker.py --test
```

Khác với CLV hiện tại trong `bet_settler.py` (raw, có vig):
- `compute_clv_proper()` remove vig từ cả 3 outcomes trước khi so sánh
- `CLVMonitor` theo dõi CLV trend theo tuần để phát hiện edge decay sớm

**Tích hợp vào bet_settler.py**: thêm vào cuối `settle_one_bet()`:
```python
from clv_tracker import settle_with_proper_clv
settle_with_proper_clv(session, live_bet, match_info)
```

---

## Edge Validation (chạy sau Bootstrap CI)

```powershell
python validate_edge.py
```

So sánh ROI khi đặt tại opening odds vs closing odds:
- ROI+ ở closing → edge thật → proceed với live trading
- ROI≈0 ở closing → edge yếu → cần early execution
- ROI- ở closing → edge không thật → re-evaluate strategy

---

## Task Scheduler Setup (Windows — chạy poller tự động)

```powershell
# Tạo scheduled task chạy poller mỗi 60 phút
schtasks /create /tn "PipelineDish_Poller" /tr "python D:\dream_project\pipeline_dish\run_poller.py --once" /sc minute /mo 60 /ru SYSTEM

# Tạo scheduled task settle + daily summary lúc 7:00 SA
schtasks /create /tn "PipelineDish_DailySummary" /tr "python D:\dream_project\pipeline_dish\run_settler.py --daily" /sc daily /st 07:00
```
