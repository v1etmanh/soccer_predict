# Workflow — How to Run Everything

## Current Config — S6 (Active as of 2026-05-04)

```env
TARGET_LEAGUES=soccer_epl
ALLOWED_SIDES=A
MOVEMENT_THRESHOLD=0.03
KELLY_FRACTION=0.25
MAX_BET_PCT=0.03
```

Bootstrap CI confirmed: ROI +20.4% | 95% CI [+0.1%, +41.0%] | p(ROI≤0)=2.5% | n=179

---

## Prerequisites

```bash
pip install sqlalchemy loguru requests pandas python-dotenv xgboost lightgbm scikit-learn flask
```

Create `.env` in project root:

```env
ODDS_API_KEY=your_key_here
DB_PATH=D:/dream_project/pipeline_dish/data/pipeline.db
TARGET_LEAGUES=soccer_epl
ALLOWED_SIDES=A
INITIAL_BANKROLL=1000
KELLY_FRACTION=0.25
```

---

## First-Time Setup (run once)

```powershell
# 1. Initialize database
python -c "import sys; sys.path.insert(0,'src'); from data.database import init_db; init_db()"

# 2. Scrape historical data (~3430 matches EPL + BL)
python -c "import sys; sys.path.insert(0,'src'); from data.scraper import run_scraper; run_scraper()"

# 3. One-time DB migration (closing odds columns)
python migrate_closing_odds.py

# 4. Backfill closing odds
python update_closing_odds.py
```

---

## Daily Health Check Commands

```powershell
# 1. Flat backtest — verify engine sau code change (< 1 giây)
python run_backtest_lm.py
# Expected S6: 1900 matches loaded | 179 signals | ROI +20.44%

# 2. Season check — health check quan trọng nhất
python test_by_season.py
# Healthy: 4/5 mùa PROFIT. Chỉ 2021-22 LOSS (suspected variance)

# 3. Kelly simulation
python run_kelly_lm.py
# Expected S6: $1000 → $1603 | MaxDD -11.05%

# 4. DB state check
python check_db.py
# Shows: snapshots, baselines, live bets count
```

---

## Strategy Optimization (chạy khi muốn re-evaluate config)

```powershell
# Test 7 scenarios bootstrap CI, tìm config tốt nhất
python optimize_strategy.py
```

So sánh: threshold, leagues, sides. Chọn config có CI lower > 0.
Output: `data/processed/strategy_comparison.json`

**Key rule:** Tăng threshold không nhất thiết tốt hơn — giảm n làm CI nới rộng.

---

## Bootstrap CI (xác nhận edge)

```powershell
python bootstrap_ci.py
```

S6 expected output:
```
[OK] All signals   ROI +20.4%  95% CI [+0.1%, +41.0%]  p(ROI≤0)=0.025  n=179
[OK] Edge CONFIRMED: 95% CI hoàn toàn dương
```

Nếu CI cắt qua 0 → chạy `optimize_strategy.py` để tìm config tốt hơn.

---

## Drawdown Guard

```powershell
python drawdown_guard.py --test     # Unit test BankrollState
python drawdown_guard.py --compare  # Plain Kelly vs Guarded Kelly
python drawdown_guard.py            # Guarded backtest summary
```

Guard đã được wire vào `odds_poller.py` và `bet_settler.py`.
BankrollState tự reconstruct từ settled bets trong DB khi process restart.

**Tiers:** >-10% DD: 60% Kelly | >-15%: 30% | >-20%: pause | >-25%: emergency stop

---

## Live Pipeline — Paper Trading

### Setup Task Scheduler (chạy 1 lần với quyền Admin)

```powershell
# Poller mỗi 60 phút
schtasks /create /tn "PipelineDish_Poller" /tr "python D:\dream_project\pipeline_dish\run_poller.py --once" /sc minute /mo 60 /ru SYSTEM /f

# Settler 7:00 SA mỗi ngày
schtasks /create /tn "PipelineDish_Settler" /tr "python D:\dream_project\pipeline_dish\run_settler.py --daily" /sc daily /st 07:00 /f

# Verify
schtasks /query /tn "PipelineDish_Poller"
schtasks /query /tn "PipelineDish_Settler"
```

### Poller commands

```powershell
# Test 1 vòng, không lưu DB
python run_poller.py --once --dry-run

# Chạy 1 vòng thật
python run_poller.py --once

# Continuous loop (thường dùng Task Scheduler thay)
python run_poller.py
```

**Logic:** Vòng 1 → tạo baselines. Vòng 2+ → tính movement vs baseline.
Movement ≥ 3% trên Away side → signal fire → LiveBet lưu vào DB.

### Settler commands

```powershell
python run_settler.py              # Settle + P&L report
python run_settler.py --dry-run   # Preview không lưu DB
python run_settler.py --report    # Chỉ in P&L
python run_settler.py --daily     # Full daily (settle + Telegram + export)
python run_settler.py --export    # Export CSV
```

### CLV check (sau khi có bets settle)

```powershell
python clv_tracker.py --report
```

- `CLV > 0`: đặt trước closing, execution tốt ✅
- `CLV < 0`: signal fire quá muộn → cần poll dày hơn (30 phút)
- Cần 20+ bets để có kết luận

---

## Telegram Alerts (Optional)

```powershell
# Setup trong .env:
TELEGRAM_BOT_TOKEN=1234567890:ABCdef...
TELEGRAM_CHAT_ID=987654321

# Test
python src/signals/telegram_alert.py
```

Hướng dẫn lấy token: @BotFather trên Telegram → /newbot

---

## Paper Trading Monitoring Schedule

| Thời điểm | Lệnh | Mục đích |
|-----------|------|---------|
| Thứ Tư/Năm | `python run_poller.py --once` | Xem movement có tích lũy chưa |
| Trước trận | `python check_db.py` | Confirm live bets đã lưu |
| Sau trận | `python run_settler.py` | Settle + xem CLV |
| Sau 10+ bets | `python clv_tracker.py --report` | Đánh giá execution quality |
| Sau 20+ bets | `python bootstrap_ci.py` | CI có chặt hơn không? |
| Sau 50+ bets CLV > 0 | Xem xét real money | Chỉ khi avg CLV dương |

---

## What To Do If ROI Drops

```powershell
# 1. Check side breakdown
python test_no_home.py

# 2. Check season/league breakdown
python test_by_season.py

# 3. Re-run scenario optimization
python optimize_strategy.py

# 4. Deep-dive vào mùa có vấn đề
python diagnose_2425.py
```

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| CI drops below 0 | Edge decaying | Chạy optimize_strategy.py, tìm config mới |
| CLV < 0 liên tục | Signal fire quá muộn | Poll 30 phút thay 60 phút |
| Bundesliga drag | Đã loại — verify .env | Check TARGET_LEAGUES=soccer_epl |
| MaxDD > 20% | Guard kích hoạt pause | Chờ recover, không override |
| 4+ seasons LOSS | Strategy broken | Re-evaluate với data mới |

---

## Updating Data (mid-season hoặc new season)

```powershell
python -c "import sys; sys.path.insert(0,'src'); from data.scraper import run_scraper; run_scraper()"
python update_closing_odds.py
python run_backtest_lm.py   # Verify
```

---

## File Output Reference

| File | Generated by | Content |
|------|-------------|---------|
| `data/processed/kelly_lm_results.csv` | `run_kelly_lm.py` | Kelly bet log |
| `data/processed/bootstrap_ci_results.json` | `bootstrap_ci.py` | CI results |
| `data/processed/strategy_comparison.json` | `optimize_strategy.py` | 7-scenario comparison |
| `data/processed/live_bets_pnl.csv` | `run_settler.py --export` | Live P&L với CLV |
| `logs/odds_poller.log` | `run_poller.py` | Poller activity |
| `logs/bet_settler.log` | `run_settler.py` | Settlement log |
| `logs/bootstrap_ci.log` | `bootstrap_ci.py` | Bootstrap log |
