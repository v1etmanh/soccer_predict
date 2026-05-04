"""
telegram_alert.py — Phase 3: Telegram Signal Alert Bot
========================================================
Gửi thông báo qua Telegram khi:
  - Signal mới fire (BET alert)
  - Bet được settle (WIN / LOSS result)
  - Daily P&L summary (7:00 SA hàng ngày)
  - Error alert nếu poller crash

Setup:
  1. Tạo bot qua @BotFather → lấy BOT_TOKEN
  2. Nhắn tin cho bot, lấy CHAT_ID từ:
     https://api.telegram.org/bot<TOKEN>/getUpdates
  3. Thêm vào .env:
     TELEGRAM_BOT_TOKEN=1234567890:ABCdef...
     TELEGRAM_CHAT_ID=987654321
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, PAPER_TRADING

# Ký hiệu side dễ đọc
SIDE_EMOJI = {"H": "🏠 HOME", "D": "🤝 DRAW", "A": "✈️ AWAY"}
LEAGUE_DISPLAY = {
    "soccer_epl":                    "🏴󠁧󠁢󠁥󠁮󠁧󠁿 EPL",
    "soccer_germany_bundesliga":     "🇩🇪 Bundesliga",
    "soccer_spain_la_liga":          "🇪🇸 La Liga",
    "soccer_italy_serie_a":          "🇮🇹 Serie A",
    "soccer_france_ligue_1":         "🇫🇷 Ligue 1",
    "soccer_netherlands_eredivisie": "🇳🇱 Eredivisie",
}


# ══════════════════════════════════════════════════════════════════
# CORE: gửi raw message
# ══════════════════════════════════════════════════════════════════

def _send(text: str, parse_mode: str = "HTML") -> bool:
    """
    Gửi 1 message tới Telegram chat.
    Trả về True nếu thành công.
    """
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.debug("[Telegram] Chưa config BOT_TOKEN/CHAT_ID — bỏ qua alert.")
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id":    TELEGRAM_CHAT_ID,
        "text":       text,
        "parse_mode": parse_mode,
    }
    try:
        resp = requests.post(url, json=payload, timeout=10)
        resp.raise_for_status()
        return True
    except Exception as e:
        logger.warning(f"[Telegram] Send failed: {e}")
        return False


# ══════════════════════════════════════════════════════════════════
# ALERT 1: Signal fire — BET ngay
# ══════════════════════════════════════════════════════════════════

def send_signal_alert(signal: dict) -> bool:
    """
    Gửi alert khi signal fire.
    
    signal dict cần có:
      home_team, away_team, league, match_date,
      signal_side, movement, baseline_odds, bet_odds,
      stake, bankroll_before, curr_prob
    """
    side      = signal.get("signal_side", "?")
    home      = signal.get("home_team", "?")
    away      = signal.get("away_team", "?")
    league    = signal.get("league", "")
    movement  = signal.get("movement", 0)
    bet_odds  = signal.get("bet_odds", 0)
    base_odds = signal.get("baseline_odds", 0)
    stake     = signal.get("stake", 0)
    bankroll  = signal.get("bankroll_before", 0)
    prob      = signal.get("curr_prob", 0)
    match_date = signal.get("match_date")

    # Format date
    date_str = ""
    if match_date:
        if hasattr(match_date, "strftime"):
            date_str = match_date.strftime("%d/%m %H:%M UTC")
        else:
            date_str = str(match_date)[:16]

    league_disp = LEAGUE_DISPLAY.get(league, league)
    side_disp   = SIDE_EMOJI.get(side, side)
    paper_tag   = " 📋 <i>PAPER</i>" if PAPER_TRADING else " 💰 <b>REAL</b>"
    pct_stake   = round(stake / bankroll * 100, 1) if bankroll > 0 else 0

    text = (
        f"⚡ <b>SHARP MONEY SIGNAL</b>{paper_tag}\n"
        f"{'─'*30}\n"
        f"🏆 {league_disp}\n"
        f"⚽ <b>{home}</b> vs <b>{away}</b>\n"
        f"📅 {date_str}\n"
        f"\n"
        f"🎯 Bet: <b>{side_disp}</b>\n"
        f"📈 Movement: <b>{movement:+.1%}</b> "
        f"(Baseline {base_odds:.2f} → Now {bet_odds:.2f})\n"
        f"🎲 Implied prob: {prob:.1%}\n"
        f"\n"
        f"💵 Stake: <b>{stake:.2f}</b> ({pct_stake:.1f}% bankroll)\n"
        f"🏦 Bankroll: {bankroll:.2f}\n"
        f"\n"
        f"<i>Threshold: {movement:.3f} ≥ movement required</i>"
    )
    return _send(text)


# ══════════════════════════════════════════════════════════════════
# ALERT 2: Bet settled — WIN / LOSS
# ══════════════════════════════════════════════════════════════════

def send_result_alert(live_bet) -> bool:
    """
    Gửi alert khi bet được settle sau trận.
    live_bet là object LiveBet từ DB.
    """
    won          = live_bet.won
    side         = live_bet.signal_side or "?"
    home         = live_bet.home_team or "?"
    away         = live_bet.away_team or "?"
    league       = live_bet.league or ""
    stake        = live_bet.stake or 0
    profit       = live_bet.profit or 0
    bankroll_aft = live_bet.bankroll_after or 0
    bet_odds     = live_bet.bet_odds or 0
    clv          = live_bet.clv

    league_disp = LEAGUE_DISPLAY.get(league, league)
    side_disp   = SIDE_EMOJI.get(side, side)
    result_icon = "✅ <b>WIN</b>" if won else "❌ <b>LOSS</b>"

    clv_str = ""
    if clv is not None:
        clv_icon = "🔥" if clv > 0 else "⚠️"
        clv_str  = f"\n{clv_icon} CLV: <b>{clv:+.3f}</b> {'(beat closing)' if clv > 0 else '(behind closing)'}"

    text = (
        f"{result_icon}\n"
        f"{'─'*30}\n"
        f"🏆 {league_disp}\n"
        f"⚽ {home} vs {away}\n"
        f"🎯 Bet: {side_disp} @ {bet_odds:.2f}\n"
        f"\n"
        f"{'💰' if won else '💸'} Profit: <b>{profit:+.2f}</b>\n"
        f"🏦 Bankroll: {bankroll_aft:.2f}"
        f"{clv_str}"
    )
    return _send(text)


# ══════════════════════════════════════════════════════════════════
# ALERT 3: Daily P&L summary
# ══════════════════════════════════════════════════════════════════

def send_daily_summary(stats: dict) -> bool:
    """
    Gửi tóm tắt ngày.
    
    stats dict cần có:
      date, total_bets, wins, losses, profit, bankroll,
      roi_today, roi_alltime, max_drawdown
    """
    date         = stats.get("date", datetime.now().strftime("%d/%m/%Y"))
    total_bets   = stats.get("total_bets", 0)
    wins         = stats.get("wins", 0)
    losses       = stats.get("losses", 0)
    profit       = stats.get("profit", 0)
    bankroll     = stats.get("bankroll", 0)
    roi_today    = stats.get("roi_today", 0)
    roi_alltime  = stats.get("roi_alltime", 0)
    max_dd       = stats.get("max_drawdown", 0)
    pending      = stats.get("pending_bets", 0)

    trend = "📈" if profit >= 0 else "📉"
    paper_tag = " (PAPER)" if PAPER_TRADING else ""

    text = (
        f"📊 <b>Daily P&L{paper_tag}</b> — {date}\n"
        f"{'─'*30}\n"
        f"🎯 Bets today: {total_bets} | ✅ {wins}W ❌ {losses}L\n"
        f"{trend} Profit today: <b>{profit:+.2f}</b>\n"
        f"📈 ROI today: <b>{roi_today:+.1%}</b>\n"
        f"\n"
        f"🏦 Bankroll: <b>{bankroll:.2f}</b>\n"
        f"📈 All-time ROI: {roi_alltime:+.1%}\n"
        f"📉 Max drawdown: {max_dd:.1%}\n"
    )
    if pending > 0:
        text += f"\n⏳ Pending settlement: {pending} bets"
    return _send(text)


# ══════════════════════════════════════════════════════════════════
# ALERT 4: Error / system alert
# ══════════════════════════════════════════════════════════════════

def send_error_alert(error_msg: str, context: str = "") -> bool:
    """Gửi alert khi có lỗi nghiêm trọng."""
    text = (
        f"🚨 <b>SYSTEM ERROR</b>\n"
        f"{'─'*30}\n"
        f"Context: {context}\n"
        f"Error: <code>{error_msg[:500]}</code>\n"
        f"Time: {datetime.now(timezone.utc).strftime('%H:%M UTC')}"
    )
    return _send(text)


def send_startup_message() -> bool:
    """Gửi message khi poller khởi động."""
    from config import TARGET_LEAGUES, POLL_INTERVAL_MINUTES
    leagues_str = ", ".join(TARGET_LEAGUES)
    text = (
        f"🚀 <b>Pipeline Dish Started</b>\n"
        f"{'─'*30}\n"
        f"Leagues: {leagues_str}\n"
        f"Poll interval: {POLL_INTERVAL_MINUTES}m\n"
        f"Mode: {'📋 PAPER' if PAPER_TRADING else '💰 REAL'}\n"
        f"Time: {datetime.now(timezone.utc).strftime('%d/%m %H:%M UTC')}"
    )
    return _send(text)


# ══════════════════════════════════════════════════════════════════
# CLI test
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    """Test gửi các loại message."""
    print("Testing Telegram alerts...")

    ok = send_startup_message()
    print(f"Startup: {'✓' if ok else '✗'}")

    test_signal = {
        "home_team": "Arsenal",
        "away_team": "Chelsea",
        "league": "soccer_epl",
        "match_date": datetime.now(timezone.utc),
        "signal_side": "A",
        "movement": 0.047,
        "baseline_odds": 3.60,
        "bet_odds": 3.10,
        "stake": 28.50,
        "bankroll_before": 1050.00,
        "curr_prob": 0.318,
    }
    ok = send_signal_alert(test_signal)
    print(f"Signal alert: {'✓' if ok else '✗'}")

    test_summary = {
        "date": datetime.now().strftime("%d/%m/%Y"),
        "total_bets": 3,
        "wins": 2,
        "losses": 1,
        "profit": 41.20,
        "bankroll": 1091.20,
        "roi_today": 0.062,
        "roi_alltime": 0.091,
        "max_drawdown": -0.087,
        "pending_bets": 1,
    }
    ok = send_daily_summary(test_summary)
    print(f"Daily summary: {'✓' if ok else '✗'}")
