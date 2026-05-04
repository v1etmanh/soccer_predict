"""
bet_settler.py — Phase 4: Live Bet Settler + P&L Tracker
==========================================================
Chạy sau mỗi gameweek (hoặc hàng ngày) để:
  1. Tìm các LiveBet chưa settle (settled=False)
  2. Match với kết quả trận từ bảng matches (DB lịch sử)
     hoặc fetch kết quả live từ The Odds API
  3. Cập nhật: won, profit, bankroll_after, clv, settled=True
  4. Gửi Telegram result alert cho mỗi bet settle
  5. Xuất báo cáo P&L tổng hợp

CLV (Closing Line Value) là metric quan trọng nhất để đánh giá
chất lượng execution: nếu bet_odds > close_odds liên tục → đang
đặt tốt hơn market, edge là thật.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    INITIAL_BANKROLL, LOGS_DIR,
    PROCESSED_DATA_DIR, PAPER_TRADING,
)
from data.database import get_session, LiveBet, Match, OddsSnapshot
from signals.telegram_alert import send_result_alert, send_daily_summary

# ── Drawdown Guard: update state sau mỗi settle ──────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from drawdown_guard import BankrollState

def _get_guard_for_settler(session) -> BankrollState:
    """Reconstruct guard state từ toàn bộ settled history."""
    settled = (
        session.query(LiveBet)
        .filter_by(settled=True)
        .order_by(LiveBet.id)
        .all()
    )
    state = BankrollState(initial=INITIAL_BANKROLL)
    for bet in settled:
        if bet.won is not None and bet.profit is not None:
            state.update(won=bet.won, profit=bet.profit)
    return state

logger.add(
    LOGS_DIR / "bet_settler.log",
    rotation="5 MB",
    retention="30 days",
    level="INFO",
)


# ══════════════════════════════════════════════════════════════════
# PHẦN 1: Tìm kết quả trận cho 1 LiveBet
# ══════════════════════════════════════════════════════════════════

def find_match_result(session, live_bet: LiveBet) -> Optional[dict]:
    """
    Tìm kết quả trận tương ứng với live_bet trong bảng matches.
    Match bằng home_team + away_team + match_date (±1 ngày).
    
    Trả về dict với result + close_odds nếu tìm thấy, None nếu chưa có.
    """
    if not live_bet.match_date:
        return None

    # Window ±1 ngày để tránh timezone mismatch
    date_from = live_bet.match_date - timedelta(hours=36)
    date_to   = live_bet.match_date + timedelta(hours=36)

    match = (
        session.query(Match)
        .filter(
            Match.home_team == live_bet.home_team,
            Match.away_team == live_bet.away_team,
            Match.date >= date_from,
            Match.date <= date_to,
        )
        .first()
    )

    if match is None or match.result is None:
        return None

    # Lấy closing odds theo side
    close_odds_map = {
        "H": match.close_h,
        "D": match.close_d,
        "A": match.close_a,
    }

    return {
        "result":     match.result,
        "close_h":    match.close_h,
        "close_d":    match.close_d,
        "close_a":    match.close_a,
        "close_odds": close_odds_map.get(live_bet.signal_side),
    }


def compute_clv(bet_odds: float, close_odds: Optional[float]) -> Optional[float]:
    """
    Closing Line Value = close_implied_prob - bet_implied_prob
    
    close_prob = 1 / close_odds (giả sử no-vig, xấp xỉ đủ tốt)
    bet_prob   = 1 / bet_odds
    
    CLV > 0: đặt tốt hơn market (bet_odds > close_odds)
    CLV < 0: đặt tệ hơn market (đã bị "eaten" một phần edge)
    """
    if not close_odds or close_odds <= 0 or not bet_odds or bet_odds <= 0:
        return None
    close_prob = 1.0 / close_odds
    bet_prob   = 1.0 / bet_odds
    return round(close_prob - bet_prob, 4)


# ══════════════════════════════════════════════════════════════════
# PHẦN 2: Settle 1 bet
# ══════════════════════════════════════════════════════════════════

def settle_one_bet(session, live_bet: LiveBet, match_info: dict) -> bool:
    """
    Settle 1 bet với kết quả đã biết.
    Cập nhật DB và gửi Telegram alert.
    
    Returns True nếu settle thành công.
    """
    result    = match_info["result"]
    won       = (result == live_bet.signal_side)
    bet_odds  = live_bet.bet_odds or 1.0
    stake     = live_bet.stake or 0.0

    profit = round(stake * (bet_odds - 1) if won else -stake, 2)
    bankroll_before = live_bet.bankroll_before or INITIAL_BANKROLL
    bankroll_after  = round(bankroll_before + profit, 2)

    clv = compute_clv(bet_odds, match_info.get("close_odds"))

    # Update LiveBet record
    live_bet.result        = result
    live_bet.won           = won
    live_bet.profit        = profit
    live_bet.bankroll_after = bankroll_after
    live_bet.close_odds    = match_info.get("close_odds")
    live_bet.clv           = clv
    live_bet.settled       = True

    session.commit()

    win_str  = "WIN ✅" if won else "LOSS ❌"
    clv_str  = f"CLV={clv:+.3f}" if clv is not None else "CLV=n/a"
    logger.success(
        f"[Settler] {live_bet.home_team} vs {live_bet.away_team} "
        f"→ {win_str} | "
        f"Profit: {profit:+.2f} | {clv_str} | "
        f"Bankroll: {bankroll_before:.2f} → {bankroll_after:.2f}"
    )

    # Telegram alert
    try:
        send_result_alert(live_bet)
    except Exception as e:
        logger.warning(f"[Settler] Telegram alert failed: {e}")

    return True


# ══════════════════════════════════════════════════════════════════
# PHẦN 3: Chạy settlement cho tất cả pending bets
# ══════════════════════════════════════════════════════════════════

def run_settlement(dry_run: bool = False) -> dict:
    """
    Tìm và settle tất cả pending bets.
    
    Returns dict summary với số bets settled, P&L, etc.
    """
    session = get_session()
    settled_count  = 0
    skipped_count  = 0
    total_profit   = 0.0

    try:
        pending = (
            session.query(LiveBet)
            .filter_by(settled=False)
            .order_by(LiveBet.match_date)
            .all()
        )

        logger.info(f"[Settler] Found {len(pending)} pending bets to settle.")

        # Reconstruct guard state từ history trước khi settle batch này
        guard = _get_guard_for_settler(session)
        logger.info(f"[Settler] Guard state: {guard.status_str()}")

        for bet in pending:
            match_info = find_match_result(session, bet)

            if match_info is None:
                logger.debug(
                    f"[Settler] No result yet: {bet.home_team} vs {bet.away_team} "
                    f"({bet.match_date})"
                )
                skipped_count += 1
                continue

            if dry_run:
                logger.info(
                    f"[Settler][DRY RUN] Would settle: "
                    f"{bet.home_team} vs {bet.away_team} → {match_info['result']}"
                )
                settled_count += 1
                continue

            success = settle_one_bet(session, bet, match_info)
            if success:
                settled_count += 1
                total_profit += bet.profit or 0
                # Cập nhật guard state sau mỗi bet settle
                guard.update(won=bet.won, profit=bet.profit or 0)
                logger.debug(f"[Settler] Guard updated: {guard.status_str()}")

    except Exception as e:
        logger.error(f"[Settler] Error during settlement: {e}", exc_info=True)
    finally:
        session.close()

    summary = {
        "settled": settled_count,
        "skipped": skipped_count,
        "total_profit": round(total_profit, 2),
    }
    logger.info(
        f"[Settler] Done. Settled: {settled_count} | "
        f"Pending: {skipped_count} | P&L: {total_profit:+.2f}"
    )
    return summary


# ══════════════════════════════════════════════════════════════════
# PHẦN 4: P&L Report
# ══════════════════════════════════════════════════════════════════

def compute_pnl_stats(session) -> dict:
    """Tính toán P&L tổng hợp từ tất cả settled bets."""
    settled = session.query(LiveBet).filter_by(settled=True).all()

    if not settled:
        return {}

    df = pd.DataFrame([{
        "date":         b.match_date,
        "home":         b.home_team,
        "away":         b.away_team,
        "league":       b.league,
        "side":         b.signal_side,
        "bet_odds":     b.bet_odds,
        "stake":        b.stake,
        "won":          b.won,
        "profit":       b.profit,
        "bankroll_bef": b.bankroll_before,
        "bankroll_aft": b.bankroll_after,
        "clv":          b.clv,
        "movement":     b.movement,
    } for b in settled])

    total_bets  = len(df)
    wins        = int(df["won"].sum())
    losses      = total_bets - wins
    total_profit = df["profit"].sum()
    avg_stake   = df["stake"].mean()
    roi         = total_profit / (df["stake"].sum()) if df["stake"].sum() > 0 else 0

    # Max drawdown
    bankrolls   = [INITIAL_BANKROLL] + list(df["bankroll_aft"].dropna())
    peak        = bankrolls[0]
    max_dd      = 0.0
    for b in bankrolls:
        peak = max(peak, b)
        dd = (b - peak) / peak if peak > 0 else 0
        max_dd = min(max_dd, dd)

    # CLV stats
    clv_valid   = df["clv"].dropna()
    avg_clv     = clv_valid.mean() if not clv_valid.empty else None
    clv_pos_pct = (clv_valid > 0).mean() if not clv_valid.empty else None

    # By league
    by_league = {}
    for lg, grp in df.groupby("league"):
        by_league[lg] = {
            "bets":     len(grp),
            "wins":     int(grp["won"].sum()),
            "roi":      round(grp["profit"].sum() / grp["stake"].sum(), 4)
                        if grp["stake"].sum() > 0 else 0,
            "profit":   round(grp["profit"].sum(), 2),
            "avg_clv":  round(grp["clv"].mean(), 4) if grp["clv"].notna().any() else None,
        }

    # Current bankroll
    last_bankroll = df["bankroll_aft"].iloc[-1] if not df.empty else INITIAL_BANKROLL

    return {
        "total_bets":    total_bets,
        "wins":          wins,
        "losses":        losses,
        "win_rate":      round(wins / total_bets, 4) if total_bets > 0 else 0,
        "total_profit":  round(total_profit, 2),
        "total_staked":  round(df["stake"].sum(), 2),
        "roi":           round(roi, 4),
        "avg_stake":     round(avg_stake, 2),
        "bankroll":      round(last_bankroll, 2),
        "max_drawdown":  round(max_dd, 4),
        "avg_clv":       round(avg_clv, 4) if avg_clv is not None else None,
        "clv_positive_pct": round(clv_pos_pct, 3) if clv_pos_pct is not None else None,
        "by_league":     by_league,
    }


def print_pnl_report(stats: dict) -> None:
    """In P&L report đẹp ra console."""
    if not stats:
        print("[Report] Chưa có bets nào được settle.")
        return

    paper_tag = " [PAPER TRADING]" if PAPER_TRADING else " [REAL MONEY]"
    print(f"\n{'='*55}")
    print(f"  LIVE BETTING P&L REPORT{paper_tag}")
    print(f"{'='*55}")
    print(f"  Total bets    : {stats['total_bets']}")
    print(f"  Win / Loss    : {stats['wins']}W / {stats['losses']}L "
          f"({stats['win_rate']:.1%})")
    print(f"  Total profit  : {stats['total_profit']:+.2f} units")
    print(f"  Total staked  : {stats['total_staked']:.2f} units")
    print(f"  ROI           : {stats['roi']:+.2%}")
    print(f"  Bankroll now  : {stats['bankroll']:.2f}")
    print(f"  Max drawdown  : {stats['max_drawdown']:.1%}")

    if stats.get("avg_clv") is not None:
        clv_flag = "✅" if stats["avg_clv"] > 0 else "⚠️"
        print(f"\n  {clv_flag} Avg CLV      : {stats['avg_clv']:+.3f}")
        print(f"  CLV+ rate     : {stats['clv_positive_pct']:.1%} of bets beat closing")

    if stats.get("by_league"):
        print(f"\n  By league:")
        for lg, s in stats["by_league"].items():
            clv_str = f" | CLV {s['avg_clv']:+.3f}" if s.get("avg_clv") is not None else ""
            print(f"    {lg:35s} {s['bets']:3d} bets | ROI {s['roi']:+.1%}{clv_str}")
    print(f"{'='*55}\n")


def export_pnl_csv(session) -> Path:
    """Xuất tất cả settled bets ra CSV."""
    settled = session.query(LiveBet).filter_by(settled=True).order_by(LiveBet.match_date).all()
    rows = [{
        "date":          b.match_date,
        "league":        b.league,
        "home_team":     b.home_team,
        "away_team":     b.away_team,
        "signal_side":   b.signal_side,
        "movement":      b.movement,
        "baseline_odds": b.baseline_odds,
        "bet_odds":      b.bet_odds,
        "stake":         b.stake,
        "result":        b.result,
        "won":           b.won,
        "profit":        b.profit,
        "bankroll_after":b.bankroll_after,
        "close_odds":    b.close_odds,
        "clv":           b.clv,
        "is_paper":      b.is_paper,
        "signal_fired":  b.signal_fired_at,
    } for b in settled]

    out = PROCESSED_DATA_DIR / "live_bets_pnl.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    logger.success(f"[Report] Exported {len(rows)} settled bets → {out}")
    return out


# ══════════════════════════════════════════════════════════════════
# PHẦN 5: Daily summary (gọi từ scheduler)
# ══════════════════════════════════════════════════════════════════

def run_daily_summary() -> bool:
    """Gọi lúc 7:00 SA mỗi ngày — settle + send daily P&L."""
    # 1. Settle pending bets
    run_settlement()

    # 2. Compute stats
    session = get_session()
    try:
        stats = compute_pnl_stats(session)
        print_pnl_report(stats)

        if not stats:
            return False

        # Pending count
        pending = session.query(LiveBet).filter_by(settled=False).count()

        # Today's bets
        today_start = datetime.now().replace(hour=0, minute=0, second=0)
        today_bets = [
            b for b in session.query(LiveBet).filter_by(settled=True).all()
            if b.match_date and b.match_date.replace(tzinfo=None) >= today_start
        ]
        today_wins   = sum(1 for b in today_bets if b.won)
        today_losses = sum(1 for b in today_bets if not b.won)
        today_profit = sum(b.profit or 0 for b in today_bets)
        today_staked = sum(b.stake or 0 for b in today_bets)
        today_roi    = today_profit / today_staked if today_staked > 0 else 0

        summary_payload = {
            "date":         datetime.now().strftime("%d/%m/%Y"),
            "total_bets":   len(today_bets),
            "wins":         today_wins,
            "losses":       today_losses,
            "profit":       round(today_profit, 2),
            "bankroll":     stats.get("bankroll", INITIAL_BANKROLL),
            "roi_today":    round(today_roi, 4),
            "roi_alltime":  stats.get("roi", 0),
            "max_drawdown": stats.get("max_drawdown", 0),
            "pending_bets": pending,
        }
        send_daily_summary(summary_payload)
        export_pnl_csv(session)
        return True
    finally:
        session.close()


# ══════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Settle live bets + P&L report")
    parser.add_argument("--dry-run",  action="store_true", help="Preview only, no DB write")
    parser.add_argument("--report",   action="store_true", help="Only print P&L, no settlement")
    parser.add_argument("--daily",    action="store_true", help="Run full daily summary")
    parser.add_argument("--export",   action="store_true", help="Export CSV only")
    args = parser.parse_args()

    if args.daily:
        run_daily_summary()
    elif args.report:
        session = get_session()
        stats = compute_pnl_stats(session)
        print_pnl_report(stats)
        session.close()
    elif args.export:
        session = get_session()
        out = export_pnl_csv(session)
        print(f"Exported → {out}")
        session.close()
    else:
        run_settlement(dry_run=args.dry_run)
        session = get_session()
        stats = compute_pnl_stats(session)
        print_pnl_report(stats)
        session.close()
