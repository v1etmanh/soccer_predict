"""
odds_poller.py — Phase 2: Intraday Odds Snapshot Poller
=========================================================
Chạy liên tục (scheduler hoặc cron), poll The Odds API mỗi POLL_INTERVAL_MINUTES.
Logic:
  1. Fetch tất cả upcoming matches + current odds
  2. Lưu vào bảng odds_snapshots (mỗi lần poll = 1 row mới)
  3. So sánh snapshot hiện tại với BASELINE (snapshot đầu tiên của trận)
  4. Nếu movement >= threshold => fire signal ngay (không cần chờ closing)
  5. Signal đã fire cho 1 trận sẽ không fire lại (idempotent)

Đây là cách live trading hoạt động thực tế:
  - Baseline ≈ odds khi match mới published (thứ Hai/Ba)
  - Current  ≈ odds tại thời điểm poll (có thể thứ Tư/Năm/Sáu)
  - Movement = current - baseline (sharp money đã vào từ đó đến giờ)
  - Đặt cược tại current odds (không phải opening, không phải closing)
"""
from __future__ import annotations

import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    ODDS_API_KEY, ODDS_API_BASE, TARGET_LEAGUES,
    MOVEMENT_THRESHOLD, MIN_CLOSE_PROB, MAX_CLOSE_PROB,
    MIN_OPEN_ODDS, ALLOWED_SIDES,
    POLL_INTERVAL_MINUTES, SIGNAL_LOOKBACK_DAYS,
    MAX_BET_PCT, MIN_BET_PCT, INITIAL_BANKROLL, LOGS_DIR,
)
from data.database import get_session, OddsSnapshot, LiveBet
from signals.live_signals import fetch_live_odds, parse_odds_event
from signals.telegram_alert import send_signal_alert   # Phase 3

import requests

logger.add(
    LOGS_DIR / "odds_poller.log",
    rotation="10 MB",
    retention="30 days",
    level="INFO",
)


# ══════════════════════════════════════════════════════════════════
# PHẦN 1: Lưu snapshot vào DB
# ══════════════════════════════════════════════════════════════════

def save_snapshot(session, event: dict, is_baseline: bool = False) -> OddsSnapshot:
    """Lưu 1 snapshot odds vào DB. Luôn tạo row mới (time series)."""
    match_key = _make_match_key(event)
    snap = OddsSnapshot(
        match_key  = match_key,
        league     = event["league"],
        home_team  = event["home_team"],
        away_team  = event["away_team"],
        match_date = event.get("commence"),
        bookmaker  = "median_all",
        odds_h     = event["odds_h"],
        odds_d     = event["odds_d"],
        odds_a     = event["odds_a"],
        is_baseline = is_baseline,
        snapshot_at = datetime.now(timezone.utc),
    )
    session.add(snap)
    return snap


def _make_match_key(event: dict) -> str:
    """Tạo unique key cho trận: 'Arsenal__Chelsea__2025-05-10'"""
    date_str = ""
    if event.get("commence"):
        c = event["commence"]
        if hasattr(c, "strftime"):
            date_str = c.strftime("%Y-%m-%d")
        else:
            date_str = str(c)[:10]
    return f"{event['home_team']}__{event['away_team']}__{date_str}"


# ══════════════════════════════════════════════════════════════════
# PHẦN 2: Tính movement giữa baseline và current snapshot
# ══════════════════════════════════════════════════════════════════

def remove_vig(h: float, d: float, a: float):
    """Remove vig để có true implied probability."""
    if not all([h, d, a]) or any(x <= 0 for x in [h, d, a]):
        return None, None, None
    ih, id_, ia = 1/h, 1/d, 1/a
    t = ih + id_ + ia
    return ih/t, id_/t, ia/t


def compute_movement(
    baseline: OddsSnapshot,
    current:  OddsSnapshot,
) -> dict:
    """
    So sánh baseline vs current snapshot.
    Trả về dict với movement per side và signal nếu có.
    """
    bh, bd, ba = remove_vig(baseline.odds_h, baseline.odds_d, baseline.odds_a)
    ch, cd, ca = remove_vig(current.odds_h,  current.odds_d,  current.odds_a)

    if not all([bh, bd, ba, ch, cd, ca]):
        return {"signal_side": None}

    move_h = ch - bh
    move_d = cd - bd
    move_a = ca - ba

    def is_signal(move, close_prob, current_odds):
        if move is None or close_prob is None or current_odds is None:
            return False
        return (
            move >= MOVEMENT_THRESHOLD
            and MIN_CLOSE_PROB <= close_prob <= MAX_CLOSE_PROB
            and current_odds >= MIN_OPEN_ODDS
        )

    sig_h = is_signal(move_h, ch, current.odds_h) and "H" in ALLOWED_SIDES
    sig_d = is_signal(move_d, cd, current.odds_d) and "D" in ALLOWED_SIDES
    sig_a = is_signal(move_a, ca, current.odds_a) and "A" in ALLOWED_SIDES

    moves_filtered = {}
    if sig_h: moves_filtered["H"] = move_h
    if sig_d: moves_filtered["D"] = move_d
    if sig_a: moves_filtered["A"] = move_a

    best_side = max(moves_filtered, key=moves_filtered.get) if moves_filtered else None

    return {
        "signal_side": best_side,
        "move_h": round(move_h, 4),
        "move_d": round(move_d, 4),
        "move_a": round(move_a, 4),
        "base_prob_h": round(bh, 4), "base_prob_d": round(bd, 4), "base_prob_a": round(ba, 4),
        "curr_prob_h": round(ch, 4), "curr_prob_d": round(cd, 4), "curr_prob_a": round(ca, 4),
        "movement": round(moves_filtered.get(best_side, 0), 4) if best_side else 0,
    }


# ══════════════════════════════════════════════════════════════════
# PHẦN 3: Kelly sizing cho live bet
# ══════════════════════════════════════════════════════════════════

def kelly_live_stake(
    close_prob: float,
    current_odds: float,
    bankroll: float,
    kelly_fraction: float = 0.25,
) -> float:
    """
    Kelly stake dùng close_prob (estimated từ current snapshot) và current_odds.
    
    Note: trong live context, "close_prob" = implied prob từ current odds.
    Đây là estimate tốt nhất của true prob tại thời điểm đặt.
    """
    b = current_odds - 1.0
    p = close_prob
    q = 1.0 - p
    if b <= 0 or p <= 0:
        return 0.0
    f_full = (b * p - q) / b
    f_frac = max(0.0, f_full * kelly_fraction)
    f_frac = min(f_frac, MAX_BET_PCT)
    if f_frac < MIN_BET_PCT:
        return 0.0
    return round(bankroll * f_frac, 2)


# ══════════════════════════════════════════════════════════════════
# PHẦN 4: Check xem match này đã có live_bet chưa (idempotent)
# ══════════════════════════════════════════════════════════════════

def already_bet(session, match_key: str) -> bool:
    """Tránh đặt bet 2 lần cho cùng 1 trận."""
    return session.query(LiveBet).filter_by(match_key=match_key).first() is not None


def get_current_bankroll(session) -> float:
    """Lấy bankroll hiện tại từ bet gần nhất, hoặc INITIAL nếu chưa có bet nào."""
    last = (
        session.query(LiveBet)
        .order_by(LiveBet.id.desc())
        .first()
    )
    if last and last.bankroll_before is not None and last.stake is not None:
        # bankroll sau bet cuối = before - stake (chưa settle)
        # Dùng bankroll_after nếu đã settle, otherwise bankroll_before
        if last.bankroll_after is not None:
            return last.bankroll_after
        return last.bankroll_before - (last.stake or 0)
    return INITIAL_BANKROLL


# ══════════════════════════════════════════════════════════════════
# PHẦN 5: Core poll cycle
# ══════════════════════════════════════════════════════════════════

def run_poll_cycle(dry_run: bool = False) -> list[dict]:
    """
    1 vòng poll: fetch → snapshot → detect → alert.
    
    Args:
        dry_run: nếu True, không lưu DB, không gửi Telegram.
    
    Returns:
        List các signals đã fire trong vòng này.
    """
    session = get_session()
    fired_signals = []
    now = datetime.now(timezone.utc)

    # Cutoff: chỉ track matches trong SIGNAL_LOOKBACK_DAYS ngày tới
    cutoff = now + timedelta(days=SIGNAL_LOOKBACK_DAYS)

    try:
        for league in TARGET_LEAGUES:
            logger.info(f"[Poller] Polling {league}...")
            raw_events = fetch_live_odds(league)
            parsed = [e for ev in raw_events if (e := parse_odds_event(ev)) is not None]

            if not parsed:
                logger.warning(f"[Poller] Không có events hợp lệ cho {league}")
                continue

            for event in parsed:
                # Bỏ qua matches quá xa tương lai
                if event.get("commence") and event["commence"] > cutoff:
                    continue

                match_key = _make_match_key(event)

                # 1. Lấy baseline snapshot (snapshot đầu tiên của trận này)
                baseline: Optional[OddsSnapshot] = (
                    session.query(OddsSnapshot)
                    .filter_by(match_key=match_key, is_baseline=True)
                    .first()
                )

                if baseline is None:
                    # Lần đầu gặp trận này → tạo baseline
                    if not dry_run:
                        baseline = save_snapshot(session, event, is_baseline=True)
                        session.commit()
                    logger.info(f"[Poller] New baseline: {event['home_team']} vs {event['away_team']} "
                                f"H:{event['odds_h']} D:{event['odds_d']} A:{event['odds_a']}")
                    continue  # Cần ít nhất 1 snapshot tiếp theo để có movement

                # 2. Lưu snapshot hiện tại
                if not dry_run:
                    current_snap = save_snapshot(session, event, is_baseline=False)
                    session.commit()
                else:
                    # Dry run: tạo object tạm để tính movement
                    current_snap = OddsSnapshot(
                        match_key=match_key, league=event["league"],
                        home_team=event["home_team"], away_team=event["away_team"],
                        odds_h=event["odds_h"], odds_d=event["odds_d"], odds_a=event["odds_a"],
                    )

                # 3. Tính movement baseline → current
                mv = compute_movement(baseline, current_snap)

                if mv["signal_side"] is None:
                    logger.debug(
                        f"[Poller] {event['home_team']} vs {event['away_team']} — "
                        f"Δ H:{mv.get('move_h',0):+.3f} "
                        f"D:{mv.get('move_d',0):+.3f} "
                        f"A:{mv.get('move_a',0):+.3f} — No signal"
                    )
                    continue

                # 4. Idempotent: đã bet trận này chưa?
                if already_bet(session, match_key):
                    logger.debug(f"[Poller] {match_key} — signal exists, skip")
                    continue

                # 5. Tính Kelly stake
                side = mv["signal_side"]
                prob_map = {"H": mv["curr_prob_h"], "D": mv["curr_prob_d"], "A": mv["curr_prob_a"]}
                odds_map = {"H": current_snap.odds_h, "D": current_snap.odds_d, "A": current_snap.odds_a}
                curr_prob = prob_map[side]
                curr_odds = odds_map[side]

                bankroll = get_current_bankroll(session)
                stake = kelly_live_stake(curr_prob, curr_odds, bankroll)

                if stake <= 0:
                    logger.info(f"[Poller] {match_key} — signal {side} Kelly=0, skip")
                    continue

                # 6. Log LiveBet
                signal_info = {
                    "match_key":    match_key,
                    "league":       event["league"],
                    "home_team":    event["home_team"],
                    "away_team":    event["away_team"],
                    "match_date":   event.get("commence"),
                    "signal_side":  side,
                    "movement":     mv["movement"],
                    "baseline_odds": {"H": baseline.odds_h, "D": baseline.odds_d, "A": baseline.odds_a}[side],
                    "bet_odds":     curr_odds,
                    "stake":        stake,
                    "bankroll_before": bankroll,
                    "curr_prob":    curr_prob,
                    "signal_fired_at": now,
                }

                if not dry_run:
                    live_bet = LiveBet(
                        match_key       = match_key,
                        league          = event["league"],
                        home_team       = event["home_team"],
                        away_team       = event["away_team"],
                        match_date      = event.get("commence"),
                        signal_side     = side,
                        movement        = mv["movement"],
                        baseline_odds   = signal_info["baseline_odds"],
                        bet_odds        = curr_odds,
                        stake           = stake,
                        bankroll_before = bankroll,
                        signal_fired_at = now,
                        is_paper        = True,  # Paper trading by default
                    )
                    session.add(live_bet)
                    session.commit()

                logger.success(
                    f"[Poller] ★ SIGNAL: {event['home_team']} vs {event['away_team']} "
                    f"→ BET {side} @ {curr_odds:.2f} | "
                    f"Δ{mv['movement']:+.3f} | Stake: {stake:.2f} | "
                    f"Bankroll: {bankroll:.2f}"
                )

                # 7. Gửi Telegram alert (Phase 3)
                try:
                    send_signal_alert(signal_info)
                except Exception as e:
                    logger.warning(f"[Poller] Telegram alert failed: {e}")

                fired_signals.append(signal_info)

    except Exception as e:
        logger.error(f"[Poller] Poll cycle error: {e}", exc_info=True)
    finally:
        session.close()

    logger.info(f"[Poller] Cycle done. Fired {len(fired_signals)} signals.")
    return fired_signals


# ══════════════════════════════════════════════════════════════════
# PHẦN 6: Main loop (scheduler)
# ══════════════════════════════════════════════════════════════════

def run_poller(dry_run: bool = False):
    """
    Vòng lặp vô hạn: poll mỗi POLL_INTERVAL_MINUTES phút.
    Dừng bằng Ctrl+C.
    """
    logger.info(
        f"[Poller] Starting. Interval: {POLL_INTERVAL_MINUTES}m | "
        f"Leagues: {TARGET_LEAGUES} | dry_run={dry_run}"
    )
    while True:
        try:
            run_poll_cycle(dry_run=dry_run)
        except KeyboardInterrupt:
            logger.info("[Poller] Stopped by user.")
            break
        except Exception as e:
            logger.error(f"[Poller] Unhandled error: {e}", exc_info=True)

        logger.info(f"[Poller] Sleeping {POLL_INTERVAL_MINUTES}m...")
        time.sleep(POLL_INTERVAL_MINUTES * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Chạy không lưu DB, không gửi Telegram")
    parser.add_argument("--once", action="store_true",
                        help="Chỉ chạy 1 vòng rồi thoát")
    args = parser.parse_args()

    if args.once:
        run_poll_cycle(dry_run=args.dry_run)
    else:
        run_poller(dry_run=args.dry_run)
