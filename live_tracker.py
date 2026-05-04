"""
live_tracker.py — Real-time Line Movement Tracker
==================================================
Cách hoạt động:
  1. Fetch upcoming odds từ The Odds API
  2. Lần đầu fetch => lưu làm "opening snapshot" vào DB
  3. Các lần sau => so sánh với opening, tính movement
  4. Nếu movement >= threshold => SIGNAL, in alert + lưu CSV
  5. Chạy lặp lại mỗi N phút (mặc định 60 phút)

Chạy:
  python live_tracker.py             # loop mãi, interval 60 phút
  python live_tracker.py --once      # chạy 1 lần rồi thoát
  python live_tracker.py --interval 30  # loop mỗi 30 phút
"""

import sys, time, argparse, json
import requests
import pandas as pd
from datetime import datetime, timezone, timedelta
from pathlib import Path
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from config import ODDS_API_KEY, ODDS_API_BASE, TARGET_LEAGUES, PROCESSED_DATA_DIR
from signals.line_movement import MOVEMENT_THRESHOLD, ALLOWED_SIDES, remove_vig
from data.database import get_session, LiveOdds

# ── Config ────────────────────────────────────────────────────────
REGIONS        = "eu"
MARKETS        = "h2h"
HOURS_AHEAD    = 72          # Chỉ xét trận trong 72h tới
SNAPSHOT_FILE  = PROCESSED_DATA_DIR / "odds_snapshots.json"
SIGNALS_FILE   = PROCESSED_DATA_DIR / "live_signals_lm.csv"


# ── Fetch odds từ API ─────────────────────────────────────────────

def fetch_odds(league: str) -> list[dict]:
    if not ODDS_API_KEY:
        logger.error("ODDS_API_KEY chưa set trong .env!")
        return []
    url = f"{ODDS_API_BASE}/sports/{league}/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": REGIONS,
        "markets": MARKETS,
        "oddsFormat": "decimal",
        "dateFormat": "iso",
    }
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        logger.info(f"[Tracker] {league}: {len(data)} events fetched")
        return data
    except Exception as e:
        logger.error(f"[Tracker] Lỗi fetch {league}: {e}")
        return []


def parse_event(event: dict) -> dict | None:
    """Parse 1 event, lấy odds Bet365 ưu tiên, fallback median tất cả bookmaker"""
    home = event.get("home_team", "")
    away = event.get("away_team", "")
    commence_raw = event.get("commence_time", "")
    try:
        commence = datetime.fromisoformat(commence_raw.replace("Z", "+00:00"))
    except Exception:
        return None

    # Chỉ xét trận trong HOURS_AHEAD giờ tới
    now = datetime.now(timezone.utc)
    if commence < now or commence > now + timedelta(hours=HOURS_AHEAD):
        return None

    # Ưu tiên Bet365 (cùng nguồn với backtest data)
    bet365_odds = None
    all_h, all_d, all_a = [], [], []

    for bm in event.get("bookmakers", []):
        for market in bm.get("markets", []):
            if market.get("key") != "h2h":
                continue
            outcomes = {o["name"]: o["price"] for o in market.get("outcomes", [])}
            h_odd = outcomes.get(home)
            a_odd = outcomes.get(away)
            draw_keys = [k for k in outcomes if k not in [home, away]]
            d_odd = outcomes.get(draw_keys[0]) if draw_keys else None

            if h_odd and d_odd and a_odd:
                all_h.append(h_odd)
                all_d.append(d_odd)
                all_a.append(a_odd)
                if "bet365" in bm.get("key", "").lower():
                    bet365_odds = (h_odd, d_odd, a_odd)

    if not all_h:
        return None

    if bet365_odds:
        odds_h, odds_d, odds_a = bet365_odds
    else:
        import numpy as np
        odds_h = float(np.median(all_h))
        odds_d = float(np.median(all_d))
        odds_a = float(np.median(all_a))

    return {
        "event_id":  event.get("id", ""),
        "league":    event.get("sport_key", ""),
        "home_team": home,
        "away_team": away,
        "commence":  commence.isoformat(),
        "odds_h":    round(odds_h, 3),
        "odds_d":    round(odds_d, 3),
        "odds_a":    round(odds_a, 3),
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }


# ── Snapshot management ───────────────────────────────────────────

def load_snapshots() -> dict:
    """Load opening snapshots từ file JSON"""
    if SNAPSHOT_FILE.exists():
        with open(SNAPSHOT_FILE, "r") as f:
            return json.load(f)
    return {}


def save_snapshots(snapshots: dict):
    SNAPSHOT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SNAPSHOT_FILE, "w") as f:
        json.dump(snapshots, f, indent=2)


# ── Line movement detection ───────────────────────────────────────

def compute_movement(opening: dict, current: dict) -> dict:
    """So sánh opening vs current odds, trả về movement per side"""
    oh, od, oa = remove_vig(opening["odds_h"], opening["odds_d"], opening["odds_a"])
    ch, cd, ca = remove_vig(current["odds_h"], current["odds_d"], current["odds_a"])

    if not all([oh, od, oa, ch, cd, ca]):
        return {}

    return {
        "open_prob_h": oh, "open_prob_d": od, "open_prob_a": oa,
        "close_prob_h": ch, "close_prob_d": cd, "close_prob_a": ca,
        "move_h": round(ch - oh, 4),
        "move_d": round(cd - od, 4),
        "move_a": round(ca - oa, 4),
    }


def check_signals(event_id: str, opening: dict, current: dict) -> list[dict]:
    """
    Kiểm tra xem có sharp signal không.
    Trả về list signals (thường 0 hoặc 1 phần tử).
    """
    mv = compute_movement(opening, current)
    if not mv:
        return []

    signals = []
    for side in ALLOWED_SIDES:
        move_key = f"move_{side.lower()}"
        prob_key = f"close_prob_{side.lower()}"
        odds_key = f"odds_{side.lower()}"

        movement   = mv.get(move_key, 0)
        close_prob = mv.get(prob_key, 0)
        open_odds  = opening.get(odds_key, 0)

        if (movement >= MOVEMENT_THRESHOLD
                and 0.20 <= close_prob <= 0.80
                and open_odds >= 1.30):

            signals.append({
                "event_id":     event_id,
                "league":       current["league"],
                "home_team":    current["home_team"],
                "away_team":    current["away_team"],
                "commence":     current["commence"],
                "signal_side":  side,
                "open_odds":    open_odds,
                "current_odds": current.get(odds_key),
                "movement":     round(movement, 4),
                "close_prob":   round(close_prob, 4),
                "opening_at":   opening["fetched_at"],
                "detected_at":  current["fetched_at"],
            })

    return signals


# ── Alert + Save ──────────────────────────────────────────────────

def print_signal_alert(sig: dict):
    side_label = {"H": "HOME", "D": "DRAW", "A": "AWAY"}[sig["signal_side"]]
    print("\n" + "=" * 60)
    print(f"  *** SHARP MONEY SIGNAL DETECTED ***")
    print(f"  {sig['home_team']} vs {sig['away_team']}")
    print(f"  League  : {sig['league']}")
    print(f"  Kickoff : {sig['commence']}")
    print(f"  Side    : BET {side_label}")
    print(f"  Odds    : {sig['open_odds']} (opening) -> {sig['current_odds']} (now)")
    print(f"  Movement: +{sig['movement']*100:.1f}% implied prob shift")
    print(f"  Action  : BET at {sig['open_odds']} or better ASAP")
    print("=" * 60)
    logger.success(
        f"[SIGNAL] {sig['home_team']} vs {sig['away_team']} | "
        f"{side_label} | move={sig['movement']*100:.1f}% | odds={sig['open_odds']}"
    )


def save_signal(sig: dict):
    """Append signal vào CSV"""
    df = pd.DataFrame([sig])
    header = not SIGNALS_FILE.exists()
    df.to_csv(SIGNALS_FILE, mode="a", header=header, index=False)


# ── Main loop ─────────────────────────────────────────────────────

def run_once() -> list[dict]:
    """Chạy 1 vòng fetch + detect signals"""
    snapshots = load_snapshots()
    all_signals = []
    fired_ids = set()  # Tránh duplicate signal cùng event

    for league in TARGET_LEAGUES:
        raw_events = fetch_odds(league)
        if not raw_events:
            continue

        for raw in raw_events:
            current = parse_event(raw)
            if not current:
                continue

            event_id = current["event_id"]

            if event_id not in snapshots:
                # Lần đầu thấy trận này => lưu làm opening
                snapshots[event_id] = current
                logger.info(
                    f"[Tracker] Opening snapshot: "
                    f"{current['home_team']} vs {current['away_team']} | "
                    f"H:{current['odds_h']} D:{current['odds_d']} A:{current['odds_a']}"
                )
            else:
                # Đã có opening => check movement
                opening = snapshots[event_id]
                signals = check_signals(event_id, opening, current)

                for sig in signals:
                    if event_id not in fired_ids:
                        print_signal_alert(sig)
                        save_signal(sig)
                        all_signals.append(sig)
                        fired_ids.add(event_id)

    # Dọn snapshot của trận đã qua
    now = datetime.now(timezone.utc)
    stale = [eid for eid, snap in snapshots.items()
             if datetime.fromisoformat(snap["commence"]) < now]
    for eid in stale:
        del snapshots[eid]

    save_snapshots(snapshots)

    tracked = len(snapshots)
    logger.info(f"[Tracker] Tracking {tracked} upcoming matches | {len(all_signals)} new signals")
    return all_signals


def run_loop(interval_minutes: int = 60):
    logger.add("logs/live_tracker.log", rotation="10 MB")
    print(f"\n[Live Tracker] Started — checking every {interval_minutes} min")
    print(f"[Live Tracker] Leagues: {TARGET_LEAGUES}")
    print(f"[Live Tracker] Signal threshold: {MOVEMENT_THRESHOLD*100:.0f}% movement | Sides: {ALLOWED_SIDES}")
    print(f"[Live Tracker] Signals saved to: {SIGNALS_FILE}\n")

    while True:
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M')}] Running check...")
        try:
            signals = run_once()
            if not signals:
                print("  No new signals this round.")
        except Exception as e:
            logger.error(f"[Tracker] Error in loop: {e}")

        print(f"  Next check in {interval_minutes} min. Press Ctrl+C to stop.")
        time.sleep(interval_minutes * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--once",     action="store_true", help="Run once then exit")
    parser.add_argument("--interval", type=int, default=60, help="Minutes between checks")
    args = parser.parse_args()

    if args.once:
        logger.add("logs/live_tracker.log", rotation="10 MB")
        signals = run_once()
        print(f"\nDone. {len(signals)} signal(s) detected.")
        if signals:
            print(pd.DataFrame(signals)[
                ["home_team","away_team","signal_side","open_odds","movement","commence"]
            ].to_string(index=False))
    else:
        run_loop(args.interval)
