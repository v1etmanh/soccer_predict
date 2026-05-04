"""
line_movement.py — Sharp Money Signal Engine
=============================================
Logic cốt lõi của Hướng B:
  - Closing odds = "wisdom of sharp money" (implied true probability)
  - Opening odds > Closing odds => public money đẩy line ngược chiều
  - Tìm khi nào public bet NGƯỢC chiều sharp => fade the public
  
Signal: opening_implied_prob < closing_implied_prob
  => thị trường nghĩ outcome này có prob cao hơn ban đầu
  => sharp money đã đẩy odds xuống (tức đặt vào outcome đó)
  => ĐÂY LÀ BÊT THEO SHARP (value bet)

Công thức:
  implied_prob = 1 / odds  (trước khi remove vig)
  movement    = close_implied - open_implied
  signal      = movement > threshold  (sharp đặt mạnh vào đây)
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from loguru import logger
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.database import get_session, Match
from config import TARGET_LEAGUES


# ── Cấu hình ngưỡng ─────────────────────────────────────────────
MOVEMENT_THRESHOLD = 0.03   # ≥3% shift trong implied prob => sharp signal
MIN_CLOSE_PROB     = 0.20   # Bỏ qua kết quả quá khó (prob < 20%)
MAX_CLOSE_PROB     = 0.80   # Bỏ qua kết quả quá dễ (vig không đáng)
MIN_OPEN_ODDS      = 1.30   # Odds tối thiểu để có value
ALLOWED_SIDES      = {"A"}       # S6 config: Away-only — EPL bootstrap CI confirmed [+0.1%, +41.0%]
                                 # Draw signal loại vì n=23 quá nhỏ (CI [-20.5%, +165.3%] vô nghĩa)
                                 # Bundesliga loại qua TARGET_LEAGUES=soccer_epl trong .env


def remove_vig(h: float, d: float, a: float) -> tuple[float, float, float]:
    """Loại bỏ vig để có true implied probability"""
    if not all([h, d, a]):
        return None, None, None
    ih, id_, ia = 1/h, 1/d, 1/a
    total = ih + id_ + ia
    return ih/total, id_/total, ia/total


def compute_line_movement(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tính line movement cho DataFrame có cột:
      odds_h/d/a (opening) và close_h/d/a (closing)
    
    Trả về df với thêm các cột:
      open_prob_h/d/a   : implied prob từ opening (no-vig)
      close_prob_h/d/a  : implied prob từ closing (no-vig)
      move_h/d/a        : = close_prob - open_prob
      signal_h/d/a      : bool — sharp money signal
      signal_side       : 'H'/'D'/'A'/None — best signal outcome
    """
    results = []
    for _, row in df.iterrows():
        rec = row.to_dict()

        # Opening no-vig probs
        oh, od, oa = remove_vig(row.get("odds_h"), row.get("odds_d"), row.get("odds_a"))
        # Closing no-vig probs
        ch, cd, ca = remove_vig(row.get("close_h"), row.get("close_d"), row.get("close_a"))

        rec["open_prob_h"] = oh
        rec["open_prob_d"] = od
        rec["open_prob_a"] = oa
        rec["close_prob_h"] = ch
        rec["close_prob_d"] = cd
        rec["close_prob_a"] = ca

        # Movement = closing - opening (dương => sharp đẩy vào outcome này)
        rec["move_h"] = (ch - oh) if (ch and oh) else None
        rec["move_d"] = (cd - od) if (cd and od) else None
        rec["move_a"] = (ca - oa) if (ca and oa) else None

        # Signal: movement vượt threshold VÀ closing prob trong range hợp lý
        def is_signal(move, close_prob, open_odds):
            if move is None or close_prob is None:
                return False
            return (
                move >= MOVEMENT_THRESHOLD
                and MIN_CLOSE_PROB <= close_prob <= MAX_CLOSE_PROB
                and (open_odds or 0) >= MIN_OPEN_ODDS
            )

        rec["signal_h"] = is_signal(rec["move_h"], ch, row.get("odds_h"))
        rec["signal_d"] = is_signal(rec["move_d"], cd, row.get("odds_d"))
        rec["signal_a"] = is_signal(rec["move_a"], ca, row.get("odds_a"))

        # Chọn signal mạnh nhất trong trận — chỉ các side được phép
        moves = {
            side: rec[f"move_{side.lower()}"] if rec[f"signal_{side.lower()}"] else -999
            for side in ALLOWED_SIDES
        }
        best = max(moves, key=moves.get)
        rec["signal_side"] = best if moves[best] > -999 else None

        results.append(rec)

    return pd.DataFrame(results)


def load_matches_with_movement(league: str = None, season: str = None) -> pd.DataFrame:
    """Load matches từ DB và tính line movement.
    Tự động filter theo TARGET_LEAGUES từ .env (trừ khi `league` được truyền tường minh).
    """
    session = get_session()
    q = session.query(Match).filter(
        Match.close_h.isnot(None),
        Match.odds_h.isnot(None),
    )
    if league:
        # Caller truyền league cụ thể — dùng đúng league đó
        q = q.filter(Match.league == league)
    elif TARGET_LEAGUES:
        # Dùng danh sách từ .env (S6: chỉ soccer_epl)
        q = q.filter(Match.league.in_(TARGET_LEAGUES))
    if season:
        q = q.filter(Match.season == season)

    rows = []
    for m in q.all():
        rows.append({
            "match_id":  m.id,
            "league":    m.league,
            "season":    m.season,
            "date":      m.date,
            "home_team": m.home_team,
            "away_team": m.away_team,
            "result":    m.result,
            "odds_h":    m.odds_h,
            "odds_d":    m.odds_d,
            "odds_a":    m.odds_a,
            "close_h":   m.close_h,
            "close_d":   m.close_d,
            "close_a":   m.close_a,
        })
    session.close()

    if not rows:
        logger.warning("[LineMovement] Không có trận nào có closing odds. Chạy migrate + re-scrape trước.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = compute_line_movement(df)
    logger.info(f"[LineMovement] Loaded {len(df)} trận, "
                f"{df['signal_side'].notna().sum()} có sharp signal")
    return df


def backtest_line_movement(df: pd.DataFrame = None) -> dict:
    """
    Backtest đơn giản: nếu đặt vào tất cả sharp signal,
    ROI và win rate như thế nào?
    """
    if df is None:
        df = load_matches_with_movement()
    if df.empty:
        return {}

    signals = df[df["signal_side"].notna()].copy()
    if signals.empty:
        logger.warning("[Backtest] Không có signal nào.")
        return {}

    # Lấy odds và kết quả theo signal_side
    def get_bet_info(row):
        side = row["signal_side"]
        odds_map  = {"H": row.get("odds_h"), "D": row.get("odds_d"), "A": row.get("odds_a")}
        won = row.get("result") == side
        return odds_map.get(side), won

    signals[["bet_odds", "won"]] = signals.apply(
        lambda r: pd.Series(get_bet_info(r)), axis=1
    )
    signals = signals.dropna(subset=["bet_odds"])
    signals["profit"] = signals.apply(
        lambda r: r["bet_odds"] - 1 if r["won"] else -1, axis=1
    )

    total    = len(signals)
    wins     = signals["won"].sum()
    roi      = signals["profit"].sum() / total
    win_rate = wins / total

    by_league = signals.groupby("league").agg(
        bets=("profit", "count"),
        roi=("profit", "mean"),
        win_rate=("won", "mean"),
    ).round(4).to_dict("index")

    result = {
        "total_bets": total,
        "wins": int(wins),
        "win_rate": round(win_rate, 4),
        "roi": round(roi, 4),
        "total_profit_units": round(signals["profit"].sum(), 2),
        "by_league": by_league,
    }

    logger.success(
        f"[Backtest] {total} bets | Win: {win_rate:.1%} | ROI: {roi:.2%} | "
        f"Profit: {signals['profit'].sum():.2f}u"
    )
    return result


if __name__ == "__main__":
    logger.add("logs/line_movement.log", rotation="5 MB")
    df   = load_matches_with_movement()
    stats = backtest_line_movement(df)
    import json
    print(json.dumps(stats, indent=2))
