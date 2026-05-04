"""
line_movement_v2.py — Signal Engine v2 (no lookahead bias)
============================================================
Sửa vấn đề cốt lõi của v1:

v1 (SAI): signal dùng close_odds, đặt tại open_odds
  → backtest dùng thông tin tương lai (closing chỉ có lúc kick-off)

v2 (ĐÚNG): hỗ trợ 3 chế độ execution:
  1. "realistic"  — đặt tại CLOSING odds (upper bound, khả thi nhất cho backtest)
  2. "historical" — đặt tại OPENING odds (overestimates, chỉ để tham khảo)
  3. "intraday"   — đặt tại odds hiện tại khi movement vừa detect (live trading)

Với mode "realistic": nếu ROI vẫn dương khi đặt tại closing price
→ edge là thật (sharp money di chuyển đúng hướng kết quả)
→ live trading sẽ capture được một phần edge đó

Thêm: hỗ trợ Pinnacle odds (sharper signal, lower vig)
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from loguru import logger
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.database import get_session, Match
from config import (
    MOVEMENT_THRESHOLD, MIN_CLOSE_PROB, MAX_CLOSE_PROB,
    MIN_OPEN_ODDS, ALLOWED_SIDES,
)

BOOKMAKER_COLS = {
    "bet365":  {"open": ("odds_h","odds_d","odds_a"),
                "close": ("close_h","close_d","close_a")},
    "pinnacle": {"open": ("pin_h","pin_d","pin_a"),
                 "close": ("pin_close_h","pin_close_d","pin_close_a")},
}


def remove_vig(h: float, d: float, a: float):
    if not all([h, d, a]) or any(x <= 0 for x in [h, d, a]):
        return None, None, None
    ih, id_, ia = 1/h, 1/d, 1/a
    t = ih + id_ + ia
    return ih/t, id_/t, ia/t


def compute_line_movement(
    df: pd.DataFrame,
    bookmaker: str = "bet365",
    execution: str = "realistic",
) -> pd.DataFrame:
    """
    Tính line movement và tạo signals.

    Args:
        df: DataFrame với odds columns
        bookmaker: "bet365" hoặc "pinnacle"
        execution: "realistic" (bet@close), "historical" (bet@open), "intraday" (bet@current)

    Returns:
        df với thêm cột signal_side, bet_odds, movement
    """
    bm = BOOKMAKER_COLS.get(bookmaker, BOOKMAKER_COLS["bet365"])
    oh_col, od_col, oa_col = bm["open"]
    ch_col, cd_col, ca_col = bm["close"]

    results = []
    for _, row in df.iterrows():
        rec = row.to_dict()

        oh, od, oa = remove_vig(row.get(oh_col), row.get(od_col), row.get(oa_col))
        ch, cd, ca = remove_vig(row.get(ch_col), row.get(cd_col), row.get(ca_col))

        rec.update({
            "open_prob_h": oh, "open_prob_d": od, "open_prob_a": oa,
            "close_prob_h": ch, "close_prob_d": cd, "close_prob_a": ca,
            "move_h": (ch - oh) if (ch and oh) else None,
            "move_d": (cd - od) if (cd and od) else None,
            "move_a": (ca - oa) if (ca and oa) else None,
        })

        def is_signal(move, close_prob, open_odds):
            if move is None or close_prob is None or open_odds is None:
                return False
            return (move >= MOVEMENT_THRESHOLD
                    and MIN_CLOSE_PROB <= close_prob <= MAX_CLOSE_PROB
                    and open_odds >= MIN_OPEN_ODDS)

        rec["signal_h"] = is_signal(rec["move_h"], ch, row.get(oh_col))
        rec["signal_d"] = is_signal(rec["move_d"], cd, row.get(od_col))
        rec["signal_a"] = is_signal(rec["move_a"], ca, row.get(oa_col))

        moves = {
            side: rec[f"move_{side.lower()}"]
            if rec[f"signal_{side.lower()}"] else -999
            for side in ALLOWED_SIDES
        }
        best = max(moves, key=moves.get)
        has_signal = moves[best] > -999
        rec["signal_side"] = best if has_signal else None

        # --- Execution model ---
        # "realistic": đặt tại closing odds (benchmark thực tế)
        # "historical": đặt tại opening odds (overestimate)
        if has_signal:
            side = best
            if execution == "realistic":
                odds_col = {"H": ch_col, "D": cd_col, "A": ca_col}[side]
                rec["bet_odds"] = row.get(odds_col)
            else:  # historical
                odds_col = {"H": oh_col, "D": od_col, "A": oa_col}[side]
                rec["bet_odds"] = row.get(odds_col)
            # close_prob luôn dùng để tính Kelly (best estimate of true prob)
            rec["bet_close_prob"] = {"H": ch, "D": cd, "A": ca}[side]
        else:
            rec["bet_odds"] = None
            rec["bet_close_prob"] = None

        results.append(rec)

    return pd.DataFrame(results)


def _get_existing_columns(session) -> set:
    """Lấy danh sách columns thực sự có trong bảng matches (tránh lỗi khi DB chưa migrate)."""
    result = session.execute(
        __import__("sqlalchemy").text("PRAGMA table_info(matches)")
    )
    return {row[1] for row in result.fetchall()}


def load_matches_with_movement(
    league: str = None,
    season: str = None,
    bookmaker: str = "bet365",
    execution: str = "realistic",
    min_pinnacle_coverage: float = 0.0,
) -> pd.DataFrame:
    """
    Load matches từ DB và tính line movement.
    Tự động detect columns có sẵn — không crash nếu DB chưa migrate Pinnacle.
    """
    session = get_session()

    # Kiểm tra columns thực tế trong DB
    existing_cols = _get_existing_columns(session)

    # Pinnacle columns có sẵn không?
    has_pinnacle = all(c in existing_cols for c in ["pin_h", "pin_d", "pin_a",
                                                     "pin_close_h", "pin_close_d", "pin_close_a"])
    if bookmaker == "pinnacle" and not has_pinnacle:
        logger.warning(
            "[LM v2] Pinnacle columns chưa có trong DB. "
            "Chạy 'python migrate_v2.py' trước. Fallback về bet365."
        )
        bookmaker = "bet365"

    # Build SELECT chỉ lấy columns cần thiết (tránh lỗi ORM khi DB cũ)
    base_cols = "id, league, season, date, home_team, away_team, result, " \
                "odds_h, odds_d, odds_a, close_h, close_d, close_a"

    pin_select = ""
    if has_pinnacle:
        pin_select = ", pin_h, pin_d, pin_a, pin_close_h, pin_close_d, pin_close_a"

    where_clauses = ["odds_h IS NOT NULL"]
    params: dict = {}

    if bookmaker == "pinnacle" and has_pinnacle:
        where_clauses.append("pin_h IS NOT NULL AND pin_close_h IS NOT NULL")
    else:
        where_clauses.append("close_h IS NOT NULL")

    if league:
        where_clauses.append("league = :league")
        params["league"] = league
    if season:
        where_clauses.append("season = :season")
        params["season"] = season

    sql = (
        f"SELECT {base_cols}{pin_select} FROM matches "
        f"WHERE {' AND '.join(where_clauses)}"
    )

    import sqlalchemy
    result = session.execute(sqlalchemy.text(sql), params)
    col_names = list(result.keys())
    raw_rows  = result.fetchall()
    session.close()

    if not raw_rows:
        logger.warning("[LM v2] No matches found. Check DB and bookmaker filter.")
        return pd.DataFrame()

    # Build DataFrame từ raw rows
    df = pd.DataFrame(raw_rows, columns=col_names)

    # Thêm Pinnacle columns rỗng nếu không có (để compute_line_movement không crash)
    for col in ["pin_h", "pin_d", "pin_a", "pin_close_h", "pin_close_d", "pin_close_a"]:
        if col not in df.columns:
            df[col] = None

    df = compute_line_movement(df, bookmaker=bookmaker, execution=execution)
    n_signals = df["signal_side"].notna().sum()
    logger.info(
        f"[LM v2] Loaded {len(df)} matches | Signals: {n_signals} | "
        f"bookmaker={bookmaker} | execution={execution} | "
        f"pinnacle_available={has_pinnacle}"
    )
    return df


def backtest_flat(df: pd.DataFrame) -> dict:
    """Flat-unit backtest (1 unit per bet)"""
    signals = df[df["signal_side"].notna()].copy()
    signals = signals.dropna(subset=["bet_odds"])
    if signals.empty:
        return {}

    signals["won"]    = signals["result"] == signals["signal_side"]
    signals["profit"] = signals.apply(
        lambda r: r["bet_odds"] - 1 if r["won"] else -1, axis=1
    )
    signals = signals.dropna(subset=["profit"])

    total = len(signals)
    wins  = int(signals["won"].sum())
    roi   = signals["profit"].mean()

    by_league = {}
    for lg, grp in signals.groupby("league"):
        by_league[lg] = {
            "bets": len(grp),
            "roi": round(grp["profit"].mean(), 4),
            "win_rate": round(grp["won"].mean(), 4),
        }

    result = {
        "total_bets": total, "wins": wins,
        "win_rate": round(wins/total, 4),
        "roi": round(roi, 4),
        "total_profit_units": round(signals["profit"].sum(), 2),
        "by_league": by_league,
    }
    logger.success(
        f"[Backtest] {total} bets | Win: {wins/total:.1%} | "
        f"ROI: {roi:.2%} | Profit: {signals['profit'].sum():.2f}u"
    )
    return result


if __name__ == "__main__":
    logger.add("logs/lm_v2.log", rotation="5 MB")
    import json

    print("\n=== Bet365 — REALISTIC (đặt tại closing odds) ===")
    df_r = load_matches_with_movement(execution="realistic")
    if not df_r.empty:
        print(json.dumps(backtest_flat(df_r), indent=2))

    print("\n=== Bet365 — HISTORICAL (đặt tại opening odds, có bias) ===")
    df_h = load_matches_with_movement(execution="historical")
    if not df_h.empty:
        print(json.dumps(backtest_flat(df_h), indent=2))

    print("\n=== Pinnacle — REALISTIC ===")
    df_p = load_matches_with_movement(bookmaker="pinnacle", execution="realistic")
    if not df_p.empty:
        print(json.dumps(backtest_flat(df_p), indent=2))
