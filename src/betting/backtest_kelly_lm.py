"""
backtest_kelly_lm.py
====================
Backtest line movement + Kelly sizing
- Edge = close_prob (closing odds = true probability)
- Stake = Fractional Kelly dua tren edge do
- Cap toi da 3% bankroll / bet de bao ve von
"""
import sys, json
import pandas as pd
from pathlib import Path
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from signals.line_movement import load_matches_with_movement

# ── Config ───────────────────────────────────────────────────────
INITIAL_BANKROLL = 1000.0   # USD / don vi ban dau
KELLY_FRACTION   = 0.25     # 1/4 Kelly - an toan hon full Kelly
MAX_BET_PCT      = 0.03     # Cap toi da 3% bankroll / 1 bet
MIN_BET_PCT      = 0.005    # Toi thieu 0.5% (bo qua bet qua nho)


def kelly_stake(close_prob: float, open_odds: float, bankroll: float) -> float:
    """
    Tinh stake theo Fractional Kelly.
    close_prob = xac suat that (tu closing odds, da remove vig)
    open_odds  = odds minh dat (opening - truoc khi sharp push)
    """
    b = open_odds - 1.0          # net profit neu thang
    p = close_prob               # xac suat thang (theo closing)
    q = 1.0 - p
    if b <= 0 or p <= 0:
        return 0.0
    f_full = (b * p - q) / b    # Full Kelly fraction
    f_frac = f_full * KELLY_FRACTION  # Fractional Kelly
    f_frac = max(0.0, f_frac)
    f_frac = min(f_frac, MAX_BET_PCT) # Hard cap
    if f_frac < MIN_BET_PCT:
        return 0.0               # Bo qua neu bet qua nho
    return round(bankroll * f_frac, 2)


def run_kelly_backtest(df: pd.DataFrame = None) -> tuple[pd.DataFrame, dict]:
    if df is None:
        df = load_matches_with_movement()

    signals = df[df["signal_side"].notna()].copy().sort_values("date")

    bankroll = INITIAL_BANKROLL
    history  = [bankroll]
    records  = []

    for _, row in signals.iterrows():
        side = row["signal_side"]

        odds_map      = {"H": row.get("odds_h"),      "D": row.get("odds_d"),      "A": row.get("odds_a")}
        close_prob_map = {"H": row.get("close_prob_h"), "D": row.get("close_prob_d"), "A": row.get("close_prob_a")}

        open_odds  = odds_map.get(side)
        close_prob = close_prob_map.get(side)
        result     = row.get("result")

        if not open_odds or not close_prob:
            continue

        stake = kelly_stake(close_prob, open_odds, bankroll)
        if stake <= 0:
            continue

        won    = (result == side)
        profit = round(stake * (open_odds - 1) if won else -stake, 2)
        bankroll = round(bankroll + profit, 2)
        history.append(bankroll)

        records.append({
            "date":       row["date"],
            "season":     row["season"],
            "league":     row["league"],
            "home_team":  row["home_team"],
            "away_team":  row["away_team"],
            "side":       side,
            "open_odds":  open_odds,
            "close_prob": round(close_prob, 4),
            "stake":      stake,
            "won":        won,
            "profit":     profit,
            "bankroll":   bankroll,
        })

    results_df = pd.DataFrame(records)

    # Summary
    peak    = max(history)
    max_dd  = min(
        (h - max(history[:i+1])) / max(history[:i+1]) * 100
        for i, h in enumerate(history)
    )
    summary = {
        "initial_bankroll": INITIAL_BANKROLL,
        "final_bankroll":   bankroll,
        "total_profit":     round(bankroll - INITIAL_BANKROLL, 2),
        "roi_pct":          round((bankroll - INITIAL_BANKROLL) / INITIAL_BANKROLL * 100, 2),
        "total_bets":       len(results_df),
        "wins":             int(results_df["won"].sum()),
        "win_rate":         round(results_df["won"].mean() * 100, 1),
        "peak_bankroll":    round(peak, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "avg_stake":        round(results_df["stake"].mean(), 2),
        "avg_odds":         round(results_df["open_odds"].mean(), 2),
    }

    logger.success(
        f"[Kelly Backtest] {summary['total_bets']} bets | "
        f"Win: {summary['win_rate']}% | "
        f"ROI: {summary['roi_pct']}% | "
        f"Bankroll: {INITIAL_BANKROLL} -> {bankroll} | "
        f"MaxDD: {summary['max_drawdown_pct']}%"
    )
    return results_df, summary
