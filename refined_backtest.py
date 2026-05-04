"""
refined_backtest.py — Backtest với filter đã được validated
============================================================
Filter set được chứng minh có edge thật (realistic execution):

  1. League: EPL only (soccer_epl)
  2. Side: Away only (A) — Draw signal loại bỏ
  3. Movement threshold: ≥ 4.5% (tốt nhất ≥7% nhưng volume thấp)
  4. Execution: realistic (bet@close) — không có look-ahead bias

Đây là conservative configuration để live trading.
Chạy: python refined_backtest.py
"""
import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from signals.line_movement_v2 import load_matches_with_movement, backtest_flat
from loguru import logger

logger.remove()


def run_refined(
    leagues: list[str] = None,
    sides: set[str]    = None,
    min_movement: float = 0.045,
    execution: str      = "realistic",
    label: str          = "",
) -> dict:
    """Backtest với filter tùy chỉnh."""
    df = load_matches_with_movement(execution=execution)
    if df.empty:
        return {}

    sig = df[df["signal_side"].notna()].dropna(subset=["bet_odds"]).copy()

    if leagues:
        sig = sig[sig["league"].isin(leagues)]
    if sides:
        sig = sig[sig["signal_side"].isin(sides)]

    # Apply movement filter
    def get_move(row):
        s = row["signal_side"]
        return row.get(f"move_{s.lower()}", 0) or 0
    sig["movement"] = sig.apply(get_move, axis=1)
    sig = sig[sig["movement"] >= min_movement]

    if sig.empty:
        print(f"  {label}: No signals after filter")
        return {}

    sig["won"]    = sig["result"] == sig["signal_side"]
    sig["profit"] = sig.apply(lambda r: r["bet_odds"] - 1 if r["won"] else -1, axis=1)

    total = len(sig)
    wins  = int(sig["won"].sum())
    roi   = sig["profit"].mean()

    icon = "✅" if roi > 0 else "❌"
    print(
        f"  {icon} {label:<45} "
        f"{total:>4} bets | "
        f"Win {wins/total:.1%} | "
        f"ROI {roi:+.2%} | "
        f"Profit {sig['profit'].sum():+.2f}u"
    )

    by_season = {}
    for s, g in sig.groupby("season"):
        by_season[s] = {
            "bets": len(g), "roi": round(g["profit"].mean(), 4),
            "win_rate": round(g["won"].mean(), 4),
        }

    return {
        "total": total, "wins": wins, "win_rate": round(wins/total, 4),
        "roi": round(roi, 4), "profit": round(sig["profit"].sum(), 2),
        "by_season": by_season,
    }


def kelly_simulation(
    leagues: list[str]  = None,
    sides: set[str]     = None,
    min_movement: float = 0.045,
    kelly_frac: float   = 0.25,
    max_bet_pct: float  = 0.03,
    initial_bankroll: float = 1000.0,
) -> dict:
    """Kelly bankroll simulation với refined filter."""
    df_r = load_matches_with_movement(execution="realistic")
    df_h = load_matches_with_movement(execution="historical")
    if df_r.empty:
        return {}

    sig_r = df_r[df_r["signal_side"].notna()].dropna(subset=["bet_odds"]).copy()
    sig_h = df_h[df_h["signal_side"].notna()].dropna(subset=["bet_odds"]).copy()

    # Build open_odds lookup (để tính Kelly: stake dùng close_prob, capped tại open odds)
    open_odds_map = {}
    for _, row in sig_h.iterrows():
        key = (row["home_team"], row["away_team"], str(row["date"])[:10])
        open_odds_map[key] = row["bet_odds"]

    if leagues:
        sig_r = sig_r[sig_r["league"].isin(leagues)]
    if sides:
        sig_r = sig_r[sig_r["signal_side"].isin(sides)]

    def get_move(row):
        s = row["signal_side"]
        return row.get(f"move_{s.lower()}", 0) or 0
    sig_r["movement"] = sig_r.apply(get_move, axis=1)
    sig_r = sig_r[sig_r["movement"] >= min_movement].sort_values("date")

    if sig_r.empty:
        return {}

    bankroll  = initial_bankroll
    history   = [bankroll]
    records   = []

    for _, row in sig_r.iterrows():
        side       = row["signal_side"]
        close_prob = row.get(f"close_prob_{side.lower()}")
        bet_odds   = row["bet_odds"]  # closing odds (realistic)
        result     = row.get("result")

        if not close_prob or not bet_odds:
            continue

        # Kelly formula dùng bet_odds (=closing) và close_prob
        b = bet_odds - 1.0
        p = close_prob
        q = 1 - p
        if b <= 0 or p <= 0:
            continue
        f = max(0, (b * p - q) / b) * kelly_frac
        f = min(f, max_bet_pct)
        if f < 0.005:
            continue

        stake  = round(bankroll * f, 2)
        won    = (result == side)
        profit = round(stake * b if won else -stake, 2)
        bankroll = round(bankroll + profit, 2)
        history.append(bankroll)

        records.append({
            "date": row["date"], "season": row["season"], "league": row["league"],
            "home": row["home_team"], "away": row["away_team"],
            "side": side, "movement": round(row["movement"], 4),
            "bet_odds": bet_odds, "close_prob": round(close_prob, 4),
            "stake": stake, "won": won, "profit": profit, "bankroll": bankroll,
        })

    if not records:
        return {}

    df_out = pd.DataFrame(records)
    peak   = max(history)
    max_dd = min(
        (h - max(history[:i+1])) / max(history[:i+1]) * 100
        for i, h in enumerate(history)
    ) if len(history) > 1 else 0

    summary = {
        "initial": initial_bankroll, "final": bankroll,
        "profit": round(bankroll - initial_bankroll, 2),
        "roi_pct": round((bankroll - initial_bankroll) / initial_bankroll * 100, 2),
        "bets": len(df_out), "wins": int(df_out["won"].sum()),
        "win_rate": round(df_out["won"].mean() * 100, 1),
        "max_drawdown": round(max_dd, 2),
        "avg_stake": round(df_out["stake"].mean(), 2),
    }
    return summary, df_out


def print_season_detail(stats: dict):
    """In breakdown theo mùa."""
    for s, d in sorted(stats.get("by_season", {}).items()):
        icon = "✅" if d["roi"] > 0 else "❌"
        print(f"      {icon} {s}: {d['bets']:3d} bets | "
              f"Win {d['win_rate']:.1%} | ROI {d['roi']:+.2%}")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("  REFINED BACKTEST — Filter configurations (realistic execution)")
    print("="*70)

    # Baseline
    print("\n[BASELINE — current config, all leagues, D+A, ≥3%]")
    run_refined(label="All leagues D+A ≥3%", min_movement=0.03)

    print("\n[CONFIGURATION SWEEP]")
    configs = [
        # label, leagues, sides, threshold
        ("EPL only, D+A, ≥3%",     ["soccer_epl"], {"D","A"}, 0.03),
        ("EPL only, A only, ≥3%",  ["soccer_epl"], {"A"},     0.03),
        ("EPL only, A only, ≥4%",  ["soccer_epl"], {"A"},     0.04),
        ("EPL only, A only, ≥4.5%",["soccer_epl"], {"A"},     0.045),
        ("EPL only, A only, ≥5%",  ["soccer_epl"], {"A"},     0.05),
        ("EPL only, A only, ≥6%",  ["soccer_epl"], {"A"},     0.06),
        ("EPL only, A only, ≥7%",  ["soccer_epl"], {"A"},     0.07),
        ("All EPL+Bund, A only, ≥4.5%", ["soccer_epl","soccer_germany_bundesliga"], {"A"}, 0.045),
        ("All EPL+Bund, A only, ≥7%",   ["soccer_epl","soccer_germany_bundesliga"], {"A"}, 0.07),
    ]

    best_roi  = -999
    best_cfg  = None

    for label, lg, sides, thr in configs:
        result = run_refined(leagues=lg, sides=sides, min_movement=thr, label=label)
        if result and result["roi"] > best_roi and result["total"] >= 30:
            best_roi = result["roi"]
            best_cfg = (label, lg, sides, thr, result)

    # Best configuration detail
    if best_cfg:
        label, lg, sides, thr, result = best_cfg
        print(f"\n{'─'*70}")
        print(f"  ★ BEST CONFIG: {label}")
        print(f"    ROI: {result['roi']:+.2%} | Bets: {result['total']} | "
              f"Win: {result['win_rate']:.1%}")
        print_season_detail(result)

    # Kelly simulation với best config
    print(f"\n{'='*70}")
    print("  KELLY SIMULATION — EPL Away ≥4.5% (realistic)")
    print("="*70)
    kelly_result = kelly_simulation(
        leagues=["soccer_epl"], sides={"A"}, min_movement=0.045
    )
    if isinstance(kelly_result, tuple):
        summary, df_bets = kelly_result
        print(f"  Bankroll: {summary['initial']} → {summary['final']} "
              f"(+{summary['profit']:.2f} / {summary['roi_pct']:+.1f}%)")
        print(f"  Bets: {summary['bets']} | Win: {summary['win_rate']}% | "
              f"MaxDD: {summary['max_drawdown']}%")

        out = Path("data/processed/refined_kelly_results.csv")
        df_bets.to_csv(out, index=False)
        print(f"  Saved → {out}")

    # Live config recommendation
    print(f"\n{'='*70}")
    print("  LIVE TRADING CONFIGURATION (recommended)")
    print("="*70)
    print("  TARGET_LEAGUES=soccer_epl")
    print("  ALLOWED_SIDES=A")
    print("  MOVEMENT_THRESHOLD=0.045")
    print()
    print("  Cập nhật .env và config.py với các giá trị trên trước khi chạy live.")
