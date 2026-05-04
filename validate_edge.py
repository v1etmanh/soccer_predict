"""
validate_edge.py — Kiểm chứng edge có thật không
===================================================
Test đơn giản nhất để biết edge là thật hay look-ahead bias:

v1 (suspicious): signal từ close_odds, đặt tại OPEN_ODDS (opening)
v2 realistic:    signal từ close_odds, đặt tại CLOSE_ODDS (closing)

Nếu v2 (realistic) vẫn dương → edge thật → live trading có thể capture
Nếu v2 âm → edge chỉ là "tôi biết trước closing" → cần re-evaluate

Chạy: python validate_edge.py
"""
import sys, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import pandas as pd
from loguru import logger
from signals.line_movement_v2 import (
    load_matches_with_movement, backtest_flat,
)

logger.add("logs/validate_edge.log", rotation="5 MB")


def compare_execution_modes():
    """So sánh ROI khi đặt tại opening vs closing odds."""
    print("\n" + "="*60)
    print("  EDGE VALIDATION: Opening vs Closing execution")
    print("="*60)

    # Mode 1: đặt tại opening (v1 behavior — suspicious)
    print("\n[1/2] Loading: HISTORICAL mode (bet @ opening odds)...")
    df_hist = load_matches_with_movement(execution="historical")
    if df_hist.empty:
        print("  ✗ No data. Run scraper + update_closing_odds first.")
        return

    stats_hist = backtest_flat(df_hist)

    # Mode 2: đặt tại closing (lower bound — realistic benchmark)
    print("\n[2/2] Loading: REALISTIC mode (bet @ closing odds)...")
    df_real = load_matches_with_movement(execution="realistic")
    stats_real = backtest_flat(df_real)

    print("\n" + "-"*60)
    print(f"  {'Mode':<25} {'Bets':>6} {'WinRate':>8} {'ROI':>8} {'Profit':>8}")
    print("-"*60)

    def row(label, s):
        if not s:
            return f"  {label:<25} {'N/A':>6}"
        return (
            f"  {label:<25} "
            f"{s['total_bets']:>6} "
            f"{s['win_rate']:>8.1%} "
            f"{s['roi']:>+8.2%} "
            f"{s['total_profit_units']:>+8.2f}u"
        )

    print(row("HISTORICAL (bet@open)", stats_hist))
    print(row("REALISTIC  (bet@close)", stats_real))
    print("-"*60)

    # Verdict
    print()
    if stats_real and stats_real["roi"] > 0:
        print("  ✅ VERDICT: Edge là THẬT")
        print(f"     ROI vẫn dương (+{stats_real['roi']:.2%}) khi đặt tại closing odds.")
        print("     → Signal detect đúng hướng sharp money.")
        print("     → Live trading có thể capture phần edge này nếu vào đủ sớm.")
        diff = (stats_hist.get("roi", 0) - stats_real["roi"]) if stats_hist else 0
        print(f"     → 'Early entry bonus' (open vs close spread): ~{diff:+.2%}")
    elif stats_real and stats_real["roi"] > -0.02:
        print("  ⚠️  VERDICT: Edge YẾU ở closing odds")
        print("     ROI gần break-even. Cần early execution để có lãi.")
        print("     → Live trading PHẢI vào trước khi odds squeeze mạnh.")
    else:
        print("  ❌ VERDICT: Edge KHÔNG THẬT hoặc đã mất")
        print("     ROI âm ở closing odds → strategy cần re-evaluate.")

    # By league breakdown
    if stats_real and stats_real.get("by_league"):
        print("\n  By League (realistic):")
        for lg, s in stats_real["by_league"].items():
            sign = "✅" if s["roi"] > 0 else "❌"
            print(f"    {sign} {lg:35s} {s['bets']:3d} bets | ROI {s['roi']:+.1%}")

    print()
    return {"historical": stats_hist, "realistic": stats_real}


def season_consistency_check(execution: str = "realistic"):
    """ROI per season để xem edge ổn định không."""
    df = load_matches_with_movement(execution=execution)
    if df.empty:
        return

    print(f"\n{'='*60}")
    print(f"  SEASON CONSISTENCY ({execution.upper()})")
    print("="*60)

    signals = df[df["signal_side"].notna()].copy()
    signals = signals.dropna(subset=["bet_odds"])
    signals["won"] = signals["result"] == signals["signal_side"]
    signals["profit"] = signals.apply(
        lambda r: r["bet_odds"] - 1 if r["won"] else -1, axis=1
    )

    by_season = (
        signals.groupby("season")
        .agg(bets=("profit","count"), roi=("profit","mean"), wins=("won","sum"))
        .reset_index()
    )
    by_season["win_rate"] = by_season["wins"] / by_season["bets"]

    for _, row in by_season.iterrows():
        icon = "✅" if row["roi"] > 0 else "❌"
        print(
            f"  {icon} {row['season']:10s} "
            f"{row['bets']:3d} bets | "
            f"Win {row['win_rate']:.1%} | "
            f"ROI {row['roi']:+.2%}"
        )

    profitable = (by_season["roi"] > 0).sum()
    total_seasons = len(by_season)
    print(f"\n  Profitable seasons: {profitable}/{total_seasons}")

    if profitable >= total_seasons * 0.7:
        print("  ✅ Edge CONSISTENT across seasons")
    elif profitable >= total_seasons * 0.5:
        print("  ⚠️  Edge MARGINAL — acceptable but monitor closely")
    else:
        print("  ❌ Edge INCONSISTENT — high overfitting risk")
    print()


if __name__ == "__main__":
    results = compare_execution_modes()
    season_consistency_check(execution="realistic")
    season_consistency_check(execution="historical")
