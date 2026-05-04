"""
deep_dive.py — Phân tích chi tiết để tìm edge thật
=====================================================
Câu hỏi cần trả lời:
  1. EPL +2.9% realistic có thật hay noise? (191 bets)
  2. Tại sao 2023-25 dương còn 2020-23 âm? Market thay đổi hay may mắn?
  3. Có threshold/filter nào cho ROI realistic dương ổn định không?
  4. Movement size có predict được win rate không?

Chạy: python deep_dive.py
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from signals.line_movement_v2 import load_matches_with_movement
from loguru import logger

logger.remove()  # Tắt loguru output để output gọn hơn


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print("="*60)


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load cả hai mode."""
    df_r = load_matches_with_movement(execution="realistic")
    df_h = load_matches_with_movement(execution="historical")
    sig_r = df_r[df_r["signal_side"].notna()].dropna(subset=["bet_odds"]).copy()
    sig_h = df_h[df_h["signal_side"].notna()].dropna(subset=["bet_odds"]).copy()
    for df in [sig_r, sig_h]:
        df["won"]    = df["result"] == df["signal_side"]
        df["profit"] = df.apply(lambda r: r["bet_odds"] - 1 if r["won"] else -1, axis=1)
    return sig_r, sig_h


# ── 1. EPL only, realistic ─────────────────────────────────────
def epl_deep_dive(sig_r: pd.DataFrame):
    section("EPL ONLY — Realistic (bet@close)")
    epl = sig_r[sig_r["league"] == "soccer_epl"]
    print(f"  Total bets: {len(epl)} | ROI: {epl['profit'].mean():+.2%} | "
          f"Win: {epl['won'].mean():.1%}")

    print("\n  By season:")
    for s, g in epl.groupby("season"):
        icon = "✅" if g["profit"].mean() > 0 else "❌"
        print(f"    {icon} {s}: {len(g)} bets | ROI {g['profit'].mean():+.2%} | "
              f"Win {g['won'].mean():.1%}")

    print("\n  By signal side:")
    for side, g in epl.groupby("signal_side"):
        icon = "✅" if g["profit"].mean() > 0 else "❌"
        print(f"    {icon} {side}: {len(g)} bets | ROI {g['profit'].mean():+.2%}")


# ── 2. Movement size buckets ────────────────────────────────────
def movement_buckets(sig_r: pd.DataFrame):
    section("MOVEMENT SIZE vs ROI (realistic)")
    sig_r = sig_r.copy()
    # movement = close_prob - open_prob của signal side
    def get_move(row):
        s = row["signal_side"]
        return row.get(f"move_{s.lower()}")
    sig_r["movement"] = sig_r.apply(get_move, axis=1)

    bins   = [0.03, 0.04, 0.05, 0.06, 0.07, 0.10, 1.0]
    labels = ["3-4%", "4-5%", "5-6%", "6-7%", "7-10%", ">10%"]
    sig_r["move_bucket"] = pd.cut(sig_r["movement"], bins=bins, labels=labels)

    print(f"  {'Bucket':<10} {'Bets':>6} {'WinRate':>8} {'ROI':>8} {'Profit':>8}")
    print(f"  {'-'*44}")
    for bucket, g in sig_r.groupby("move_bucket", observed=True):
        icon = "✅" if g["profit"].mean() > 0 else "❌"
        print(f"  {icon} {str(bucket):<8} {len(g):>6} "
              f"{g['won'].mean():>8.1%} "
              f"{g['profit'].mean():>+8.2%} "
              f"{g['profit'].sum():>+8.2f}u")

    # Tìm threshold tốt nhất
    print("\n  --- Threshold sweep (realistic) ---")
    print(f"  {'Threshold':>10} {'Bets':>6} {'WinRate':>8} {'ROI':>8}")
    for thr in [0.03, 0.035, 0.04, 0.045, 0.05, 0.06, 0.07]:
        subset = sig_r[sig_r["movement"] >= thr]
        if len(subset) < 20:
            continue
        icon = "✅" if subset["profit"].mean() > 0 else "❌"
        print(f"  {icon} ≥{thr:.1%}    {len(subset):>6} "
              f"{subset['won'].mean():>8.1%} "
              f"{subset['profit'].mean():>+8.2%}")


# ── 3. Odds range analysis ──────────────────────────────────────
def odds_range_analysis(sig_r: pd.DataFrame):
    section("ODDS RANGE vs ROI (realistic)")
    bins   = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 20.0]
    labels = ["1.0-1.5", "1.5-2.0", "2.0-2.5", "2.5-3.0", "3.0-4.0", "4.0-6.0", ">6.0"]
    sig_r = sig_r.copy()
    sig_r["odds_bucket"] = pd.cut(sig_r["bet_odds"], bins=bins, labels=labels)

    print(f"  {'Odds':>10} {'Bets':>6} {'WinRate':>8} {'ROI':>8}")
    print(f"  {'-'*36}")
    for bucket, g in sig_r.groupby("odds_bucket", observed=True):
        if len(g) < 5:
            continue
        icon = "✅" if g["profit"].mean() > 0 else "❌"
        print(f"  {icon} {str(bucket):<10} {len(g):>5} "
              f"{g['won'].mean():>8.1%} "
              f"{g['profit'].mean():>+8.2%}")


# ── 4. "Early entry" simulation ─────────────────────────────────
def early_entry_simulation(sig_r: pd.DataFrame, sig_h: pd.DataFrame):
    """
    Simulate đặt tại odds = midpoint giữa opening và closing.
    Đây là ước tính nếu detect signal vào giữa tuần.
    """
    section("EARLY ENTRY SIMULATION (bet @ midpoint open/close)")
    merged = sig_r.copy()

    # Lấy opening odds từ sig_h
    open_odds_map = {}
    for _, row in sig_h.iterrows():
        key = (row["home_team"], row["away_team"], str(row["date"])[:10])
        open_odds_map[key] = row["bet_odds"]

    def get_mid_odds(row):
        key  = (row["home_team"], row["away_team"], str(row["date"])[:10])
        open_o = open_odds_map.get(key)
        close_o = row["bet_odds"]
        if open_o and close_o:
            return round((open_o + close_o) / 2, 3)
        return close_o

    merged["mid_odds"] = merged.apply(get_mid_odds, axis=1)
    merged["mid_profit"] = merged.apply(
        lambda r: r["mid_odds"] - 1 if r["won"] else -1, axis=1
    )

    total   = len(merged)
    roi_mid = merged["mid_profit"].mean()
    roi_cls = merged["profit"].mean()

    print(f"  bet@close  ROI: {roi_cls:+.2%}")
    print(f"  bet@mid    ROI: {roi_mid:+.2%}  ← estimate nếu vào giữa tuần")
    print(f"  bet@open   ROI: {sig_h['profit'].mean():+.2%}")

    if roi_mid > 0:
        print(f"\n  ✅ Midpoint execution khả thi → cần poll sớm (Thứ 3/4)")
    else:
        print(f"\n  ❌ Ngay cả midpoint cũng âm → cần vào rất sớm hoặc re-think")

    # By season midpoint
    print("\n  By season (bet@mid):")
    for s, g in merged.groupby("season"):
        icon = "✅" if g["mid_profit"].mean() > 0 else "❌"
        print(f"    {icon} {s}: {len(g)} bets | ROI {g['mid_profit'].mean():+.2%}")


# ── 5. Bundesliga isolation ──────────────────────────────────────
def bundesliga_analysis(sig_r: pd.DataFrame):
    section("BUNDESLIGA PROBLEM ANALYSIS (realistic)")
    bun = sig_r[sig_r["league"] == "soccer_germany_bundesliga"]
    print(f"  Total: {len(bun)} bets | ROI: {bun['profit'].mean():+.2%}")
    print("\n  By season:")
    for s, g in bun.groupby("season"):
        icon = "✅" if g["profit"].mean() > 0 else "❌"
        print(f"    {icon} {s}: {len(g)} bets | ROI {g['profit'].mean():+.2%}")
    print("\n  → Khuyến nghị: exclude Bundesliga khỏi live trading")
    epl_r = sig_r[sig_r["league"] == "soccer_epl"]
    print(f"  EPL-only realistic ROI: {epl_r['profit'].mean():+.2%} "
          f"({len(epl_r)} bets)")


# ── 6. Recent form analysis (2023-25 only) ──────────────────────
def recent_seasons_analysis(sig_r: pd.DataFrame):
    section("RECENT SEASONS 2023-25 — Detailed (realistic)")
    recent = sig_r[sig_r["season"].isin(["2023-24", "2024-25"])]
    print(f"  Total: {len(recent)} bets | ROI: {recent['profit'].mean():+.2%}")

    print("\n  By league x season:")
    for (lg, s), g in recent.groupby(["league", "season"]):
        icon = "✅" if g["profit"].mean() > 0 else "❌"
        print(f"    {icon} {lg:<35} {s}: {len(g):3d} bets | ROI {g['profit'].mean():+.2%}")

    def get_move(row):
        s = row["signal_side"]
        return row.get(f"move_{s.lower()}")
    recent = recent.copy()
    recent["movement"] = recent.apply(get_move, axis=1)

    print("\n  At threshold ≥4% (recent only):")
    sub = recent[recent["movement"] >= 0.04]
    if len(sub) > 0:
        icon = "✅" if sub["profit"].mean() > 0 else "❌"
        print(f"    {icon} {len(sub)} bets | ROI {sub['profit'].mean():+.2%}")


# ── Main ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading data...")
    sig_r, sig_h = load()
    print(f"Loaded {len(sig_r)} signals (realistic) | {len(sig_h)} (historical)")

    epl_deep_dive(sig_r)
    bundesliga_analysis(sig_r)
    movement_buckets(sig_r)
    odds_range_analysis(sig_r)
    early_entry_simulation(sig_r, sig_h)
    recent_seasons_analysis(sig_r)

    print("\n" + "="*60)
    print("  SUMMARY: Câu hỏi cần trả lời tiếp theo")
    print("="*60)
    print("  1. EPL-only realistic ROI có dương và ổn định không?")
    print("  2. Threshold ≥4-5% có cải thiện realistic ROI không?")
    print("  3. Midpoint execution (poll thứ 3/4) có ROI dương không?")
    print("  4. Bundesliga có nên exclude hoàn toàn không?")
    print()
