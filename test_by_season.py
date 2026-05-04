import sys; sys.path.insert(0, 'src')
from signals.line_movement import load_matches_with_movement
import pandas as pd

df = load_matches_with_movement()

def get_profit(row):
    side = row["signal_side"]
    odds_map = {"H": row.get("odds_h"), "D": row.get("odds_d"), "A": row.get("odds_a")}
    won = row.get("result") == side
    odds = odds_map.get(side) or 0
    return (odds - 1) if won else -1, won

signals = df[df["signal_side"].notna()].copy()
signals[["profit", "won"]] = signals.apply(lambda r: pd.Series(get_profit(r)), axis=1)
signals = signals.dropna(subset=["profit"])

print(f"\n{'Season':>10} {'Bets':>6} {'Win%':>7} {'ROI':>8} {'Profit':>9} {'Status'}")
print("-" * 58)

cumulative_profit = 0
for season in sorted(signals["season"].unique()):
    s = signals[signals["season"] == season]
    n = len(s)
    win_rate = s["won"].mean()
    roi = s["profit"].mean()
    profit = s["profit"].sum()
    cumulative_profit += profit
    status = "PROFIT" if roi > 0 else "LOSS"
    print(f"{season:>10} {n:>6} {win_rate:>7.1%} {roi:>8.2%} {profit:>9.2f}u  {status}")

print("-" * 58)
print(f"{'TOTAL':>10} {len(signals):>6} {signals['won'].mean():>7.1%} {signals['profit'].mean():>8.2%} {cumulative_profit:>9.2f}u")

# Break down by league x season
print("\n--- BY LEAGUE x SEASON ---")
pivot = signals.groupby(["league", "season"]).agg(
    bets=("profit", "count"),
    roi=("profit", "mean"),
).round(4)
print(pivot.to_string())
