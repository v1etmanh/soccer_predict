import sys; sys.path.insert(0, 'src')
from signals.line_movement import load_matches_with_movement, compute_line_movement, MOVEMENT_THRESHOLD
import pandas as pd

df = load_matches_with_movement()

print(f"\n{'Threshold':>10} {'Bets':>6} {'Win%':>7} {'ROI':>8} {'Profit':>9}")
print("-" * 45)

for thresh in [0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05]:
    signals = df[df["signal_side"].notna()].copy()

    # Re-filter bằng threshold này
    def recheck(row):
        side = row["signal_side"]
        move_map = {"H": row.get("move_h"), "D": row.get("move_d"), "A": row.get("move_a")}
        move = move_map.get(side)
        return move is not None and move >= thresh

    filtered = signals[signals.apply(recheck, axis=1)].copy()
    if filtered.empty:
        continue

    def get_profit(row):
        side = row["signal_side"]
        odds_map = {"H": row.get("odds_h"), "D": row.get("odds_d"), "A": row.get("odds_a")}
        won = row.get("result") == side
        odds = odds_map.get(side) or 0
        return (odds - 1) if won else -1, won

    filtered[["profit", "won"]] = filtered.apply(lambda r: pd.Series(get_profit(r)), axis=1)
    filtered = filtered.dropna(subset=["profit"])

    n = len(filtered)
    win_rate = filtered["won"].mean()
    roi = filtered["profit"].mean()
    profit = filtered["profit"].sum()
    print(f"{thresh:>10.1%} {n:>6} {win_rate:>7.1%} {roi:>8.2%} {profit:>9.2f}u")
