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

s2425 = signals[signals["season"] == "2024-25"].copy()

# 1. Movement distribution - season nay sharp signal co manh khong?
print("=== MOVEMENT SIZE (2024-25 vs truoc) ===")
def get_move(row):
    side = row["signal_side"]
    move_map = {"H": row.get("move_h"), "D": row.get("move_d"), "A": row.get("move_a")}
    return move_map.get(side)

signals["move"] = signals.apply(get_move, axis=1)
s2425["move"] = s2425.apply(get_move, axis=1)
sold = signals[signals["season"] != "2024-25"]

print(f"Avg movement (truoc 2024-25) : {sold['move'].mean():.4f}")
print(f"Avg movement (2024-25)       : {s2425['move'].mean():.4f}")

# 2. Chia theo side - H/D/A cai nao dang thua?
print("\n=== ROI BY SIDE (2024-25) ===")
print(f"{'Side':>6} {'Bets':>6} {'Win%':>7} {'ROI':>8}")
print("-" * 32)
for side in ["H", "D", "A"]:
    sub = s2425[s2425["signal_side"] == side]
    if sub.empty: continue
    print(f"{side:>6} {len(sub):>6} {sub['won'].mean():>7.1%} {sub['profit'].mean():>8.2%}")

# 3. Cumulative profit 2024-25 theo thoi gian
print("\n=== CUMULATIVE PROFIT 2024-25 (theo thang) ===")
s2425["month"] = pd.to_datetime(s2425["date"]).dt.to_period("M")
monthly = s2425.groupby("month").agg(
    bets=("profit","count"),
    roi=("profit","mean"),
    profit=("profit","sum")
).round(3)
print(monthly.to_string())
