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

# So sanh: ALL vs NO_HOME vs AWAY_ONLY
print(f"\n{'Filter':>15} {'Bets':>6} {'Win%':>7} {'ROI':>8} {'Profit':>9}")
print("-" * 50)

for label, mask in [
    ("All signals",   signals["signal_side"].isin(["H","D","A"])),
    ("No Home",       signals["signal_side"].isin(["D","A"])),
    ("Away only",     signals["signal_side"] == "A"),
]:
    sub = signals[mask]
    print(f"{label:>15} {len(sub):>6} {sub['won'].mean():>7.1%} {sub['profit'].mean():>8.2%} {sub['profit'].sum():>9.2f}u")

print("\n=== NO HOME — BY SEASON ===")
no_home = signals[signals["signal_side"].isin(["D","A"])]
print(f"\n{'Season':>10} {'Bets':>6} {'Win%':>7} {'ROI':>8} {'Profit':>9}")
print("-" * 48)
cumulative = 0
for season in sorted(no_home["season"].unique()):
    s = no_home[no_home["season"] == season]
    roi = s["profit"].mean()
    profit = s["profit"].sum()
    cumulative += profit
    flag = "PROFIT" if roi > 0 else "LOSS"
    print(f"{season:>10} {len(s):>6} {s['won'].mean():>7.1%} {roi:>8.2%} {profit:>9.2f}u  {flag}")
print("-" * 48)
print(f"{'TOTAL':>10} {len(no_home):>6} {no_home['won'].mean():>7.1%} {no_home['profit'].mean():>8.2%} {cumulative:>9.2f}u")
