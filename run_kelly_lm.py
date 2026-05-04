import sys, json
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from betting.backtest_kelly_lm import run_kelly_backtest

results_df, summary = run_kelly_backtest()

print("\n========== KELLY BACKTEST SUMMARY ==========")
print(json.dumps(summary, indent=2))

print("\n========== BY SEASON ==========")
print(f"{'Season':>10} {'Bets':>5} {'Win%':>7} {'Profit':>9} {'Bankroll':>10}")
print("-" * 48)
for season in sorted(results_df["season"].unique()):
    s = results_df[results_df["season"] == season]
    last_br = s["bankroll"].iloc[-1]
    print(f"{season:>10} {len(s):>5} {s['won'].mean()*100:>7.1f}% {s['profit'].sum():>9.2f}  {last_br:>10.2f}")

print("\n========== BY LEAGUE ==========")
for league in results_df["league"].unique():
    s = results_df[results_df["league"] == league]
    print(f"{league}: {len(s)} bets | profit={s['profit'].sum():.2f} | win={s['won'].mean()*100:.1f}%")

# Luu CSV
out = Path("data/processed/kelly_lm_results.csv")
results_df.to_csv(out, index=False)
print(f"\nSaved -> {out}")
