import pandas as pd
import numpy as np

df = pd.read_csv('data/processed/backtest_results.csv')
print(f'Total bets: {len(df)}')
print(f'Win rate: {df.won.mean()*100:.1f}%')
print(f'Avg odds: {df.odds.mean():.3f}')
print(f'Avg EV model: {df.ev.mean():.3f}')
print(f'Avg prob model: {df.prob.mean():.3f}')

flat_profit = df.apply(lambda r: r.odds - 1 if r.won else -1, axis=1)
print(f'\n=== FLAT BETTING ===')
print(f'Total flat profit: {flat_profit.sum():.2f} units')
print(f'ROI flat: {flat_profit.sum()/len(df)*100:.2f}%')

true_ev = df.won.mean() * df.odds.mean() - 1
print(f'\n=== TRUE EV ===')
print(f'True EV per bet:  {true_ev*100:.2f}%')
print(f'Model EV per bet: {df.ev.mean()*100:.2f}%')

print(f'\n=== KELLY FRACTION SIMULATION ===')
BASE_FRAC = 0.15
print(f'{"Fraction":>10} | {"Final BR":>10} | {"Max DD":>10} | {"ROI":>8}')
for frac in [0.01, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25]:
    br = 1000.0
    peak = 1000.0
    max_dd = 0.0
    for _, row in df.iterrows():
        k = row.kelly * (frac / BASE_FRAC)
        stake = min(br * k, br * 0.10)
        if row.won:
            br += stake * (row.odds - 1)
        else:
            br -= stake
        if br < 0:
            br = 0
            break
        peak = max(peak, br)
        dd = (br - peak) / peak * 100
        max_dd = min(max_dd, dd)
    roi = (br - 1000) / 1000 * 100
    print(f'{frac:>10.3f} | {br:>10.2f} | {max_dd:>9.1f}% | {roi:>7.1f}%')

print(f'\n=== BY SIDE ===')
g = df.groupby('side').agg(bets=('won','count'), win_rate=('won','mean'),
                            avg_odds=('odds','mean'), flat_roi=('odds', lambda x: None))
for side, grp in df.groupby('side'):
    wr = grp.won.mean()
    ao = grp.odds.mean()
    fp = grp.apply(lambda r: r.odds-1 if r.won else -1, axis=1).sum()
    print(f'{side:6s}  bets={len(grp):4d}  win={wr*100:.1f}%  avg_odds={ao:.2f}  flat_profit={fp:.1f}u  flat_roi={fp/len(grp)*100:.1f}%')

print(f'\n=== BY SEASON ===')
for s, grp in df.groupby('season'):
    fp = grp.apply(lambda r: r.odds-1 if r.won else -1, axis=1).sum()
    print(f'{s}  bets={len(grp):4d}  win={grp.won.mean()*100:.1f}%  flat_profit={fp:.1f}u')
