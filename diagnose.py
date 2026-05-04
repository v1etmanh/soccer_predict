import pandas as pd, numpy as np, pickle, sys
sys.path.insert(0, 'src')
from models.trainer import EnsembleWrapper  # cần import để pickle load được
from features.features import build_feature_matrix
from data.database import get_session, Match

session = get_session()
rows = session.query(Match).all()
df = pd.DataFrame([{
    'date': m.date, 'season': m.season,
    'home_team': m.home_team, 'away_team': m.away_team,
    'home_goals': m.home_goals, 'away_goals': m.away_goals,
    'result': m.result,
    'odds_h': m.odds_h, 'odds_d': m.odds_d, 'odds_a': m.odds_a,
    'home_shots_target': m.home_shots_target,
    'away_shots_target': m.away_shots_target,
} for m in rows])
session.close()

with open('models/calibrated/calibrated_all.pkl', 'rb') as f:
    model = pickle.load(f)

X, y, df_feat = build_feature_matrix(df)
X = X.fillna(X.median())
probs = model.predict_proba(X)

print('=== PROB DISTRIBUTION ===')
print(f'Home  mean={probs[:,0].mean():.3f} std={probs[:,0].std():.3f} min={probs[:,0].min():.3f} max={probs[:,0].max():.3f}')
print(f'Draw  mean={probs[:,1].mean():.3f} std={probs[:,1].std():.3f} min={probs[:,1].min():.3f} max={probs[:,1].max():.3f}')
print(f'Away  mean={probs[:,2].mean():.3f} std={probs[:,2].std():.3f} min={probs[:,2].min():.3f} max={probs[:,2].max():.3f}')

odds_ok = df_feat[['odds_h','odds_d','odds_a']].notna().all(axis=1)
df_ok = df_feat[odds_ok].copy()
p_ok  = probs[odds_ok.values]

ev_h = p_ok[:,0] * df_ok['odds_h'].values - 1
ev_d = p_ok[:,1] * df_ok['odds_d'].values - 1
ev_a = p_ok[:,2] * df_ok['odds_a'].values - 1

print(f'\n=== EV DISTRIBUTION (n={len(df_ok)} matches co odds) ===')
print(f'EV home  mean={ev_h.mean():.3f}  pct>0.08: {(ev_h>0.08).mean()*100:.1f}%')
print(f'EV draw  mean={ev_d.mean():.3f}  pct>0.08: {(ev_d>0.08).mean()*100:.1f}%')
print(f'EV away  mean={ev_a.mean():.3f}  pct>0.08: {(ev_a>0.08).mean()*100:.1f}%')

print(f'\n=== DATA QUALITY ===')
print(f'home_shots_target null: {df["home_shots_target"].isna().mean()*100:.1f}%')
print(f'odds_h null:            {df["odds_h"].isna().mean()*100:.1f}%')
print(f'Total matches: {len(df)}  |  Seasons: {sorted(df["season"].unique())}')

print(f'\n=== ACTUAL RESULT DISTRIBUTION ===')
print(df['result'].value_counts(normalize=True).round(3))
