import sys
sys.path.insert(0, 'src')
from data.database import get_session, OddsSnapshot, LiveBet

s = get_session()
snaps     = s.query(OddsSnapshot).all()
baselines = [x for x in snaps if x.is_baseline]
bets      = s.query(LiveBet).all()

print(f"OddsSnapshots : {len(snaps)}")
print(f"  Baselines   : {len(baselines)}")
print(f"  Non-baseline: {len(snaps) - len(baselines)}")
print(f"LiveBets      : {len(bets)}")
print()
print("Matches with baselines:")
for b in baselines:
    print(f"  {b.home_team} vs {b.away_team} | {str(b.match_date)[:10]} | H:{b.odds_h} D:{b.odds_d} A:{b.odds_a}")
s.close()
