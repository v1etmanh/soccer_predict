import sys; sys.path.insert(0, 'src')
from signals.line_movement import load_matches_with_movement, backtest_line_movement
import json
df = load_matches_with_movement()
print(f"Matches loaded: {len(df)}")
print(f"Signals: {df['signal_side'].notna().sum()}")
stats = backtest_line_movement(df)
print(json.dumps(stats, indent=2))
