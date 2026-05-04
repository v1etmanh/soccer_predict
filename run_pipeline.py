"""
run_pipeline.py — Master runner
Chạy toàn bộ pipeline từ đầu đến cuối:
  1. Init DB
  2. Scrape historical data
  3. Feature engineering + train models
  4. Walk-forward backtest
  5. Generate live signals

Usage:
    python run_pipeline.py [--step STEP]
    python run_pipeline.py --step scrape
    python run_pipeline.py --step train
    python run_pipeline.py --step backtest
    python run_pipeline.py --step signals
    python run_pipeline.py          (chạy toàn bộ)
"""
import sys
import argparse
import pandas as pd
from pathlib import Path
from loguru import logger

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
logger.add("logs/pipeline.log", rotation="20 MB", retention="10 days")


def step_init():
    logger.info("═══ STEP 1: Init Database ═══")
    from data.database import init_db
    init_db()
    logger.success("Database initialized ✓")


def step_scrape():
    logger.info("═══ STEP 2: Scrape Historical Data ═══")
    from data.scraper import run_scraper
    total = run_scraper()
    logger.success(f"Scraped {total} new matches ✓")


def step_train():
    logger.info("═══ STEP 3: Train Models ═══")
    from data.database import get_session, Match
    from features.features import build_feature_matrix
    from models.trainer import run_training_pipeline

    session = get_session()
    rows = session.query(Match).all()
    session.close()

    if not rows:
        logger.error("No data in DB. Run scrape first!")
        return

    df = pd.DataFrame([{
        "date": m.date, "home_team": m.home_team, "away_team": m.away_team,
        "home_goals": m.home_goals, "away_goals": m.away_goals,
        "result": m.result,
        "home_shots_target": m.home_shots_target,
        "away_shots_target": m.away_shots_target,
    } for m in rows])

    X, y, _ = build_feature_matrix(df)
    X = X.fillna(X.median())
    _, metrics = run_training_pipeline(X, y, league="all")
    logger.success(f"Training complete. Metrics: {metrics} ✓")


def step_backtest():
    logger.info("═══ STEP 4: Walk-Forward Backtest ═══")
    from data.database import get_session, Match
    from betting.backtest import walk_forward_backtest

    session = get_session()
    rows = session.query(Match).all()
    session.close()

    df = pd.DataFrame([{
        "date": m.date, "season": m.season,
        "home_team": m.home_team, "away_team": m.away_team,
        "home_goals": m.home_goals, "away_goals": m.away_goals,
        "result": m.result,
        "odds_h": m.odds_h, "odds_d": m.odds_d, "odds_a": m.odds_a,
        "home_shots_target": m.home_shots_target,
        "away_shots_target": m.away_shots_target,
    } for m in rows])

    results_df, summary = walk_forward_backtest(df, train_seasons=3)
    out = ROOT / "data" / "processed" / "backtest_results.csv"
    results_df.to_csv(out, index=False)
    logger.success(f"Backtest complete → {out}")
    logger.success(f"Summary: {summary} ✓")
    print("\n" + "═"*50)
    print("BACKTEST SUMMARY")
    print("═"*50)
    for k, v in summary.items():
        print(f"  {k:20s}: {v}")


def step_signals():
    logger.info("═══ STEP 5: Generate Live Signals ═══")
    from signals.live_signals import run_live_signals
    df_s = run_live_signals()
    if not df_s.empty:
        ev_count = df_s.get("has_signal", pd.Series(False)).sum()
        logger.success(f"Live signals generated: {ev_count} EV+ opportunities ✓")
    else:
        logger.warning("No signals generated (check ODDS_API_KEY in .env)")


STEPS = {
    "init":     step_init,
    "scrape":   step_scrape,
    "train":    step_train,
    "backtest": step_backtest,
    "signals":  step_signals,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Soccer Betting Pipeline Runner")
    parser.add_argument(
        "--step", choices=list(STEPS.keys()),
        help="Chạy 1 step cụ thể. Không chỉ định = chạy toàn bộ."
    )
    args = parser.parse_args()

    if args.step:
        STEPS[args.step]()
    else:
        logger.info("🚀 Running FULL pipeline...")
        for name, fn in STEPS.items():
            try:
                fn()
            except Exception as e:
                logger.error(f"Step {name} failed: {e}")
                raise
        logger.success("✅ Full pipeline complete!")
        print("\n✅ Pipeline xong! Chạy dashboard:")
        print("   streamlit run src/dashboard/app.py")
