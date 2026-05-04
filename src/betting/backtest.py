"""
backtest.py — Walk-Forward Validation
Train → Test theo từng mùa, tránh data leakage tuyệt đối
"""
import pandas as pd
import numpy as np
from loguru import logger
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import INITIAL_BANKROLL, KELLY_FRACTION, MIN_EV_THRESHOLD, MIN_ODDS, MAX_ODDS, MIN_PROB
from features.features import build_feature_matrix
from models.trainer import run_training_pipeline, EnsembleWrapper
from betting.kelly import evaluate_bet_signals, BankrollManager


def walk_forward_backtest(df: pd.DataFrame, train_seasons: int = 3) -> pd.DataFrame:
    """
    Walk-Forward: train trên N mùa → test mùa tiếp theo → roll forward.
    Trả về DataFrame với tất cả tín hiệu và P&L.
    """
    df = df.sort_values("date").copy()
    seasons = sorted(df["season"].unique())
    logger.info(f"[Backtest] Seasons: {seasons}")

    all_results = []
    manager = BankrollManager(INITIAL_BANKROLL)

    for i in range(train_seasons, len(seasons)):
        train_s = seasons[i - train_seasons : i]
        test_s  = seasons[i]

        df_train = df[df["season"].isin(train_s)].copy()
        df_test  = df[df["season"] == test_s].copy()

        if len(df_train) < 100 or len(df_test) < 20:
            logger.warning(f"[Backtest] Bỏ qua {test_s}: dữ liệu quá ít")
            continue

        logger.info(f"[Backtest] Train: {train_s} ({len(df_train)} trận) | Test: {test_s} ({len(df_test)} trận)")

        # Build features — QUAN TRỌNG: chỉ gọi trên df_train để tránh leakage
        # Sau đó tính features cho test dựa trên lịch sử đã bao gồm train
        X_train, y_train, _ = build_feature_matrix(df_train)

        # Với test: concat train+test để Elo/H2H/rolling có lịch sử đầy đủ,
        # nhưng shift(1) trong compute_rolling_stats đảm bảo test rows
        # chỉ thấy dữ liệu TRƯỚC ngày thi đấu của chúng (không leakage).
        df_combined = pd.concat([df_train, df_test]).sort_values("date").reset_index(drop=True)
        _, _, df_combined_feat = build_feature_matrix(df_combined)

        # Tách lấy phần test theo date (an toàn hơn iloc khi index bị reset)
        test_dates = set(df_test["date"].astype(str))
        df_test_feat = df_combined_feat[
            (df_combined_feat["date"].astype(str).isin(test_dates)) &
            (df_combined_feat["season"] == test_s)
        ].copy()

        feat_cols = [c for c in X_train.columns]
        X_test = df_test_feat[feat_cols].copy()
        y_test = df_test_feat["result"].map({"H": 0, "D": 1, "A": 2})

        X_train = X_train.fillna(X_train.median())
        X_test  = X_test.fillna(X_train.median())

        # Train
        try:
            calibrated, metrics = run_training_pipeline(X_train, y_train, league=f"wf_{test_s}")
        except Exception as e:
            logger.error(f"[Backtest] Train failed: {e}")
            continue

        # Predict + generate signals
        probs = calibrated.predict_proba(X_test)

        for j, (idx, row) in enumerate(df_test_feat.iterrows()):
            p_h, p_d, p_a = probs[j]
            o_h = row.get("odds_h"); o_d = row.get("odds_d"); o_a = row.get("odds_a")
            if pd.isna(o_h) or pd.isna(o_d) or pd.isna(o_a):
                continue

            signals = evaluate_bet_signals(
                p_h, p_d, p_a, o_h, o_d, o_a,
                min_ev=MIN_EV_THRESHOLD,
                kelly_frac=KELLY_FRACTION,
                min_odds=MIN_ODDS,
                max_odds=MAX_ODDS,
                min_prob=MIN_PROB,
            )
            result = row.get("result", "")

            for side, info in signals.items():
                if not info["signal"]:
                    continue
                actual_win = (
                    (side == "home" and result == "H") or
                    (side == "draw" and result == "D") or
                    (side == "away" and result == "A")
                )
                stake = manager.bankroll * info["kelly"]
                manager.settle_bet(stake, info["odds"], actual_win)

                all_results.append({
                    "season":    test_s,
                    "date":      row["date"],
                    "home_team": row["home_team"],
                    "away_team": row["away_team"],
                    "side":      side,
                    "prob":      info["prob"],
                    "odds":      info["odds"],
                    "ev":        info["ev"],
                    "kelly":     info["kelly"],
                    "stake":     round(stake, 2),
                    "won":       actual_win,
                    "profit":    round(stake * (info["odds"] - 1) if actual_win else -stake, 2),
                    "bankroll":  manager.bankroll,
                })

    summary = manager.summary()
    logger.success(f"[Backtest] Kết quả: {summary}")
    results_df = pd.DataFrame(all_results)
    return results_df, summary


if __name__ == "__main__":
    logger.add("logs/backtest.log", rotation="10 MB")
    from data.database import get_session, Match

    session = get_session()
    rows = session.query(Match).all()
    df = pd.DataFrame([{
        "date": m.date, "season": m.season,
        "home_team": m.home_team, "away_team": m.away_team,
        "home_goals": m.home_goals, "away_goals": m.away_goals,
        "result": m.result,
        "odds_h": m.odds_h, "odds_d": m.odds_d, "odds_a": m.odds_a,
        "home_shots_target": m.home_shots_target,
        "away_shots_target": m.away_shots_target,
    } for m in rows])
    session.close()

    results_df, summary = walk_forward_backtest(df, train_seasons=3)
    out = Path("data/processed/backtest_results.csv")
    results_df.to_csv(out, index=False)
    logger.info(f"[Backtest] Saved → {out}")
    print(summary)
