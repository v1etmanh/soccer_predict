"""
trainer.py — Train XGBoost + LightGBM stacked ensemble
Calibrate xác suất bằng Platt Scaling, lưu model
"""
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import XGB_PARAMS, LGB_PARAMS, MODELS_DIR


def train_base_models(X_train, y_train):
    """Train XGBoost và LightGBM"""
    logger.info("[Trainer] Training XGBoost...")
    xgb = XGBClassifier(**XGB_PARAMS)
    xgb.fit(X_train, y_train)

    logger.info("[Trainer] Training LightGBM...")
    lgb = LGBMClassifier(**LGB_PARAMS)
    lgb.fit(X_train, y_train)

    return xgb, lgb


def get_oof_predictions(X, y, model_cls, params, n_splits=5):
    """Out-of-fold predictions để tránh data leakage khi stacking"""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    oof = np.zeros((len(X), 3))

    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr = y.iloc[tr_idx]
        m = model_cls(**params)
        m.fit(X_tr, y_tr)
        oof[val_idx] = m.predict_proba(X_val)
        logger.debug(f"[Trainer] Fold {fold+1}/{n_splits} done")

    return oof


def train_meta_model(oof_xgb, oof_lgb, y_train):
    """Meta-learner: Logistic Regression trên OOF predictions"""
    X_meta = np.hstack([oof_xgb, oof_lgb])
    meta = LogisticRegression(C=1.0, max_iter=1000)  # bỏ multi_class (deprecated sklearn 1.5+)
    meta.fit(X_meta, y_train)
    logger.info("[Trainer] Meta model trained")
    return meta


class EnsembleWrapper:
    """Wrapper gộp XGB + LGB + Meta — định nghĩa ở module level để pickle được"""
    _estimator_type = "classifier"

    def __init__(self, xgb, lgb, meta):
        self.xgb = xgb
        self.lgb = lgb
        self.meta = meta
        self.classes_ = np.array([0, 1, 2])

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_proba(self, X):
        Xm = np.hstack([self.xgb.predict_proba(X), self.lgb.predict_proba(X)])
        return self.meta.predict_proba(Xm)


def calibrate_model(model, X_cal, y_cal):
    """Platt Scaling calibration"""
    calibrated = CalibratedClassifierCV(model, method="sigmoid", cv="prefit")
    calibrated.fit(X_cal, y_cal)
    return calibrated


def evaluate(probs, y_true):
    """Brier Score per class + Log Loss"""
    results = {}
    labels = {0: "Home", 1: "Draw", 2: "Away"}
    for cls, name in labels.items():
        binary = (y_true == cls).astype(int)
        results[f"brier_{name}"] = round(brier_score_loss(binary, probs[:, cls]), 4)
    results["log_loss"] = round(log_loss(y_true, probs), 4)
    return results

def save_models(xgb, lgb, meta, calibrated, league="all"):
    """Lưu toàn bộ models vào disk"""
    save_dir = MODELS_DIR / "saved"
    cal_dir  = MODELS_DIR / "calibrated"
    save_dir.mkdir(parents=True, exist_ok=True)
    cal_dir.mkdir(parents=True, exist_ok=True)

    for name, obj in [("xgb", xgb), ("lgb", lgb), ("meta", meta)]:
        path = save_dir / f"{name}_{league}.pkl"
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        logger.info(f"[Trainer] Saved {path}")

    cal_path = cal_dir / f"calibrated_{league}.pkl"
    with open(cal_path, "wb") as f:
        pickle.dump(calibrated, f)
    logger.info(f"[Trainer] Saved calibrated model → {cal_path}")


def load_models(league="all"):
    """Load models từ disk"""
    save_dir = MODELS_DIR / "saved"
    cal_dir  = MODELS_DIR / "calibrated"
    models = {}
    for name in ["xgb", "lgb", "meta"]:
        path = save_dir / f"{name}_{league}.pkl"
        with open(path, "rb") as f:
            models[name] = pickle.load(f)
    cal_path = cal_dir / f"calibrated_{league}.pkl"
    with open(cal_path, "rb") as f:
        models["calibrated"] = pickle.load(f)
    return models


def run_training_pipeline(X, y, league="all"):
    """
    Full training pipeline:
    1. OOF → train meta → calibrate → evaluate → save
    """
    from sklearn.model_selection import train_test_split
    # Time-aware split: 80% train, 20% calibration
    split = int(len(X) * 0.8)
    X_train, X_cal = X.iloc[:split], X.iloc[split:]
    y_train, y_cal = y.iloc[:split], y.iloc[split:]

    logger.info(f"[Trainer] Train size: {len(X_train)} | Cal size: {len(X_cal)}")

    # OOF predictions
    logger.info("[Trainer] Computing OOF for XGBoost...")
    oof_xgb = get_oof_predictions(X_train, y_train, XGBClassifier, XGB_PARAMS)

    logger.info("[Trainer] Computing OOF for LightGBM...")
    oof_lgb = get_oof_predictions(X_train, y_train, LGBMClassifier, LGB_PARAMS)

    # Train final base models on full train set
    xgb, lgb = train_base_models(X_train, y_train)

    # Meta model
    meta = train_meta_model(oof_xgb, oof_lgb, y_train)

    # Evaluate trước calibration
    X_cal_meta = np.hstack([xgb.predict_proba(X_cal), lgb.predict_proba(X_cal)])
    probs_raw  = meta.predict_proba(X_cal_meta)
    metrics_raw = evaluate(probs_raw, y_cal.values)
    logger.info(f"[Trainer] Before calibration: {metrics_raw}")

    # Calibrate — dùng EnsembleWrapper định nghĩa ở module level (pickle-safe)
    ensemble   = EnsembleWrapper(xgb, lgb, meta)
    calibrated = calibrate_model(ensemble, X_cal, y_cal)

    # Evaluate sau calibration
    probs_cal = calibrated.predict_proba(X_cal)
    metrics_cal = evaluate(probs_cal, y_cal.values)
    logger.info(f"[Trainer] After calibration:  {metrics_cal}")

    # Save
    save_models(xgb, lgb, meta, calibrated, league)
    logger.success(f"[Trainer] Pipeline hoàn thành cho league={league}")
    return calibrated, metrics_cal


if __name__ == "__main__":
    logger.add("logs/trainer.log", rotation="10 MB")
    import pandas as pd
    from data.database import get_session, Match
    from features.features import build_feature_matrix

    session = get_session()
    rows = session.query(Match).all()
    df = pd.DataFrame([{
        "date": m.date, "home_team": m.home_team, "away_team": m.away_team,
        "home_goals": m.home_goals, "away_goals": m.away_goals, "result": m.result,
        "home_shots_target": m.home_shots_target, "away_shots_target": m.away_shots_target,
    } for m in rows])
    session.close()

    X, y, _ = build_feature_matrix(df)
    X = X.fillna(X.median())
    run_training_pipeline(X, y, league="all")
