"""
live_signals.py — Live odds fetcher + real-time EV+ signal generator
Pipeline:
  1. Tải upcoming matches + odds từ the-odds-api.com
  2. Load calibrated model (trained từ walk-forward)
  3. Xây dựng features cho upcoming matches dùng rolling stats hiện tại
  4. Predict probs → Tính EV+, Kelly → lưu DB + xuất signals.csv
"""
import sys
import pickle
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    ODDS_API_KEY, ODDS_API_BASE, TARGET_LEAGUES,
    MODELS_DIR, PROCESSED_DATA_DIR,
    KELLY_FRACTION, MIN_EV_THRESHOLD, ROLLING_WINDOWS
)
from data.database import get_session, Match, Prediction, LiveOdds
from features.features import (
    compute_elo_ratings, compute_rolling_stats,
    ELO_START,
)
from betting.kelly import evaluate_bet_signals, remove_overround


# ══════════════════════════════════════════════════════
# PHẦN 1: Fetch live odds từ the-odds-api.com
# ══════════════════════════════════════════════════════

def fetch_live_odds(league: str, regions: str = "eu", markets: str = "h2h") -> list[dict]:
    """
    Tải upcoming odds cho 1 league từ the-odds-api.com.
    Trả về list các event với odds từ nhiều bookmaker.
    """
    if not ODDS_API_KEY:
        logger.warning("[LiveSignals] ODDS_API_KEY chưa được set trong .env — bỏ qua fetch odds")
        return []

    url = f"{ODDS_API_BASE}/sports/{league}/odds"
    params = {
        "apiKey":  ODDS_API_KEY,
        "regions": regions,
        "markets": markets,
        "oddsFormat": "decimal",
        "dateFormat":  "iso",
    }
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        logger.info(f"[LiveSignals] Fetched {len(data)} events for {league}")
        return data
    except Exception as e:
        logger.error(f"[LiveSignals] Lỗi fetch odds {league}: {e}")
        return []


def parse_odds_event(event: dict) -> dict | None:
    """
    Parse 1 event từ API → dict chuẩn hóa.
    Lấy odds trung bình từ tất cả bookmaker (giảm overround bias).
    """
    home = event.get("home_team", "")
    away = event.get("away_team", "")
    commence_raw = event.get("commence_time", "")
    try:
        commence = datetime.fromisoformat(commence_raw.replace("Z", "+00:00"))
    except Exception:
        commence = None

    odds_list = {"H": [], "D": [], "A": []}
    for bm in event.get("bookmakers", []):
        for market in bm.get("markets", []):
            if market.get("key") != "h2h":
                continue
            outcomes = {o["name"]: o["price"] for o in market.get("outcomes", [])}
            if home in outcomes:
                odds_list["H"].append(outcomes[home])
            if away in outcomes:
                odds_list["A"].append(outcomes[away])
            # Draw key varies
            draw_key = [k for k in outcomes if k not in [home, away]]
            if draw_key:
                odds_list["D"].append(outcomes[draw_key[0]])

    if not odds_list["H"] or not odds_list["D"] or not odds_list["A"]:
        return None

    return {
        "event_id":  event.get("id", ""),
        "league":    event.get("sport_key", ""),
        "home_team": home,
        "away_team": away,
        "commence":  commence,
        "odds_h":    round(float(np.median(odds_list["H"])), 3),
        "odds_d":    round(float(np.median(odds_list["D"])), 3),
        "odds_a":    round(float(np.median(odds_list["A"])), 3),
    }


# ══════════════════════════════════════════════════════
# PHẦN 2: Xây dựng features cho upcoming matches
# ══════════════════════════════════════════════════════

def get_current_team_stats(df_history: pd.DataFrame) -> tuple[dict, dict]:
    """
    Lấy rolling stats và Elo HIỆN TẠI (sau trận cuối cùng) của từng đội.
    Dùng để dự đoán cho upcoming matches.
    Returns:
        team_stats: {team: Series of rolling features}
        team_elo:   {team: float elo_rating}
    """
    df = df_history.sort_values("date").copy()
    rolling = compute_rolling_stats(df)

    # Tính Elo cuối cùng
    df_elo = compute_elo_ratings(df)
    # Elo sau trận cuối mỗi đội
    elo_map = {}
    for _, row in df_elo.iterrows():
        elo_map[row["home_team"]] = row["home_elo"]
        elo_map[row["away_team"]] = row["away_elo"]

    return rolling, elo_map


def build_upcoming_features(events: list[dict], df_history: pd.DataFrame) -> pd.DataFrame:
    """
    Xây dựng feature matrix cho list upcoming matches.
    Dùng rolling stats + Elo hiện tại của từng đội.
    """
    rolling, elo_map = get_current_team_stats(df_history)
    rows = []
    for ev in events:
        h, a = ev["home_team"], ev["away_team"]
        hr = rolling.loc[h] if h in rolling.index else pd.Series(dtype=float)
        ar = rolling.loc[a] if a in rolling.index else pd.Series(dtype=float)
        h_elo = elo_map.get(h, ELO_START)
        a_elo = elo_map.get(a, ELO_START)
        feat = {
            "elo_diff":  round(h_elo - a_elo, 2),
            "home_elo":  h_elo,
            "away_elo":  a_elo,
            "h2h_home":  0.4,   # Prior — không có H2H cho future match
            "h2h_draw":  0.3,
            "h2h_away":  0.3,
        }
        for w in ROLLING_WINDOWS:
            feat[f"home_form_{w}"]     = hr.get(f"form_{w}", 1.0)
            feat[f"home_xg_{w}"]       = hr.get(f"xg_{w}", 0.5)
            feat[f"home_scored_{w}"]   = hr.get(f"scored_{w}", 1.2)
            feat[f"home_conceded_{w}"] = hr.get(f"conceded_{w}", 1.2)
            feat[f"away_form_{w}"]     = ar.get(f"form_{w}", 1.0)
            feat[f"away_xg_{w}"]       = ar.get(f"xg_{w}", 0.5)
            feat[f"away_scored_{w}"]   = ar.get(f"scored_{w}", 1.2)
            feat[f"away_conceded_{w}"] = ar.get(f"conceded_{w}", 1.2)
        rows.append(feat)

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════
# PHẦN 3: Load model + Predict
# ══════════════════════════════════════════════════════

def load_best_model(league: str = "all") -> object | None:
    """
    Load calibrated model. Ưu tiên model theo league,
    fallback về 'all' nếu không có.
    """
    for tag in [league, "all"]:
        cal_path = MODELS_DIR / "calibrated" / f"calibrated_{tag}.pkl"
        if cal_path.exists():
            with open(cal_path, "rb") as f:
                model = pickle.load(f)
            logger.info(f"[LiveSignals] Loaded model: {cal_path.name}")
            return model
    logger.error("[LiveSignals] Không tìm thấy model. Chạy trainer.py trước!")
    return None


def predict_upcoming(events: list[dict], df_history: pd.DataFrame, league: str = "all") -> list[dict]:
    """
    Dự đoán xác suất cho upcoming matches.
    Returns list signals với prob, EV, Kelly.
    """
    if not events:
        return []

    model = load_best_model(league)
    if model is None:
        return []

    X = build_upcoming_features(events, df_history)
    # Fill NaN bằng median của tập train (an toàn)
    X = X.fillna(0.5)

    try:
        probs = model.predict_proba(X)
    except Exception as e:
        logger.error(f"[LiveSignals] Predict error: {e}")
        return []

    signals_out = []
    for i, ev in enumerate(events):
        p_h, p_d, p_a = probs[i]
        signals = evaluate_bet_signals(
            p_h, p_d, p_a,
            ev["odds_h"], ev["odds_d"], ev["odds_a"],
            min_ev=MIN_EV_THRESHOLD,
            kelly_frac=KELLY_FRACTION,
        )
        # Fair probs từ odds (market consensus)
        fair_h, fair_d, fair_a = remove_overround(
            ev["odds_h"], ev["odds_d"], ev["odds_a"]
        )
        signals_out.append({
            "league":    ev["league"],
            "home_team": ev["home_team"],
            "away_team": ev["away_team"],
            "commence":  ev["commence"],
            "odds_h":    ev["odds_h"],
            "odds_d":    ev["odds_d"],
            "odds_a":    ev["odds_a"],
            "prob_home": round(p_h, 4),
            "prob_draw": round(p_d, 4),
            "prob_away": round(p_a, 4),
            "market_h":  round(fair_h, 4),
            "market_d":  round(fair_d, 4),
            "market_a":  round(fair_a, 4),
            "ev_home":   signals["home"]["ev"],
            "ev_draw":   signals["draw"]["ev"],
            "ev_away":   signals["away"]["ev"],
            "kelly_home": signals["home"]["kelly"],
            "kelly_draw": signals["draw"]["kelly"],
            "kelly_away": signals["away"]["kelly"],
            "signal_home": signals["home"]["signal"],
            "signal_draw": signals["draw"]["signal"],
            "signal_away": signals["away"]["signal"],
            "has_signal":  any([
                signals["home"]["signal"],
                signals["draw"]["signal"],
                signals["away"]["signal"],
            ]),
        })

    return signals_out


# ══════════════════════════════════════════════════════
# PHẦN 4: Save vào DB + export CSV
# ══════════════════════════════════════════════════════

def save_live_odds_to_db(events: list[dict], session) -> int:
    """Lưu raw odds vào bảng live_odds, tránh duplicate"""
    saved = 0
    for ev in events:
        key = f"{ev['home_team']}__{ev['away_team']}__{ev.get('commence', '')}"
        exists = session.query(LiveOdds).filter_by(
            match_key=key, league=ev["league"]
        ).first()
        if exists:
            continue
        rec = LiveOdds(
            league    = ev["league"],
            match_key = key,
            bookmaker = "median_all",
            home_team = ev["home_team"],
            away_team = ev["away_team"],
            commence  = ev.get("commence"),
            odds_h    = ev["odds_h"],
            odds_d    = ev["odds_d"],
            odds_a    = ev["odds_a"],
        )
        session.add(rec)
        saved += 1
    session.commit()
    return saved


def export_signals_csv(signals: list[dict], out_path: Path = None) -> Path:
    """Xuất EV+ signals ra CSV, chỉ giữ các trận có signal"""
    out_path = out_path or PROCESSED_DATA_DIR / "live_signals.csv"
    df = pd.DataFrame(signals)
    # Chỉ export trận có ít nhất 1 EV+ signal
    ev_df = df[df["has_signal"] == True] if "has_signal" in df.columns else df
    ev_df = ev_df.sort_values(
        ["ev_home", "ev_draw", "ev_away"],
        ascending=False
    )
    ev_df.to_csv(out_path, index=False)
    logger.success(f"[LiveSignals] Exported {len(ev_df)} EV+ signals → {out_path}")
    return out_path


# ══════════════════════════════════════════════════════
# PHẦN 5: Master entry point
# ══════════════════════════════════════════════════════

def run_live_signals(leagues: list[str] = None) -> pd.DataFrame:
    """
    Entry point chính:
    1. Fetch odds từ API cho tất cả target leagues
    2. Load lịch sử từ DB
    3. Predict + tính EV+
    4. Lưu DB + export CSV
    5. In bảng tóm tắt
    """
    leagues = leagues or TARGET_LEAGUES
    session = get_session()

    # Load lịch sử
    from data.database import Match
    rows = session.query(Match).all()
    df_history = pd.DataFrame([{
        "date":       m.date,
        "season":     m.season,
        "home_team":  m.home_team,
        "away_team":  m.away_team,
        "home_goals": m.home_goals,
        "away_goals": m.away_goals,
        "result":     m.result,
        "home_shots_target": m.home_shots_target,
        "away_shots_target": m.away_shots_target,
    } for m in rows])

    if df_history.empty:
        logger.error("[LiveSignals] DB trống. Chạy scraper.py trước!")
        session.close()
        return pd.DataFrame()

    all_signals = []
    for league in leagues:
        logger.info(f"[LiveSignals] Processing {league}...")
        raw_events = fetch_live_odds(league)
        parsed     = [e for ev in raw_events if (e := parse_odds_event(ev)) is not None]

        if not parsed:
            logger.warning(f"[LiveSignals] Không có odds hợp lệ cho {league}")
            continue

        save_live_odds_to_db(parsed, session)

        # Filter lịch sử theo league
        df_lg = df_history[df_history.get("league", league) == league] \
            if "league" in df_history.columns else df_history

        signals = predict_upcoming(parsed, df_lg if not df_lg.empty else df_history, league)
        all_signals.extend(signals)

    session.close()

    if not all_signals:
        logger.warning("[LiveSignals] Không có signal nào được tạo ra")
        return pd.DataFrame()

    export_signals_csv(all_signals)

    # In tóm tắt
    df_s = pd.DataFrame(all_signals)
    ev_count = df_s["has_signal"].sum() if "has_signal" in df_s.columns else 0
    logger.success(
        f"[LiveSignals] Tổng: {len(df_s)} trận | "
        f"EV+ signals: {ev_count} | "
        f"Top EV: home={df_s['ev_home'].max():.3f} "
        f"draw={df_s['ev_draw'].max():.3f} "
        f"away={df_s['ev_away'].max():.3f}"
    )
    return df_s


if __name__ == "__main__":
    logger.add("logs/live_signals.log", rotation="10 MB")
    df_signals = run_live_signals()
    if not df_signals.empty:
        ev_rows = df_signals[df_signals["has_signal"] == True]
        print(f"\n{'='*60}")
        print(f"  EV+ SIGNALS ({len(ev_rows)} trận có value bet)")
        print(f"{'='*60}")
        cols = ["home_team", "away_team", "commence",
                "prob_home", "prob_draw", "prob_away",
                "ev_home", "ev_draw", "ev_away"]
        print(ev_rows[cols].to_string(index=False))
