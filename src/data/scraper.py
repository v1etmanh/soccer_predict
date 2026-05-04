"""
scraper.py v2 — Tải historical data từ football-data.co.uk
Thêm: Pinnacle odds (PSH/PSD/PSA, PSCH/PSCD/PSCA) + 3 leagues mới
"""
import requests, io
import pandas as pd
from loguru import logger
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import RAW_DATA_DIR, FD_LEAGUE_MAP, TARGET_LEAGUES
from data.database import get_session, Match

FD_BASE  = "https://www.football-data.co.uk/mmz4281"
SEASONS  = ["2021", "2122", "2223", "2324", "2425"]

COLS = {
    "Date": "date",
    "HomeTeam": "home_team", "AwayTeam": "away_team",
    "FTHG": "home_goals",    "FTAG": "away_goals",
    "FTR":  "result",
    # Bet365 opening
    "B365H": "odds_h", "B365D": "odds_d", "B365A": "odds_a",
    # Bet365 closing
    "B365CH": "close_h", "B365CD": "close_d", "B365CA": "close_a",
    # Pinnacle opening (sharper market, ~2-3% vig vs Bet365 ~5-6%)
    "PSH": "pin_h", "PSD": "pin_d", "PSA": "pin_a",
    # Pinnacle closing
    "PSCH": "pin_close_h", "PSCD": "pin_close_d", "PSCA": "pin_close_a",
    # Stats
    "HS": "home_shots",   "AS": "away_shots",
    "HST": "home_shots_target", "AST": "away_shots_target",
    "HC": "home_corners", "AC": "away_corners",
}


def download_season(league_key: str, season: str) -> pd.DataFrame | None:
    fd_code = FD_LEAGUE_MAP[league_key]["code"]
    url = f"{FD_BASE}/{season}/{fd_code}.csv"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text), encoding="latin-1")
        keep = [c for c in COLS if c in df.columns]
        df = df[keep].rename(columns={c: COLS[c] for c in keep})
        df["league"] = league_key
        df["season"] = f"20{season[:2]}-{season[2:]}"
        df["date"]   = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
        df = df.dropna(subset=["date", "home_team", "away_team", "home_goals"])
        n_pin = df["pin_h"].notna().sum() if "pin_h" in df.columns else 0
        logger.info(f"[Scraper] {league_key} {season}: {len(df)} matches, Pinnacle: {n_pin}")
        return df
    except Exception as e:
        logger.warning(f"[Scraper] Error {league_key} {season}: {e}")
        return None


def _float(val):
    try:
        v = float(val)
        return v if v > 0 else None
    except (TypeError, ValueError):
        return None


def save_to_db(df: pd.DataFrame, session) -> int:
    saved = 0
    for _, row in df.iterrows():
        exists = session.query(Match).filter_by(
            league=row["league"], date=row["date"],
            home_team=row["home_team"], away_team=row["away_team"],
        ).first()
        if exists:
            # Backfill Pinnacle odds nếu chưa có
            changed = False
            for col in ["pin_h","pin_d","pin_a","pin_close_h","pin_close_d","pin_close_a"]:
                if col in row.index and pd.notna(row.get(col)) and getattr(exists, col, None) is None:
                    setattr(exists, col, _float(row[col]))
                    changed = True
            if changed:
                saved += 1  # count backfill as saved
            continue

        match = Match(
            league=row["league"], season=row.get("season",""),
            date=row["date"], home_team=row["home_team"], away_team=row["away_team"],
            home_goals=int(row.get("home_goals",0) or 0),
            away_goals=int(row.get("away_goals",0) or 0),
            result=str(row.get("result","")),
            odds_h=_float(row.get("odds_h")), odds_d=_float(row.get("odds_d")),
            odds_a=_float(row.get("odds_a")),
            close_h=_float(row.get("close_h")), close_d=_float(row.get("close_d")),
            close_a=_float(row.get("close_a")),
            pin_h=_float(row.get("pin_h")), pin_d=_float(row.get("pin_d")),
            pin_a=_float(row.get("pin_a")),
            pin_close_h=_float(row.get("pin_close_h")),
            pin_close_d=_float(row.get("pin_close_d")),
            pin_close_a=_float(row.get("pin_close_a")),
            home_shots=int(row["home_shots"]) if pd.notna(row.get("home_shots")) else None,
            away_shots=int(row["away_shots"]) if pd.notna(row.get("away_shots")) else None,
            home_shots_target=int(row["home_shots_target"]) if pd.notna(row.get("home_shots_target")) else None,
            away_shots_target=int(row["away_shots_target"]) if pd.notna(row.get("away_shots_target")) else None,
            home_corners=int(row["home_corners"]) if pd.notna(row.get("home_corners")) else None,
            away_corners=int(row["away_corners"]) if pd.notna(row.get("away_corners")) else None,
        )
        session.add(match)
        saved += 1
    session.commit()
    return saved


def run_scraper(leagues=None, seasons=None):
    leagues = leagues or TARGET_LEAGUES
    seasons = seasons or SEASONS
    session = get_session()
    total   = 0
    for league in leagues:
        if league not in FD_LEAGUE_MAP:
            logger.warning(f"[Scraper] Unsupported league: {league}")
            continue
        for season in seasons:
            df = download_season(league, season)
            if df is not None:
                n = save_to_db(df, session)
                logger.info(f"[Scraper] Saved {n} rows — {league} {season}")
                total += n
    session.close()
    logger.success(f"[Scraper] Done. Total: {total} new/updated rows.")
    return total


if __name__ == "__main__":
    logger.add("logs/scraper.log", rotation="10 MB")
    run_scraper()
