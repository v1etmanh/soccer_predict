"""
update_closing_odds.py
Tải closing odds cho các trận đã có trong DB (không duplicate).
Chạy sau khi migrate_closing_odds.py.
"""
import sys, io, requests
import pandas as pd
from pathlib import Path
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from config import FD_LEAGUE_MAP, TARGET_LEAGUES
from data.database import get_session, Match

FD_BASE = "https://www.football-data.co.uk/mmz4281"
SEASONS = ["2021", "2122", "2223", "2324", "2425"]

CLOSE_COLS = {
    "B365CH": "close_h",
    "B365CD": "close_d",
    "B365CA": "close_a",
}
KEY_COLS = ["Date", "HomeTeam", "AwayTeam"]


def fetch_closing_odds(league_key: str, season: str) -> pd.DataFrame | None:
    fd_code = FD_LEAGUE_MAP[league_key]["code"]
    url = f"{FD_BASE}/{season}/{fd_code}.csv"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text), encoding="latin-1")
        need = KEY_COLS + [c for c in CLOSE_COLS if c in df.columns]
        if len(need) == len(KEY_COLS):
            logger.warning(f"[Update] {league_key} {season}: không có closing odds trong file")
            return None
        df = df[need].rename(columns=CLOSE_COLS)
        df["date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
        df = df.dropna(subset=["date"])
        return df
    except Exception as e:
        logger.warning(f"[Update] Lỗi {league_key} {season}: {e}")
        return None


def update_db(df: pd.DataFrame, league_key: str, session) -> int:
    updated = 0
    for _, row in df.iterrows():
        home = str(row.get("HomeTeam", "")).strip()
        away = str(row.get("AwayTeam", "")).strip()
        date = row["date"]
        match = session.query(Match).filter_by(
            league=league_key, date=date, home_team=home, away_team=away
        ).first()
        if not match:
            continue
        changed = False
        for col in ["close_h", "close_d", "close_a"]:
            val = row.get(col)
            if pd.notna(val) and getattr(match, col) is None:
                setattr(match, col, float(val))
                changed = True
        if changed:
            updated += 1
    session.commit()
    return updated


def run():
    session = get_session()
    total = 0
    for league in TARGET_LEAGUES:
        if league not in FD_LEAGUE_MAP:
            continue
        for season in SEASONS:
            df = fetch_closing_odds(league, season)
            if df is not None:
                n = update_db(df, league, session)
                logger.info(f"[Update] {league} {season}: {n} trận updated")
                total += n
    session.close()
    logger.success(f"[Update] Hoàn thành. {total} trận có closing odds mới.")
    return total


if __name__ == "__main__":
    logger.add("logs/update_closing.log", rotation="5 MB")
    run()
