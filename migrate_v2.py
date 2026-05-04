"""
migrate_v2.py — One-time migration: thêm Pinnacle odds columns vào bảng matches
Cập nhật: thêm kiểm tra an toàn, idempotent hoàn toàn.
Chạy: python migrate_v2.py
"""
import sys, sqlite3
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from config import DB_PATH

NEW_COLUMNS = [
    ("pin_h",       "REAL"),
    ("pin_d",       "REAL"),
    ("pin_a",       "REAL"),
    ("pin_close_h", "REAL"),
    ("pin_close_d", "REAL"),
    ("pin_close_a", "REAL"),
]

def migrate():
    print(f"[Migrate v2] DB path: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()

    # Kiểm tra bảng matches có tồn tại không
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='matches'")
    if not cur.fetchone():
        print("[Migrate v2] ERROR: Bảng 'matches' chưa tồn tại. Chạy init_db() trước.")
        conn.close()
        return

    cur.execute("PRAGMA table_info(matches)")
    existing = {row[1] for row in cur.fetchall()}

    added = 0
    for col_name, col_type in NEW_COLUMNS:
        if col_name not in existing:
            cur.execute(f"ALTER TABLE matches ADD COLUMN {col_name} {col_type}")
            print(f"  [+] Added: matches.{col_name} {col_type}")
            added += 1
        else:
            print(f"  [=] Exists: matches.{col_name}")

    # Đảm bảo odds_snapshots table tồn tại (Phase 2)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS odds_snapshots (
            id          INTEGER PRIMARY KEY,
            match_key   TEXT NOT NULL,
            league      TEXT,
            home_team   TEXT,
            away_team   TEXT,
            match_date  DATETIME,
            snapshot_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            bookmaker   TEXT DEFAULT 'median',
            odds_h      REAL,
            odds_d      REAL,
            odds_a      REAL,
            is_baseline INTEGER DEFAULT 0
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS ix_snapshots_match_key ON odds_snapshots(match_key)")
    cur.execute("CREATE INDEX IF NOT EXISTS ix_snapshots_league ON odds_snapshots(league, match_date)")

    # Đảm bảo live_bets table tồn tại (Phase 4)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS live_bets (
            id              INTEGER PRIMARY KEY,
            match_key       TEXT,
            league          TEXT,
            season          TEXT,
            home_team       TEXT,
            away_team       TEXT,
            match_date      DATETIME,
            signal_side     TEXT,
            movement        REAL,
            baseline_odds   REAL,
            bet_odds        REAL,
            stake           REAL,
            bankroll_before REAL,
            result          TEXT,
            won             INTEGER,
            profit          REAL,
            bankroll_after  REAL,
            close_odds      REAL,
            clv             REAL,
            signal_fired_at DATETIME,
            bet_placed_at   DATETIME,
            settled         INTEGER DEFAULT 0,
            is_paper        INTEGER DEFAULT 1,
            notes           TEXT
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS ix_live_bets_match ON live_bets(match_key)")
    cur.execute("CREATE INDEX IF NOT EXISTS ix_live_bets_settled ON live_bets(settled)")

    conn.commit()
    conn.close()
    print(f"\n[Migrate v2] Done. {added} column(s) added. Tables ensured.")

if __name__ == "__main__":
    migrate()
