"""
migrate_closing_odds.py
Thêm 3 cột closing odds vào bảng matches (nếu chưa có).
Chạy 1 lần, an toàn với data cũ.
"""
import sqlite3
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from config import DB_PATH

COLS_TO_ADD = [
    ("close_h", "REAL"),
    ("close_d", "REAL"),
    ("close_a", "REAL"),
]

def migrate():
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()
    cur.execute("PRAGMA table_info(matches)")
    existing = {row[1] for row in cur.fetchall()}

    added = 0
    for col_name, col_type in COLS_TO_ADD:
        if col_name not in existing:
            cur.execute(f"ALTER TABLE matches ADD COLUMN {col_name} {col_type}")
            print(f"[Migrate] Added column: {col_name}")
            added += 1
        else:
            print(f"[Migrate] Column already exists: {col_name}")

    conn.commit()
    conn.close()
    print(f"[Migrate] Done. {added} column(s) added.")

if __name__ == "__main__":
    migrate()
