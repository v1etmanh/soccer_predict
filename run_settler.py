"""
run_settler.py — Phase 4: Settle bets + P&L report
====================================================
Chạy sau khi có kết quả trận để settle pending bets.

Usage:
  # Settle tất cả pending bets + in report
  python run_settler.py

  # Preview không lưu DB
  python run_settler.py --dry-run

  # Chỉ in P&L report (không settle)
  python run_settler.py --report

  # Daily summary (settle + Telegram + export CSV)
  python run_settler.py --daily

  # Export CSV
  python run_settler.py --export
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from loguru import logger
from config import LOGS_DIR

logger.add(
    LOGS_DIR / "settler_main.log",
    rotation="5 MB",
    retention="30 days",
    level="INFO",
)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Bet Settler + P&L Report")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report",  action="store_true")
    parser.add_argument("--daily",   action="store_true")
    parser.add_argument("--export",  action="store_true")
    args = parser.parse_args()

    from signals.bet_settler import (
        run_settlement, run_daily_summary,
        compute_pnl_stats, print_pnl_report, export_pnl_csv,
    )
    from data.database import get_session

    if args.daily:
        run_daily_summary()
    elif args.report:
        session = get_session()
        stats = compute_pnl_stats(session)
        print_pnl_report(stats)
        session.close()
    elif args.export:
        session = get_session()
        out = export_pnl_csv(session)
        print(f"Exported → {out}")
        session.close()
    else:
        run_settlement(dry_run=args.dry_run)
        session = get_session()
        stats = compute_pnl_stats(session)
        print_pnl_report(stats)
        session.close()
