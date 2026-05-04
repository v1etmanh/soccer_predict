"""
run_poller.py — Phase 2+3+4 entry point
=========================================
Chạy odds poller + Telegram alerts + live bet tracking.

Usage:
  # Chạy liên tục (production)
  python run_poller.py

  # Test 1 vòng poll không lưu DB
  python run_poller.py --once --dry-run

  # Chạy 1 vòng thật (lưu DB, gửi Telegram)
  python run_poller.py --once
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from loguru import logger
from config import LOGS_DIR, POLL_INTERVAL_MINUTES, PAPER_TRADING
from signals.telegram_alert import send_startup_message

logger.add(
    LOGS_DIR / "poller_main.log",
    rotation="10 MB",
    retention="30 days",
    level="INFO",
)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Odds Poller — Sharp Money Signal Engine")
    parser.add_argument("--once",    action="store_true", help="Chỉ chạy 1 vòng poll")
    parser.add_argument("--dry-run", action="store_true", help="Không lưu DB, không gửi Telegram")
    args = parser.parse_args()

    mode = "PAPER" if PAPER_TRADING else "REAL MONEY"
    logger.info(f"[Main] Starting poller | Mode: {mode} | Interval: {POLL_INTERVAL_MINUTES}m")

    if not args.dry_run:
        send_startup_message()

    # Import sau khi logger đã setup
    from signals.odds_poller import run_poll_cycle, run_poller

    if args.once:
        signals = run_poll_cycle(dry_run=args.dry_run)
        logger.info(f"[Main] Done. Fired {len(signals)} signals.")
    else:
        run_poller(dry_run=args.dry_run)
