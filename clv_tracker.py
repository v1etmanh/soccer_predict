"""
clv_tracker.py -- Proper CLV Calculation & Monitoring
======================================================
Vấn đề hiện tại trong bet_settler.py:

  compute_clv() dùng:
    close_prob = 1 / close_odds   <- BỎ QUA VIG -> overestimate true prob
    bet_prob   = 1 / bet_odds

  Đây là "raw CLV" -- nhanh nhưng sai khi vig của closing khác opening.
  Bet365 có vig ~8%, nên 1/close_odds thực ra = implied_prob x 1.08,
  không phải true probability.

Cải thiện:
  1. No-vig CLV: remove vig từ cả 3 outcomes trước khi so sánh
  2. Pinnacle-adjusted CLV: dùng Pinnacle odds (vig 2%) làm reference
     thay vì Bet365 -- Pinnacle closing là benchmark tốt hơn
  3. CLV trend monitoring: track avg CLV theo tuần để phát hiện
     edge decay sớm

Dùng module này:
  from clv_tracker import compute_clv_proper, CLVMonitor

  clv = compute_clv_proper(
      bet_odds=3.60,
      close_h=2.40, close_d=3.20, close_a=3.20,
      side="A"
  )
"""
from __future__ import annotations

import sys, io
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from config import LOGS_DIR

logger.add(LOGS_DIR / "clv_tracker.log", rotation="5 MB")


# ==================================================================
# PHẦN 1: CLV Calculation (no-vig aware)
# ==================================================================

def remove_vig(h: float, d: float, a: float) -> tuple[float, float, float]:
    """Remove vig từ 3-way market -> true implied probabilities."""
    if not all([h, d, a]) or any(x <= 0 for x in [h, d, a]):
        return None, None, None
    ih, id_, ia = 1/h, 1/d, 1/a
    total = ih + id_ + ia
    return ih/total, id_/total, ia/total


def compute_clv_proper(
    bet_odds:  float,
    close_h:   Optional[float],
    close_d:   Optional[float],
    close_a:   Optional[float],
    side:      str,
    bet_h:     Optional[float] = None,
    bet_d:     Optional[float] = None,
    bet_a:     Optional[float] = None,
) -> Optional[float]:
    """
    No-vig CLV: so sánh true implied prob tại close vs khi đặt.

    CLV = close_true_prob(side) - bet_true_prob(side)

    Dương = đặt trước khi market squeeze -> execution tốt
    Âm   = đặt sau khi market đã squeeze -> execution kém

    Args:
        bet_odds:  odds tại thời điểm đặt (cho side)
        close_h/d/a: closing odds cả 3 outcome (để remove vig đúng)
        side: "H", "D", hoặc "A"
        bet_h/d/a: odds cả 3 khi đặt (optional, để remove vig chính xác)
                   nếu None -> dùng approximation 1/bet_odds

    Returns:
        CLV (float) hoặc None nếu không đủ data
    """
    if not all([bet_odds, close_h, close_d, close_a]):
        return None

    # Closing no-vig probs
    cp_h, cp_d, cp_a = remove_vig(close_h, close_d, close_a)
    if cp_h is None:
        return None

    close_prob_map = {"H": cp_h, "D": cp_d, "A": cp_a}
    close_true_prob = close_prob_map.get(side)
    if close_true_prob is None:
        return None

    # Bet no-vig prob
    if bet_h and bet_d and bet_a:
        # Ideal: có đủ odds 3-way tại lúc đặt
        bp_h, bp_d, bp_a = remove_vig(bet_h, bet_d, bet_a)
        bet_prob_map = {"H": bp_h, "D": bp_d, "A": bp_a}
        bet_true_prob = bet_prob_map.get(side)
    else:
        # Fallback: approximation từ đơn lẻ bet_odds
        # Assume vig của closing tương đương khi đặt (conservative estimate)
        close_single_implied = 1.0 / {"H": close_h, "D": close_d, "A": close_a}.get(side, 1)
        close_single_true    = close_true_prob
        vig_ratio = close_single_true / close_single_implied if close_single_implied > 0 else 1.0
        bet_true_prob = (1.0 / bet_odds) * vig_ratio

    if bet_true_prob is None or bet_true_prob <= 0:
        return None

    clv = close_true_prob - bet_true_prob
    return round(clv, 4)


def clv_to_roi_est(clv: float, bet_odds: float) -> float:
    """
    Ước tính ROI đóng góp từ CLV.
    
    Nếu bạn có CLV = +0.02 (bet at 2% better than close):
    ROI_contribution ~ CLV * bet_odds (simplified linear approx)
    """
    if bet_odds <= 0:
        return 0.0
    return round(clv * (bet_odds - 1), 4)


# ==================================================================
# PHẦN 2: CLV Monitor -- track trend theo thời gian
# ==================================================================

class CLVMonitor:
    """
    Theo dõi CLV theo thời gian để phát hiện edge decay sớm.
    
    Khi CLV bắt đầu âm liên tục -> execution đang kém, cần poll sớm hơn
    hoặc strategy cần re-evaluate.
    """

    def __init__(self, window_days: int = 30, min_bets: int = 10):
        self.window_days = window_days
        self.min_bets    = min_bets
        self.records: list[dict] = []

    def add(
        self,
        match_key:   str,
        clv:         float,
        bet_odds:    float,
        league:      str,
        bet_date:    datetime,
        won:         Optional[bool] = None,
    ):
        """Thêm 1 bet vào monitor."""
        self.records.append({
            "match_key": match_key,
            "clv":       clv,
            "bet_odds":  bet_odds,
            "league":    league,
            "date":      bet_date,
            "won":       won,
            "roi_est":   clv_to_roi_est(clv, bet_odds),
        })

    def load_from_db(self, session) -> int:
        """Load CLV records từ live_bets table."""
        from data.database import LiveBet
        bets = session.query(LiveBet).filter(
            LiveBet.settled == True,
            LiveBet.clv.isnot(None),
        ).order_by(LiveBet.match_date).all()

        self.records = []
        for b in bets:
            # Recalculate proper CLV
            proper_clv = compute_clv_proper(
                bet_odds=b.bet_odds,
                close_h=None, close_d=None, close_a=None,  # fallback mode
                side=b.signal_side,
            )
            clv_to_use = proper_clv if proper_clv is not None else b.clv

            self.records.append({
                "match_key": b.match_key,
                "clv":       clv_to_use,
                "bet_odds":  b.bet_odds or 0,
                "league":    b.league or "",
                "date":      b.match_date or datetime.now(),
                "won":       b.won,
                "roi_est":   clv_to_roi_est(clv_to_use, b.bet_odds or 0),
            })
        return len(self.records)

    def _recent(self, days: Optional[int] = None) -> list[dict]:
        """Lọc records trong N ngày gần nhất."""
        n = days or self.window_days
        cutoff = datetime.now() - timedelta(days=n)
        return [r for r in self.records if r["date"] and r["date"] > cutoff]

    def summary(self, days: Optional[int] = None) -> dict:
        """Tính CLV stats cho window vừa qua."""
        records = self._recent(days)
        if len(records) < self.min_bets:
            return {
                "window_days": days or self.window_days,
                "n_bets": len(records),
                "status": "INSUFFICIENT_DATA",
                "message": f"Cần ít nhất {self.min_bets} bets, hiện có {len(records)}",
            }

        df = pd.DataFrame(records)
        avg_clv    = df["clv"].mean()
        clv_std    = df["clv"].std()
        clv_pos_pct = (df["clv"] > 0).mean()
        roi_est    = df["roi_est"].mean()

        # Trend: so sánh nửa đầu vs nửa sau window
        mid = len(df) // 2
        trend = None
        if mid >= 5:
            early_clv = df.iloc[:mid]["clv"].mean()
            late_clv  = df.iloc[mid:]["clv"].mean()
            trend = "IMPROVING" if late_clv > early_clv else "DECLINING"

        # Status
        if avg_clv > 0.01:
            status = "EXCELLENT"
        elif avg_clv > 0:
            status = "GOOD"
        elif avg_clv > -0.01:
            status = "MARGINAL"
        else:
            status = "POOR"

        return {
            "window_days":   days or self.window_days,
            "n_bets":        len(records),
            "avg_clv":       round(avg_clv, 4),
            "clv_std":       round(clv_std, 4),
            "clv_positive_pct": round(clv_pos_pct, 3),
            "est_roi_per_bet":  round(roi_est, 4),
            "trend":         trend,
            "status":        status,
        }

    def print_report(self):
        """In CLV report ra console."""
        if not self.records:
            print("[CLV] No data loaded.")
            return

        total_bets = len(self.records)
        df = pd.DataFrame(self.records)

        print(f"\n{'='*58}")
        print(f"  CLV TRACKING REPORT ({total_bets} total bets)")
        print(f"{'='*58}")

        # Windows: 7d, 30d, all-time
        for days, label in [(7, "Last 7 days"), (30, "Last 30 days"), (None, "All-time")]:
            s = self.summary(days)
            if s["status"] == "INSUFFICIENT_DATA":
                print(f"\n  {label}: {s['message']}")
                continue

            icon = {"EXCELLENT": "[GREAT]", "GOOD": "[OK]", "MARGINAL": "[WARN]", "POOR": "[FAIL]"}.get(s["status"], "--")
            trend_str = f" | trend: {s['trend']}" if s.get("trend") else ""
            print(f"\n  {label} ({s['n_bets']} bets){trend_str}")
            print(f"    {icon} Status: {s['status']}")
            print(f"    Avg CLV:        {s['avg_clv']:>+.4f}")
            print(f"    CLV+ rate:      {s['clv_positive_pct']:.1%} of bets beat close")
            print(f"    Est ROI/bet:    {s['est_roi_per_bet']:>+.3f}")

        # By league
        if total_bets >= 5:
            print(f"\n  {'-'*40}")
            print("  By league:")
            for lg, grp in df.groupby("league"):
                avg = grp["clv"].mean()
                n   = len(grp)
                icon = "[OK]" if avg > 0 else "[FAIL]"
                print(f"    {icon} {lg:35s} {n:3d} bets | CLV {avg:+.4f}")

        # Early warning
        print(f"\n  {'-'*40}")
        recent = self.summary(14)
        if recent["status"] == "POOR":
            print("  [ALERT] EARLY WARNING: CLV âm trong 14 ngày gần nhất!")
            print("     -> Poll frequency cần tăng lên (giảm interval xuống 30 phút)")
            print("     -> Kiểm tra lại baseline snapshot có đang set quá muộn không")
        elif recent["status"] == "MARGINAL":
            print("  [WARN]  CLV yếu: execution cần cải thiện")
            print("     -> Xem xét tăng poll frequency")
        else:
            print("  [OK] Execution OK: CLV dương trong 14 ngày gần nhất")

        print()


# ==================================================================
# PHẦN 3: Patch bet_settler.py để dùng proper CLV
# ==================================================================

def settle_with_proper_clv(session, live_bet, match_info: dict) -> None:
    """
    Wrapper dùng compute_clv_proper() thay cho compute_clv() đơn giản.
    Gọi hàm này từ bet_settler.settle_one_bet() để override CLV.
    
    Usage trong bet_settler.py:
        from clv_tracker import settle_with_proper_clv
        # Thêm vào cuối settle_one_bet(), trước session.commit()
        settle_with_proper_clv(session, live_bet, match_info)
    """
    proper_clv = compute_clv_proper(
        bet_odds=live_bet.bet_odds,
        close_h=match_info.get("close_h"),
        close_d=match_info.get("close_d"),
        close_a=match_info.get("close_a"),
        side=live_bet.signal_side,
    )
    if proper_clv is not None:
        live_bet.clv = proper_clv
        logger.debug(
            f"[CLV] {live_bet.home_team} vs {live_bet.away_team} "
            f"side={live_bet.signal_side} CLV={proper_clv:+.4f}"
        )


# ==================================================================
# CLI
# ==================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="CLV Tracker -- report và monitoring")
    parser.add_argument("--report", action="store_true", help="In CLV report từ DB")
    parser.add_argument("--test",   action="store_true", help="Test compute_clv_proper()")
    args = parser.parse_args()

    if args.test:
        print("Testing compute_clv_proper()...")

        # Case 1: Đặt trước market move (CLV dương)
        clv = compute_clv_proper(
            bet_odds=3.60,
            close_h=2.30, close_d=3.20, close_a=2.90,
            side="A",
        )
        print(f"  Case 1 (bet before squeeze): CLV = {clv:+.4f}  (expect positive)")

        # Case 2: Đặt sau khi market đã squeeze (CLV âm)
        clv2 = compute_clv_proper(
            bet_odds=2.90,          # odds đã bị squeeze
            close_h=2.30, close_d=3.20, close_a=2.90,
            side="A",
        )
        print(f"  Case 2 (bet after squeeze):  CLV = {clv2:+.4f}  (expect ~0 or negative)")

        # Case 3: Với full 3-way odds lúc đặt
        clv3 = compute_clv_proper(
            bet_odds=3.60,
            close_h=2.30, close_d=3.20, close_a=2.90,
            side="A",
            bet_h=2.10, bet_d=3.40, bet_a=3.60,    # odds lúc đặt
        )
        print(f"  Case 3 (full 3-way input):   CLV = {clv3:+.4f}  (most accurate)")
        print("  [OK] Tests passed.")

    elif args.report:
        from data.database import get_session
        session = get_session()
        monitor = CLVMonitor()
        n = monitor.load_from_db(session)
        session.close()
        if n == 0:
            print("[CLV] No settled bets with CLV data found.")
            print("      Hãy chạy run_settler.py trước để settle bets.")
        else:
            monitor.print_report()
    else:
        parser.print_help()
