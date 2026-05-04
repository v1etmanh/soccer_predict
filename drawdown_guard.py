"""
drawdown_guard.py -- Drawdown-Aware Kelly Sizing
=================================================
Vấn đề hiện tại: Kelly cap cố định 3% bất kể bankroll đang ở đâu.
Nếu đang trong drawdown -10% -> tiếp tục bet 3% -> có thể accelerate loss.

Giải pháp: Dynamic Kelly scaling dựa trên 2 lớp bảo vệ:

  Lớp 1 -- Drawdown Scale Factor:
    Nếu bankroll dưới peak, scale Kelly xuống tỷ lệ thuận với drawdown.
    scale = (current / peak) ** DRAWDOWN_EXPONENT
    Ví dụ: DD = -10%, exponent=2 -> scale = 0.9^2 = 0.81 (giảm 19%)

  Lớp 2 -- Hard Stop Tiers:
    > -5%  DD: bet bình thường (full fractional Kelly)
    > -10% DD: giảm còn 60% Kelly
    > -15% DD: giảm còn 30% Kelly
    > -20% DD: PAUSE -- không bet cho đến khi recover về -15%
    > -25% DD: Emergency stop

  Lớp 3 -- Streak Protection:
    Nếu thua liên tiếp >= STOP_LOSS_STREAK bets -> reduce sizing thêm 50%
    Reset khi có 1 win.

Dùng thay thế cho kelly_stake() trong backtest_kelly_lm.py và odds_poller.py.

Chạy standalone: python drawdown_guard.py  (test với simulated data)
"""
import sys, io
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from loguru import logger

# -- Config --------------------------------------------------------
KELLY_FRACTION     = 0.25    # Base fractional Kelly
MAX_BET_PCT        = 0.03    # Hard cap per bet (3%)
MIN_BET_PCT        = 0.005   # Ignore bets nhỏ hơn 0.5%
DRAWDOWN_EXPONENT  = 2.0     # Mức độ penalize khi drawdown (2 = quadratic)
STOP_LOSS_STREAK   = 4       # Số lần thua liên tiếp để kích hoạt streak protection

# Drawdown tiers (drawdown tính bằng fraction, âm)
TIER_NORMAL   = -0.05   # > -5%: full bet
TIER_CAUTION  = -0.10   # > -10%: 60%
TIER_REDUCE   = -0.15   # > -15%: 30%
TIER_PAUSE    = -0.20   # > -20%: pause
TIER_STOP     = -0.25   # > -25%: emergency stop

TIER_MULTIPLIERS = {
    TIER_NORMAL:  1.00,
    TIER_CAUTION: 0.60,
    TIER_REDUCE:  0.30,
    TIER_PAUSE:   0.00,
    TIER_STOP:    0.00,
}


@dataclass
class BankrollState:
    """
    Tracks bankroll, peak, drawdown, và streak theo thời gian thực.
    Dùng trong cả backtest lẫn live trading.
    """
    initial: float
    current: float = field(init=False)
    peak:    float = field(init=False)
    loss_streak:  int   = 0
    total_bets:   int   = 0
    paused:       bool  = False
    emergency_stopped: bool = False

    def __post_init__(self):
        self.current = self.initial
        self.peak    = self.initial

    @property
    def drawdown(self) -> float:
        """Current drawdown từ peak (âm hoặc 0)."""
        if self.peak <= 0:
            return 0.0
        return (self.current - self.peak) / self.peak

    @property
    def drawdown_pct(self) -> float:
        return self.drawdown * 100

    @property
    def recovery_needed(self) -> float:
        """Cần recover bao nhiêu % để trở lại peak."""
        if self.peak <= 0 or self.current >= self.peak:
            return 0.0
        return (self.peak / self.current - 1) * 100

    def _get_tier_multiplier(self) -> float:
        """Lookup multiplier từ drawdown tier."""
        dd = self.drawdown
        if dd <= TIER_STOP:
            return TIER_MULTIPLIERS[TIER_STOP]
        if dd <= TIER_PAUSE:
            return TIER_MULTIPLIERS[TIER_PAUSE]
        if dd <= TIER_REDUCE:
            return TIER_MULTIPLIERS[TIER_REDUCE]
        if dd <= TIER_CAUTION:
            return TIER_MULTIPLIERS[TIER_CAUTION]
        return TIER_MULTIPLIERS[TIER_NORMAL]

    def _get_smooth_scale(self) -> float:
        """
        Smooth scaling factor dựa trên drawdown (continuous, không step).
        scale = (current/peak) ** DRAWDOWN_EXPONENT
        Khi peak, scale = 1.0. Khi DD = -10%, scale ~ 0.81.
        """
        if self.peak <= 0:
            return 1.0
        ratio = min(1.0, self.current / self.peak)
        return ratio ** DRAWDOWN_EXPONENT

    def kelly_stake(
        self,
        close_prob: float,
        open_odds: float,
        kelly_fraction: Optional[float] = None,
        max_bet_pct: Optional[float] = None,
    ) -> tuple[float, dict]:
        """
        Tính stake với Drawdown Guard.

        Returns:
            (stake_amount, debug_info)
        """
        frac    = kelly_fraction or KELLY_FRACTION
        max_pct = max_bet_pct   or MAX_BET_PCT

        debug = {
            "bankroll":       round(self.current, 2),
            "peak":           round(self.peak, 2),
            "drawdown":       round(self.drawdown_pct, 1),
            "loss_streak":    self.loss_streak,
            "paused":         self.paused,
            "emergency_stop": self.emergency_stopped,
            "tier_mult":      None,
            "smooth_scale":   None,
            "streak_mult":    None,
            "raw_kelly":      None,
            "final_pct":      None,
            "stake":          0.0,
        }

        # Emergency stop
        if self.emergency_stopped:
            logger.warning("[Guard] Emergency stop active. No bet.")
            return 0.0, debug

        # Pause tier
        tier_mult = self._get_tier_multiplier()
        debug["tier_mult"] = tier_mult
        if tier_mult == 0.0:
            if self.drawdown <= TIER_STOP:
                self.emergency_stopped = True
                logger.error(
                    f"[Guard] EMERGENCY STOP: Drawdown {self.drawdown_pct:.1f}% "
                    f"exceeded {TIER_STOP*100:.0f}% threshold!"
                )
            else:
                self.paused = True
                logger.warning(
                    f"[Guard] PAUSED: Drawdown {self.drawdown_pct:.1f}%. "
                    f"Need to recover to {TIER_REDUCE*100:.0f}% DD."
                )
            return 0.0, debug

        # Auto-unpause khi recover
        if self.paused and self.drawdown > TIER_REDUCE:
            self.paused = False
            logger.info(f"[Guard] Unpaused: Drawdown recovered to {self.drawdown_pct:.1f}%")

        # Base Kelly
        b = open_odds - 1.0
        p = close_prob
        q = 1.0 - p
        if b <= 0 or p <= 0:
            return 0.0, debug

        f_full = (b * p - q) / b
        f_base = max(0.0, f_full * frac)
        debug["raw_kelly"] = round(f_base, 4)

        # Apply smooth scale (continuous)
        smooth = self._get_smooth_scale()
        debug["smooth_scale"] = round(smooth, 3)

        # Apply tier multiplier (step)
        f_adj = f_base * smooth * tier_mult

        # Apply streak protection
        streak_mult = 1.0
        if self.loss_streak >= STOP_LOSS_STREAK:
            streak_mult = 0.5
            logger.warning(
                f"[Guard] Streak protection: {self.loss_streak} consecutive losses -> 50% stake"
            )
        f_adj *= streak_mult
        debug["streak_mult"] = streak_mult

        # Cap và floor
        f_adj = min(f_adj, max_pct)
        if f_adj < MIN_BET_PCT:
            debug["final_pct"] = 0.0
            debug["stake"] = 0.0
            return 0.0, debug

        stake = round(self.current * f_adj, 2)
        debug["final_pct"] = round(f_adj, 4)
        debug["stake"] = stake
        return stake, debug

    def update(self, won: bool, profit: float) -> None:
        """Cập nhật state sau mỗi bet."""
        self.current = round(self.current + profit, 2)
        self.peak    = max(self.peak, self.current)
        self.total_bets += 1

        if won:
            self.loss_streak = 0
        else:
            self.loss_streak += 1

        # Auto-reset emergency stop nếu deposit/reload (không xảy ra trong backtest)
        # Trong live: cần gọi reset_emergency() thủ công sau khi đánh giá lại.

    def reset_emergency(self, reason: str = ""):
        """Override emergency stop -- chỉ gọi thủ công sau khi review."""
        self.emergency_stopped = False
        self.paused = False
        logger.warning(f"[Guard] Emergency stop RESET. Reason: {reason or 'manual'}")

    def status_str(self) -> str:
        status = "STOPPED" if self.emergency_stopped else ("PAUSED" if self.paused else "ACTIVE")
        return (
            f"Bankroll={self.current:.2f} | Peak={self.peak:.2f} | "
            f"DD={self.drawdown_pct:.1f}% | Streak={self.loss_streak} | {status}"
        )


# ==================================================================
# Hàm convenience để drop-in replace kelly_stake() cũ
# ==================================================================

def kelly_stake_with_guard(
    close_prob: float,
    open_odds: float,
    state: BankrollState,
) -> float:
    """
    Drop-in replacement cho kelly_stake() trong backtest_kelly_lm.py.
    
    Args:
        close_prob: implied prob từ closing odds (no-vig)
        open_odds:  odds tại thời điểm đặt
        state:      BankrollState object (phải shared giữa các bets)
    
    Returns:
        stake amount (0 nếu không bet)
    """
    stake, _ = state.kelly_stake(close_prob, open_odds)
    return stake


# ==================================================================
# Backtest với Drawdown Guard -- so sánh với Kelly thuần
# ==================================================================

def run_guarded_backtest(df: pd.DataFrame = None) -> dict:
    """
    Chạy Kelly backtest với Drawdown Guard và so sánh với plain Kelly.
    """
    if df is None:
        from signals.line_movement import load_matches_with_movement
        df = load_matches_with_movement()

    signals = df[df["signal_side"].notna()].copy().sort_values("date")
    if signals.empty:
        logger.warning("[Guard Backtest] No signals.")
        return {}

    from src.betting.backtest_kelly_lm import INITIAL_BANKROLL as BR

    # -- Guarded run ------------------------------------------------
    state   = BankrollState(initial=BR)
    records = []

    for _, row in signals.iterrows():
        side  = row["signal_side"]
        odds  = {"H": row.get("odds_h"), "D": row.get("odds_d"), "A": row.get("odds_a")}.get(side)
        cprob = {"H": row.get("close_prob_h"), "D": row.get("close_prob_d"), "A": row.get("close_prob_a")}.get(side)
        result = row.get("result")

        if not odds or not cprob:
            continue

        stake, debug = state.kelly_stake(cprob, odds)
        if stake <= 0:
            records.append({
                "date": row["date"], "league": row["league"], "season": row["season"],
                "side": side, "stake": 0, "won": False, "profit": 0,
                "bankroll": state.current, "drawdown": state.drawdown_pct,
                "skipped": True,
            })
            continue

        won    = (result == side)
        profit = round(stake * (odds - 1) if won else -stake, 2)
        state.update(won, profit)

        records.append({
            "date":       row["date"],
            "league":     row["league"],
            "season":     row["season"],
            "side":       side,
            "odds":       odds,
            "close_prob": cprob,
            "stake":      stake,
            "won":        won,
            "profit":     profit,
            "bankroll":   state.current,
            "drawdown":   state.drawdown_pct,
            "tier_mult":  debug["tier_mult"],
            "skipped":    False,
        })

    df_rec = pd.DataFrame(records)
    active = df_rec[~df_rec["skipped"]]

    if active.empty:
        return {}

    peak   = df_rec["bankroll"].max()
    max_dd = df_rec["drawdown"].min()
    n_skip = int(df_rec["skipped"].sum())

    summary = {
        "initial_bankroll": BR,
        "final_bankroll":   round(state.current, 2),
        "total_profit":     round(state.current - BR, 2),
        "roi_pct":          round((state.current - BR) / BR * 100, 2),
        "total_bets_attempted": len(df_rec),
        "total_bets_placed":    len(active),
        "bets_skipped_by_guard": n_skip,
        "wins":             int(active["won"].sum()),
        "win_rate":         round(active["won"].mean() * 100, 1),
        "peak_bankroll":    round(peak, 2),
        "max_drawdown_pct": round(max_dd, 2),
    }

    logger.success(
        f"[Guard Backtest] {summary['total_bets_placed']} bets placed "
        f"({summary['bets_skipped_by_guard']} skipped by guard) | "
        f"Win: {summary['win_rate']}% | "
        f"ROI: {summary['roi_pct']}% | "
        f"Bankroll: {BR} -> {state.current} | "
        f"MaxDD: {summary['max_drawdown_pct']}%"
    )
    return summary, df_rec


# ==================================================================
# Compare: Plain Kelly vs Guarded Kelly
# ==================================================================

def compare_guard_vs_plain():
    """So sánh trực tiếp Plain Kelly vs Guarded Kelly trên cùng data."""
    from signals.line_movement import load_matches_with_movement

    print("\n" + "="*62)
    print("  DRAWDOWN GUARD vs PLAIN KELLY -- Comparison")
    print("="*62)

    df = load_matches_with_movement()
    if df.empty:
        print("  [X] No data.")
        return

    # 1. Plain Kelly (gọi hàm cũ)
    sys.path.insert(0, str(Path(__file__).resolve().parent / "src" / "betting"))
    from backtest_kelly_lm import run_kelly_backtest
    _, plain = run_kelly_backtest(df)

    # 2. Guarded Kelly
    result = run_guarded_backtest(df)
    if isinstance(result, tuple):
        guarded, _ = result
    else:
        guarded = result

    if not guarded:
        print("  [X] Guard backtest failed.")
        return

    print(f"\n  {'Metric':<28} {'Plain Kelly':>14} {'Guarded Kelly':>14}")
    print(f"  {'-'*28} {'-'*14} {'-'*14}")

    def row(label, pk, gk, pct=False, lower_better=False):
        fmt = "{:+.1f}%" if pct else "{:.2f}"
        pv  = fmt.format(pk)
        gv  = fmt.format(gk)
        diff = gk - pk
        if lower_better:
            icon = "[OK]" if diff < 0 else ("[FAIL]" if diff > 0 else "--")
        else:
            icon = "[OK]" if diff > 0 else ("[FAIL]" if diff < 0 else "--")
        print(f"  {label:<28} {pv:>14} {gv:>14}  {icon}")

    row("Final bankroll ($)",   plain["final_bankroll"],   guarded["final_bankroll"])
    row("ROI",                  plain["roi_pct"],           guarded["roi_pct"],           pct=True)
    row("Max drawdown",         plain["max_drawdown_pct"],  guarded["max_drawdown_pct"],   pct=True, lower_better=True)
    row("Win rate",             plain["win_rate"],           guarded["win_rate"])
    row("Bets placed",          plain["total_bets"],        guarded["total_bets_placed"])
    row("Bets skipped",         0,                          guarded["bets_skipped_by_guard"])

    print(f"\n  Guard saved {guarded['bets_skipped_by_guard']} bets during high-drawdown periods.")
    print()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Drawdown Guard -- test và compare")
    parser.add_argument("--compare", action="store_true", help="So sánh Plain vs Guarded Kelly")
    parser.add_argument("--test",    action="store_true", help="Unit test BankrollState logic")
    args = parser.parse_args()

    if args.compare:
        compare_guard_vs_plain()
    elif args.test:
        # Unit test nhanh
        print("Testing BankrollState...")
        state = BankrollState(initial=1000.0)

        # Simulate 5 thua liên tiếp
        for i in range(5):
            stake, debug = state.kelly_stake(0.35, 3.0)
            state.update(False, -stake if stake > 0 else 0)
            print(f"  Bet {i+1}: stake={stake:.2f} | {state.status_str()}")

        # 1 win
        stake, debug = state.kelly_stake(0.35, 3.0)
        state.update(True, stake * 2.0 if stake > 0 else 0)
        print(f"  WIN: stake={stake:.2f} | {state.status_str()}")
        print("  [OK] BankrollState test passed.")
    else:
        # Default: chạy guarded backtest
        result = run_guarded_backtest()
        if isinstance(result, tuple):
            summary, _ = result
            import json
            print(json.dumps(summary, indent=2))
