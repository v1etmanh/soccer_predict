"""
kelly.py — EV+ filter và Kelly Criterion
Tính stake tối ưu, lọc value bet, quản lý bankroll
"""
import numpy as np
from loguru import logger


def implied_probability(decimal_odds: float) -> float:
    """Chuyển decimal odds → implied probability (đã có overround)"""
    if decimal_odds <= 1.0:
        return 1.0
    return 1.0 / decimal_odds


def remove_overround(odds_h: float, odds_d: float, odds_a: float):
    """
    Loại bỏ overround nhà cái, trả về fair probabilities.
    Dùng phương pháp proportional scaling.
    """
    raw = [implied_probability(o) for o in [odds_h, odds_d, odds_a]]
    total = sum(raw)
    return [p / total for p in raw]


def compute_ev(prob_model: float, decimal_odds: float) -> float:
    """
    Expected Value = prob_model * decimal_odds - 1
    Dương = value bet, âm = không bet
    """
    return prob_model * decimal_odds - 1.0


def kelly_fraction(prob: float, decimal_odds: float, fraction: float = 0.25) -> float:
    """
    Fractional Kelly Criterion
    f* = fraction * (b*p - q) / b
    b = decimal_odds - 1, p = prob thắng, q = 1 - p
    """
    b = decimal_odds - 1.0
    p = prob
    q = 1.0 - p
    if b <= 0 or p <= 0:
        return 0.0
    f_full = (b * p - q) / b
    f_full = max(0.0, f_full)      # Không bao giờ âm
    f_full = min(f_full, 0.25)     # Cap tối đa 25% bankroll dù Kelly nói cao hơn
    return round(f_full * fraction, 4)


def evaluate_bet_signals(
    prob_home: float, prob_draw: float, prob_away: float,
    odds_h: float, odds_d: float, odds_a: float,
    min_ev: float = 0.05,
    kelly_frac: float = 0.25,
    min_odds: float = 1.5,
    max_odds: float = 5.0,
    min_prob: float = 0.30,
) -> dict:
    """
    Tính EV+ và Kelly cho cả 3 outcome của 1 trận.
    Lọc thêm: odds ngoài range, prob model quá thấp.
    """
    results = {}
    sides = {
        "home": (prob_home, odds_h),
        "draw": (prob_draw, odds_d),
        "away": (prob_away, odds_a),
    }
    for side, (prob, odds) in sides.items():
        ev = compute_ev(prob, odds)
        # Lọc odds ngoài range hợp lý
        odds_ok = (odds is not None) and (min_odds <= odds <= max_odds)
        # Lọc khi model không đủ tự tin
        prob_ok = prob >= min_prob
        signal = ev > min_ev and odds_ok and prob_ok
        kelly = kelly_fraction(prob, odds, kelly_frac) if signal else 0.0
        results[side] = {
            "prob":   round(prob, 4),
            "odds":   odds,
            "ev":     round(ev, 4),
            "kelly":  kelly,
            "signal": signal,
        }
    return results


def apply_bankroll(bankroll: float, kelly_frac: float) -> float:
    """Tính stake thực tế từ bankroll và kelly fraction"""
    return round(bankroll * kelly_frac, 2)


class BankrollManager:
    """Theo dõi bankroll theo thời gian"""

    def __init__(self, initial: float = 1000.0):
        self.bankroll   = initial
        self.initial    = initial
        self.history    = [initial]
        self.total_bets = 0
        self.wins       = 0

    def settle_bet(self, stake: float, odds: float, won: bool):
        before = self.bankroll
        if won:
            profit = stake * (odds - 1)
            self.bankroll += profit
            self.wins += 1
        else:
            self.bankroll -= stake
        self.bankroll = round(self.bankroll, 2)
        self.history.append(self.bankroll)
        self.total_bets += 1
        roi = (self.bankroll - self.initial) / self.initial * 100
        logger.info(
            f"[Bankroll] {'WIN' if won else 'LOSS'} | "
            f"stake={stake:.2f} | odds={odds} | "
            f"bankroll={self.bankroll:.2f} | ROI={roi:.1f}%"
        )
        return self.bankroll

    @property
    def roi(self) -> float:
        return round((self.bankroll - self.initial) / self.initial * 100, 2)

    @property
    def win_rate(self) -> float:
        if self.total_bets == 0: return 0.0
        return round(self.wins / self.total_bets * 100, 1)

    def summary(self) -> dict:
        return {
            "initial":    self.initial,
            "current":    self.bankroll,
            "roi_pct":    self.roi,
            "total_bets": self.total_bets,
            "win_rate":   self.win_rate,
            "peak":       max(self.history),
            "max_dd":     round(min(
                (h - max(self.history[:i+1])) / max(self.history[:i+1]) * 100
                for i, h in enumerate(self.history)
            ), 2),
        }
