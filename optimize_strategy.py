"""
optimize_strategy.py — Strategy Optimization via Bootstrap CI
=============================================================
Vấn đề từ bootstrap_ci.py:
  - Overall ROI +12.4% nhưng 95% CI = [-3.6%, +28.0%] — edge chưa xác nhận
  - Bundesliga p(ROI≤0) = 33.5% — gần coin flip, kéo tổng thể xuống
  - Draw signal n=23, CI quá rộng [-20.5%, +165.3%] — vô nghĩa thống kê
  - Away signal n=309, CI [-6.4%, +24.0%] — bulk data nhưng cũng marginal

Giải pháp: Test 4 scenario, so sánh CI để tìm config tối ưu:
  S1: Baseline (current)        — threshold=3%, EPL+BL, sides=D+A
  S2: EPL-only                  — loại Bundesliga
  S3: Threshold 4%              — raise threshold, ít bets hơn nhưng tốt hơn?
  S4: EPL-only + threshold 4%   — kết hợp S2+S3
  S5: Away-only                 — loại Draw signal (n=23 quá nhỏ)
  S6: EPL-only + Away-only      — combo tốt nhất trên paper

Chạy: python optimize_strategy.py
Output: bảng so sánh + JSON tại data/processed/strategy_comparison.json
"""
import sys, io
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import sys, json
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from signals.line_movement import load_matches_with_movement

logger.remove()  # suppress loguru output for cleaner table

BOOTSTRAP_N = 10_000
SEED        = 42

# ── Bootstrap core ────────────────────────────────────────────────

def bootstrap_roi(profits: np.ndarray, n_samples: int = BOOTSTRAP_N, seed: int = SEED) -> dict:
    rng = np.random.default_rng(seed)
    n = len(profits)
    if n < 10:
        return {"error": f"Too few bets (n={n})", "n_bets": n}

    boot_rois = np.array([
        rng.choice(profits, size=n, replace=True).mean()
        for _ in range(n_samples)
    ])
    lower = float(np.percentile(boot_rois, 2.5))
    upper = float(np.percentile(boot_rois, 97.5))
    observed = float(profits.mean())
    p_zero   = float((boot_rois <= 0).mean())

    return {
        "n_bets":       n,
        "observed_roi": round(observed, 4),
        "ci_lower":     round(lower, 4),
        "ci_upper":     round(upper, 4),
        "p_zero":       round(p_zero, 4),
        "significant":  lower > 0,
        "ci_width":     round(upper - lower, 4),
    }

# ── Profit series builder ─────────────────────────────────────────

def build_signals(df: pd.DataFrame, threshold: float, leagues: list, sides: set) -> pd.DataFrame:
    """
    Filter df theo threshold, leagues, sides — tái tính signal_side từ đầu
    để tránh dùng signal_side cũ (được tính với threshold=3%).
    """
    if leagues:
        df = df[df["league"].isin(leagues)].copy()

    results = []
    for _, row in df.iterrows():
        # Tìm side mạnh nhất đủ điều kiện
        candidates = {}
        for side in sides:
            move_col = f"move_{side.lower()}"
            cp_col   = f"close_prob_{side.lower()}"
            op_col   = f"odds_{side.lower()}"

            move = row.get(move_col)
            cp   = row.get(cp_col)
            op   = row.get(op_col)

            if move is None or cp is None:
                continue
            if move >= threshold and 0.20 <= cp <= 0.80 and (op or 0) >= 1.30:
                candidates[side] = move

        if not candidates:
            continue

        best_side = max(candidates, key=candidates.get)
        odds_map  = {"H": row.get("odds_h"), "D": row.get("odds_d"), "A": row.get("odds_a")}
        open_odds = odds_map.get(best_side)
        if not open_odds:
            continue

        won    = row.get("result") == best_side
        profit = (open_odds - 1) if won else -1.0

        results.append({
            "league":      row.get("league"),
            "season":      row.get("season"),
            "signal_side": best_side,
            "open_odds":   open_odds,
            "won":         won,
            "profit":      profit,
            "movement":    candidates[best_side],
        })

    return pd.DataFrame(results)


# ── Scenario definitions ──────────────────────────────────────────

SCENARIOS = [
    {
        "name":      "S1: Baseline",
        "desc":      "threshold=3%, EPL+BL, sides=D+A",
        "threshold": 0.03,
        "leagues":   None,   # None = all leagues
        "sides":     {"D", "A"},
    },
    {
        "name":      "S2: EPL-only",
        "desc":      "threshold=3%, EPL only, sides=D+A",
        "threshold": 0.03,
        "leagues":   ["soccer_epl"],
        "sides":     {"D", "A"},
    },
    {
        "name":      "S3: Threshold 4%",
        "desc":      "threshold=4%, EPL+BL, sides=D+A",
        "threshold": 0.04,
        "leagues":   None,
        "sides":     {"D", "A"},
    },
    {
        "name":      "S4: EPL + Threshold 4%",
        "desc":      "threshold=4%, EPL only, sides=D+A",
        "threshold": 0.04,
        "leagues":   ["soccer_epl"],
        "sides":     {"D", "A"},
    },
    {
        "name":      "S5: Away-only",
        "desc":      "threshold=3%, EPL+BL, side=A only",
        "threshold": 0.03,
        "leagues":   None,
        "sides":     {"A"},
    },
    {
        "name":      "S6: EPL + Away-only",
        "desc":      "threshold=3%, EPL only, side=A only",
        "threshold": 0.03,
        "leagues":   ["soccer_epl"],
        "sides":     {"A"},
    },
    {
        "name":      "S7: EPL + Away + 4%",
        "desc":      "threshold=4%, EPL only, side=A only",
        "threshold": 0.04,
        "leagues":   ["soccer_epl"],
        "sides":     {"A"},
    },
]


# ── Runner ────────────────────────────────────────────────────────

def run_all_scenarios():
    print("\n" + "="*80)
    print("  STRATEGY OPTIMIZATION — Bootstrap CI Comparison")
    print("  Mục tiêu: tìm config có 95% CI lower > 0 (edge xác nhận)")
    print("="*80)

    # Load một lần, reuse cho tất cả scenarios
    print("\n  Loading match data...")
    df_all = load_matches_with_movement()
    if df_all.empty:
        print("  [X] No data found. Run migrate + update_closing_odds first.")
        return

    total_matches = len(df_all)
    print(f"  Loaded {total_matches} matches with closing odds.\n")

    # Header
    col_w = [28, 7, 9, 12, 12, 8, 10]
    header = (
        f"{'Scenario':<{col_w[0]}} "
        f"{'Bets':>{col_w[1]}} "
        f"{'ROI':>{col_w[2]}} "
        f"{'CI lower':>{col_w[3]}} "
        f"{'CI upper':>{col_w[4]}} "
        f"{'p≤0':>{col_w[5]}} "
        f"{'Edge?':>{col_w[6]}}"
    )
    print(f"  {header}")
    print("  " + "-"*80)

    all_results = {}
    best_scenario = None
    best_score    = -999  # score = ci_lower (higher = better)

    for sc in SCENARIOS:
        sdf = build_signals(
            df_all,
            threshold=sc["threshold"],
            leagues=sc["leagues"],
            sides=sc["sides"],
        )

        if sdf.empty:
            row_str = (
                f"  {sc['name']:<{col_w[0]}} "
                f"{'0':>{col_w[1]}} "
                f"{'n/a':>{col_w[2]}} "
                f"{'n/a':>{col_w[3]}} "
                f"{'n/a':>{col_w[4]}} "
                f"{'n/a':>{col_w[5]}} "
                f"{'NO DATA':>{col_w[6]}}"
            )
            print(row_str)
            all_results[sc["name"]] = {"error": "no signals"}
            continue

        res = bootstrap_roi(sdf["profit"].values)
        all_results[sc["name"]] = {**res, "desc": sc["desc"]}

        if "error" in res:
            print(f"  {sc['name']:<{col_w[0]}} {res['error']}")
            continue

        sig_str = "[CONFIRMED]" if res["significant"] else ("[MARGINAL]" if res["ci_upper"] > 0 else "[FAIL]")

        row_str = (
            f"  {sc['name']:<{col_w[0]}} "
            f"{res['n_bets']:>{col_w[1]}} "
            f"{res['observed_roi']:>+{col_w[2]}.1%} "
            f"{res['ci_lower']:>+{col_w[3]}.1%} "
            f"{res['ci_upper']:>+{col_w[4]}.1%} "
            f"{res['p_zero']:>{col_w[5]}.1%} "
            f"{sig_str:>{col_w[6]}}"
        )
        print(row_str)

        # Track best (prioritise: confirmed > ci_lower value)
        score = res["ci_lower"] + (0.1 if res["significant"] else 0)
        if score > best_score:
            best_score    = score
            best_scenario = sc["name"]

    # ── Recommendation ───────────────────────────────────────────
    print("\n" + "="*80)
    print("  RECOMMENDATION")
    print("="*80)

    if best_scenario:
        best = all_results[best_scenario]
        if best.get("significant"):
            verdict = "[EDGE CONFIRMED] — an toàn để chuyển sang paper trading"
        elif best.get("ci_upper", 0) > 0:
            verdict = "[MARGINAL] — cần ~200-300 bets paper trading trước khi real money"
        else:
            verdict = "[FAIL] — cần re-evaluate strategy"

        print(f"\n  Best scenario : {best_scenario}")
        print(f"  Description   : {best.get('desc', '')}")
        print(f"  n bets        : {best.get('n_bets', '?')}")
        print(f"  Observed ROI  : {best.get('observed_roi', 0):+.1%}")
        print(f"  95% CI        : [{best.get('ci_lower', 0):+.1%}, {best.get('ci_upper', 0):+.1%}]")
        print(f"  p(ROI≤0)      : {best.get('p_zero', 0):.1%}")
        print(f"  Verdict       : {verdict}")

    # ── Next steps ───────────────────────────────────────────────
    print("\n" + "-"*80)
    print("  NEXT STEPS (áp dụng config tốt nhất):")
    print("-"*80)
    print("  1. Sửa src/signals/line_movement.py:")
    print("       MOVEMENT_THRESHOLD = 0.04   # nếu S3/S4/S7 tốt hơn")
    print("       ALLOWED_SIDES = {'A'}        # nếu Away-only tốt hơn")
    print("")
    print("  2. Sửa .env:")
    print("       TARGET_LEAGUES=soccer_epl    # nếu EPL-only tốt hơn")
    print("")
    print("  3. Verify bằng:")
    print("       python run_backtest_lm.py")
    print("       python test_by_season.py")
    print("       python bootstrap_ci.py       # CI có chặt hơn không?")
    print("")
    print("  4. Paper trade 1-2 mùa (~100-200 bets) trước real money")
    print("     Track CLV với: python clv_tracker.py --report")
    print("-"*80)

    # ── Save JSON ────────────────────────────────────────────────
    out_path = Path("data/processed/strategy_comparison.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_results, indent=2, default=str), encoding="utf-8")
    print(f"\n  Full results saved -> {out_path}\n")

    return all_results


if __name__ == "__main__":
    run_all_scenarios()
