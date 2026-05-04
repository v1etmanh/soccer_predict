"""
bootstrap_ci.py -- Bootstrap Confidence Interval cho ROI
=========================================================
Vấn đề: ROI 12.4% từ 332 bets nghe có vẻ tốt, nhưng với n nhỏ,
confidence interval có thể rất rộng -- edge thực tế có thể âm.

Phương pháp Bootstrap (không parametric):
  1. Lấy toàn bộ 332 profit records
  2. Resample ngẫu nhiên có hoàn lại 10,000 lần, mỗi lần n=332
  3. Tính ROI của mỗi sample -> phân phối empirical của ROI
  4. Lấy percentile 2.5% và 97.5% làm 95% CI

Nếu CI lower bound > 0 -> edge thực sự dương với xác suất 97.5%.
Nếu CI lower bound < 0 -> edge chưa được xác nhận thống kê.

Chạy: python bootstrap_ci.py
"""
import sys, io
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from signals.line_movement import load_matches_with_movement
from config import BOOTSTRAP_N, LOGS_DIR

logger.add(LOGS_DIR / "bootstrap_ci.log", rotation="5 MB")

BOOTSTRAP_N_DEFAULT = 10_000


def bootstrap_roi(
    profits: np.ndarray,
    n_samples: int = BOOTSTRAP_N_DEFAULT,
    ci: float = 0.95,
    seed: int = 42,
) -> dict:
    """
    Bootstrap CI cho ROI.

    Args:
        profits: array của profit per bet (-1 nếu thua, odds-1 nếu thắng)
        n_samples: số lần resample
        ci: confidence level (0.95 = 95%)
        seed: random seed để reproducible

    Returns:
        dict với mean_roi, ci_lower, ci_upper, std, và distribution stats
    """
    rng = np.random.default_rng(seed)
    n = len(profits)
    if n < 10:
        logger.error(f"Quá ít bets ({n}) để bootstrap.")
        return {}

    logger.info(f"[Bootstrap] Running {n_samples:,} resamples on {n} bets...")

    boot_rois = np.empty(n_samples)
    for i in range(n_samples):
        sample = rng.choice(profits, size=n, replace=True)
        boot_rois[i] = sample.mean()

    alpha = (1 - ci) / 2
    lower = float(np.percentile(boot_rois, alpha * 100))
    upper = float(np.percentile(boot_rois, (1 - alpha) * 100))
    mean  = float(np.mean(boot_rois))
    std   = float(np.std(boot_rois))
    observed_roi = float(profits.mean())

    # Tính p-value: tỉ lệ bootstrap samples có ROI <= 0
    p_value = float((boot_rois <= 0).mean())

    return {
        "observed_roi":  round(observed_roi, 4),
        "n_bets":        n,
        "n_bootstrap":   n_samples,
        "ci_level":      ci,
        "ci_lower":      round(lower, 4),
        "ci_upper":      round(upper, 4),
        "mean_boot_roi": round(mean, 4),
        "std_boot_roi":  round(std, 4),
        "p_value_zero":  round(p_value, 4),  # prob(ROI <= 0) under bootstrap dist
        "significant":   lower > 0,
    }


def bootstrap_by_group(df: pd.DataFrame, group_col: str, profits_col: str = "profit") -> dict:
    """Bootstrap CI per league hoặc per season."""
    results = {}
    for key, group in df.groupby(group_col):
        profits = group[profits_col].values
        if len(profits) < 10:
            results[key] = {"n_bets": len(profits), "error": "Too few bets"}
            continue
        results[key] = bootstrap_roi(profits, n_samples=5_000)
    return results


def build_profit_series(df: pd.DataFrame) -> pd.DataFrame:
    """Từ df có signal_side và result, tạo cột profit."""
    signals = df[df["signal_side"].notna()].copy()
    if signals.empty:
        return pd.DataFrame()

    def get_info(row):
        side = row["signal_side"]
        odds = {"H": row.get("odds_h"), "D": row.get("odds_d"), "A": row.get("odds_a")}.get(side)
        won  = row.get("result") == side
        if odds is None:
            return None, None
        profit = (odds - 1) if won else -1.0
        return odds, profit

    signals[["bet_odds", "profit"]] = signals.apply(
        lambda r: pd.Series(get_info(r)), axis=1
    )
    return signals.dropna(subset=["profit"])


def print_results(label: str, result: dict):
    if not result:
        print(f"  {label}: No data")
        return
    if "error" in result:
        print(f"  {label}: {result['error']} (n={result.get('n_bets', '?')})")
        return

    sig_icon = "[OK]" if result["significant"] else "[FAIL]"
    print(
        f"  {sig_icon} {label:<25} "
        f"ROI {result['observed_roi']:>+7.1%}  "
        f"95% CI [{result['ci_lower']:>+7.1%}, {result['ci_upper']:>+7.1%}]  "
        f"p(ROI≤0)={result['p_value_zero']:.3f}  "
        f"n={result['n_bets']}"
    )


def run_full_bootstrap():
    print("\n" + "="*72)
    print("  BOOTSTRAP CONFIDENCE INTERVAL -- pipeline_dish Line Movement Edge")
    print("="*72)

    df = load_matches_with_movement()
    if df.empty:
        print("  [X] No data. Run migrate + update_closing_odds first.")
        return

    signals = build_profit_series(df)
    if signals.empty:
        print("  [X] No signals found.")
        return

    profits = signals["profit"].values

    # -- 1. Overall -------------------------------------------------
    print(f"\n{'-'*72}")
    print("  OVERALL")
    print(f"{'-'*72}")
    overall = bootstrap_roi(profits)
    print_results("All signals", overall)

    if overall:
        print()
        if overall["significant"]:
            print(f"  [OK] Edge CONFIRMED: 95% CI hoàn toàn dương -> edge thật")
            print(f"     Expected ROI mỗi bet: {overall['ci_lower']:+.1%} đến {overall['ci_upper']:+.1%}")
        elif overall["ci_upper"] > 0:
            print(f"  [WARN] Edge MARGINAL: CI cắt qua 0 -> edge không đủ mạnh để khẳng định")
            print(f"     Cần thêm bets hoặc raise threshold để tăng confidence.")
            print(f"     Có thể cần ~{int((1.96 * overall['std_boot_roi'] / abs(overall['observed_roi']))**2)} bets để xác nhận.")
        else:
            print(f"  [FAIL] Edge KHÔNG XÁC NHẬN: CI < 0 -> edge có thể không tồn tại")

    # -- 2. By League -----------------------------------------------
    print(f"\n{'-'*72}")
    print("  BY LEAGUE")
    print(f"{'-'*72}")
    by_league = bootstrap_by_group(signals, "league")
    for k, v in by_league.items():
        print_results(k, v)

    # -- 3. By Season -----------------------------------------------
    print(f"\n{'-'*72}")
    print("  BY SEASON")
    print(f"{'-'*72}")
    by_season = bootstrap_by_group(signals, "season")
    for k, v in sorted(by_season.items()):
        print_results(k, v)

    # -- 4. By Side ------------------------------------------------
    print(f"\n{'-'*72}")
    print("  BY SIGNAL SIDE")
    print(f"{'-'*72}")
    by_side = bootstrap_by_group(signals, "signal_side")
    for k, v in sorted(by_side.items()):
        print_results(k, v)

    # -- 5. Save full result ----------------------------------------
    output = {
        "overall":    overall,
        "by_league":  by_league,
        "by_season":  by_season,
        "by_side":    by_side,
    }
    out_path = Path("data/processed/bootstrap_ci_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, default=str))
    print(f"\n  Results saved -> {out_path}")
    print()

    return output


if __name__ == "__main__":
    run_full_bootstrap()
