"""
app.py — Streamlit Dashboard
Theo dõi: EV+ Signals | Bankroll Curve | P&L | Calibration Chart

Chạy:
    cd D:/dream_project/pipeline_dish
    streamlit run src/dashboard/app.py
"""
import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
from config import (
    PROCESSED_DATA_DIR, MODELS_DIR, INITIAL_BANKROLL,
    MIN_EV_THRESHOLD, KELLY_FRACTION
)

# ── Page config ────────────────────────────────────────
st.set_page_config(
    page_title="⚽ Football Betting Dashboard",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #1e2130;
        border-radius: 12px;
        padding: 20px;
        border-left: 4px solid #00d4aa;
        margin: 6px 0;
    }
    .signal-positive { color: #00d4aa; font-weight: bold; }
    .signal-negative { color: #ff4b6e; font-weight: bold; }
    .stTabs [data-baseweb="tab"] { font-size: 16px; padding: 12px 24px; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
# DATA LOADERS (cached)
# ══════════════════════════════════════════════════════

@st.cache_data(ttl=300)
def load_backtest_results() -> pd.DataFrame:
    path = PROCESSED_DATA_DIR / "backtest_results.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["date"])
    return df.sort_values("date")


@st.cache_data(ttl=60)
def load_live_signals() -> pd.DataFrame:
    path = PROCESSED_DATA_DIR / "live_signals.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "commence" in df.columns:
        df["commence"] = pd.to_datetime(df["commence"], errors="coerce", utc=True)
    return df


@st.cache_data(ttl=600)
def load_db_matches() -> pd.DataFrame:
    try:
        from data.database import get_session, Match
        session = get_session()
        rows = session.query(Match).all()
        session.close()
        return pd.DataFrame([{
            "date": m.date, "league": m.league, "season": m.season,
            "home_team": m.home_team, "away_team": m.away_team,
            "result": m.result, "home_goals": m.home_goals,
            "away_goals": m.away_goals,
        } for m in rows])
    except Exception:
        return pd.DataFrame()


def compute_bankroll_curve(df: pd.DataFrame) -> pd.DataFrame:
    """Tính bankroll curve từ backtest results"""
    if df.empty or "bankroll" not in df.columns:
        return pd.DataFrame()
    curve = df[["date", "bankroll", "profit", "season"]].copy()
    curve["cumulative_profit"] = df["profit"].cumsum()
    return curve


# ══════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════

def render_sidebar(df_bt: pd.DataFrame) -> dict:
    st.sidebar.image("https://img.icons8.com/fluency/96/football.png", width=60)
    st.sidebar.title("⚽ Pipeline Dashboard")
    st.sidebar.markdown("---")

    filters = {}
    if not df_bt.empty and "season" in df_bt.columns:
        seasons = ["All"] + sorted(df_bt["season"].unique().tolist())
        filters["season"] = st.sidebar.selectbox("📅 Season", seasons)
        filters["side"]   = st.sidebar.multiselect(
            "🎯 Bet Side", ["home", "draw", "away"],
            default=["home", "draw", "away"]
        )
        filters["min_ev"] = st.sidebar.slider(
            "Min EV threshold", 0.0, 0.20, float(MIN_EV_THRESHOLD), 0.005
        )

    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Refresh Signals", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.sidebar.markdown(f"""
    **Config**
    - Initial Bankroll: `${INITIAL_BANKROLL:,.0f}`
    - Kelly Fraction: `{KELLY_FRACTION*100:.0f}%`
    - Min EV: `{MIN_EV_THRESHOLD*100:.1f}%`
    """)
    return filters


# ══════════════════════════════════════════════════════
# TAB 1: EV+ LIVE SIGNALS
# ══════════════════════════════════════════════════════

def render_signals_tab(df_signals: pd.DataFrame):
    st.header("🎯 Live EV+ Signals")

    if df_signals.empty:
        st.warning("⚠️ Chưa có signals. Chạy `python src/signals/live_signals.py` để fetch odds.")
        st.code("cd D:/dream_project/pipeline_dish\npython src/signals/live_signals.py")
        return

    total     = len(df_signals)
    ev_count  = int(df_signals.get("has_signal", pd.Series(False)).sum())
    best_ev   = max(
        df_signals["ev_home"].max(),
        df_signals["ev_draw"].max(),
        df_signals["ev_away"].max()
    ) if "ev_home" in df_signals.columns else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📋 Total Matches", total)
    col2.metric("✅ EV+ Signals", ev_count)
    col3.metric("📈 Best EV", f"{best_ev:.1%}")
    col4.metric("🔍 EV Threshold", f"{MIN_EV_THRESHOLD:.1%}")

    st.markdown("---")

    # Filter chỉ hiện EV+ matches
    show_all = st.toggle("Show all matches (kể cả không có EV+)", False)
    if not show_all and "has_signal" in df_signals.columns:
        df_show = df_signals[df_signals["has_signal"] == True]
    else:
        df_show = df_signals

    if df_show.empty:
        st.info("Không có EV+ signal trong vòng tiếp theo.")
        return

    # Format table
    display_cols = {
        "home_team": "Home", "away_team": "Away", "commence": "Kick-off",
        "odds_h": "Odds H", "odds_d": "Odds D", "odds_a": "Odds A",
        "prob_home": "P(H)", "prob_draw": "P(D)", "prob_away": "P(A)",
        "ev_home": "EV H", "ev_draw": "EV D", "ev_away": "EV A",
        "kelly_home": "Kelly H", "kelly_draw": "Kelly D", "kelly_away": "Kelly A",
    }
    disp = df_show[[c for c in display_cols if c in df_show.columns]].rename(columns=display_cols)

    # Highlight EV+ cells
    def highlight_ev(val):
        try:
            v = float(val)
            if v > MIN_EV_THRESHOLD:
                return "background-color: #0d3b2e; color: #00d4aa; font-weight: bold"
            elif v > 0:
                return "color: #ffa94d"
            return "color: #888"
        except Exception:
            return ""

    ev_cols = ["EV H", "EV D", "EV A"]
    ev_present = [c for c in ev_cols if c in disp.columns]
    styled = disp.style.applymap(highlight_ev, subset=ev_present)

    # Format numeric
    for col in ["P(H)", "P(D)", "P(A)"]:
        if col in disp.columns:
            styled = styled.format({col: "{:.1%}"})
    for col in ev_present:
        styled = styled.format({col: "{:+.3f}"})

    st.dataframe(styled, use_container_width=True, height=450)

    # EV bar chart
    if ev_count > 0 and "ev_home" in df_show.columns:
        fig = go.Figure()
        match_labels = df_show["home_team"] + " vs " + df_show["away_team"]
        for side, col, color in [
            ("Home", "ev_home", "#00d4aa"),
            ("Draw", "ev_draw", "#ffa94d"),
            ("Away", "ev_away", "#ff4b6e"),
        ]:
            if col in df_show.columns:
                fig.add_trace(go.Bar(
                    name=side, x=match_labels, y=df_show[col],
                    marker_color=color, opacity=0.85,
                ))
        fig.add_hline(y=MIN_EV_THRESHOLD, line_dash="dash",
                      line_color="white", annotation_text=f"Min EV ({MIN_EV_THRESHOLD:.1%})")
        fig.update_layout(
            title="Expected Value per Outcome",
            barmode="group", height=400,
            plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
            font=dict(color="white"),
            yaxis=dict(tickformat="+.1%"),
        )
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════
# TAB 2: BANKROLL CURVE
# ══════════════════════════════════════════════════════

def render_bankroll_tab(df_bt: pd.DataFrame, filters: dict):
    st.header("💰 Bankroll Curve & P&L")

    if df_bt.empty:
        st.warning("⚠️ Chưa có dữ liệu backtest. Chạy `python src/betting/backtest.py` trước.")
        st.code("cd D:/dream_project/pipeline_dish\npython src/betting/backtest.py")
        return

    # Apply filters
    df = df_bt.copy()
    if filters.get("season") and filters["season"] != "All":
        df = df[df["season"] == filters["season"]]
    if filters.get("side"):
        df = df[df["side"].isin(filters["side"])]
    if filters.get("min_ev"):
        df = df[df["ev"] >= filters["min_ev"]]

    if df.empty:
        st.info("Không có dữ liệu với filter hiện tại.")
        return

    # Summary metrics
    total_bets  = len(df)
    wins        = df["won"].sum()
    win_rate    = wins / total_bets * 100
    total_profit = df["profit"].sum()
    roi         = total_profit / (df["stake"].sum() + 1e-9) * 100
    final_br    = df["bankroll"].iloc[-1] if "bankroll" in df.columns else INITIAL_BANKROLL
    peak_br     = df["bankroll"].max() if "bankroll" in df.columns else INITIAL_BANKROLL
    max_dd      = 0
    if "bankroll" in df.columns:
        roll_max = df["bankroll"].cummax()
        dd_series = (df["bankroll"] - roll_max) / roll_max * 100
        max_dd = dd_series.min()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Bets", f"{total_bets:,}")
    c2.metric("Win Rate", f"{win_rate:.1f}%")
    c3.metric("Total Profit", f"${total_profit:,.0f}",
              delta=f"{roi:.1f}% ROI")
    c4.metric("Final Bankroll", f"${final_br:,.0f}",
              delta=f"${final_br - INITIAL_BANKROLL:+,.0f}")
    c5.metric("Max Drawdown", f"{max_dd:.1f}%")

    st.markdown("---")

    # Bankroll Curve
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.65, 0.35],
        subplot_titles=["Bankroll over Time", "Rolling Win Rate (20 bets)"],
        vertical_spacing=0.1,
    )

    if "bankroll" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["bankroll"],
            mode="lines", name="Bankroll",
            line=dict(color="#00d4aa", width=2),
            fill="tozeroy", fillcolor="rgba(0,212,170,0.07)",
        ), row=1, col=1)
        fig.add_hline(
            y=INITIAL_BANKROLL, line_dash="dot",
            line_color="#888", annotation_text="Initial",
            row=1, col=1,
        )
        # Peak highlight
        fig.add_hline(
            y=peak_br, line_dash="dash",
            line_color="#ffa94d",
            annotation_text=f"Peak ${peak_br:,.0f}",
            row=1, col=1,
        )

    # Rolling win rate
    df["won_num"]   = df["won"].astype(int)
    df["roll_wr"]   = df["won_num"].rolling(20, min_periods=5).mean() * 100
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["roll_wr"],
        mode="lines", name="Rolling Win Rate",
        line=dict(color="#ffa94d", width=1.5),
    ), row=2, col=1)
    fig.add_hline(y=win_rate, line_dash="dash",
                  line_color="#888", row=2, col=1)

    fig.update_layout(
        height=600, showlegend=True,
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="white"),
        margin=dict(t=40, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Monthly P&L heatmap
    st.subheader("📅 Monthly P&L")
    df["month"] = df["date"].dt.to_period("M").astype(str)
    monthly = df.groupby("month")["profit"].sum().reset_index()
    monthly.columns = ["month", "profit"]
    colors = ["#ff4b6e" if p < 0 else "#00d4aa" for p in monthly["profit"]]
    fig2 = go.Figure(go.Bar(
        x=monthly["month"], y=monthly["profit"],
        marker_color=colors, name="Monthly P&L"
    ))
    fig2.update_layout(
        height=280, plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="white"), yaxis_tickprefix="$",
    )
    st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════
# TAB 3: P&L TABLE (chi tiết backtest)
# ══════════════════════════════════════════════════════

def render_pnl_tab(df_bt: pd.DataFrame, filters: dict):
    st.header("📊 Backtest P&L Detail")

    if df_bt.empty:
        st.warning("Chưa có dữ liệu backtest.")
        return

    df = df_bt.copy()
    if filters.get("season") and filters["season"] != "All":
        df = df[df["season"] == filters["season"]]
    if filters.get("side"):
        df = df[df["side"].isin(filters.get("side", ["home","draw","away"]))]

    # Season breakdown
    st.subheader("📈 Performance by Season")
    if "season" in df.columns:
        season_stats = df.groupby("season").agg(
            Bets=("profit", "count"),
            Wins=("won", "sum"),
            Profit=("profit", "sum"),
            Stake=("stake", "sum"),
        ).reset_index()
        season_stats["Win%"]  = (season_stats["Wins"] / season_stats["Bets"] * 100).round(1)
        season_stats["ROI%"]  = (season_stats["Profit"] / season_stats["Stake"] * 100).round(1)
        season_stats["Profit"] = season_stats["Profit"].round(2)

        def color_profit(val):
            try:
                return "color: #00d4aa" if float(val) > 0 else "color: #ff4b6e"
            except Exception:
                return ""

        st.dataframe(
            season_stats.style.applymap(color_profit, subset=["Profit", "ROI%"]),
            use_container_width=True, hide_index=True,
        )

    # Bet side breakdown
    st.subheader("🎯 Performance by Bet Side")
    side_stats = df.groupby("side").agg(
        Bets=("profit", "count"),
        Wins=("won", "sum"),
        AvgEV=("ev", "mean"),
        Profit=("profit", "sum"),
    ).reset_index()
    side_stats["Win%"]  = (side_stats["Wins"] / side_stats["Bets"] * 100).round(1)
    side_stats["AvgEV"] = side_stats["AvgEV"].round(3)
    side_stats["Profit"] = side_stats["Profit"].round(2)
    st.dataframe(side_stats, use_container_width=True, hide_index=True)

    # Raw bet table
    st.subheader("📋 All Bets")
    display = df[["date", "season", "home_team", "away_team", "side",
                  "prob", "odds", "ev", "stake", "won", "profit", "bankroll"]].copy()
    display["prob"]   = display["prob"].map("{:.1%}".format)
    display["ev"]     = display["ev"].map("{:+.3f}".format)
    display["stake"]  = display["stake"].map("${:.2f}".format)
    display["profit"] = display["profit"].map("${:+.2f}".format)
    display["bankroll"] = display["bankroll"].map("${:,.0f}".format)

    def color_won(val):
        return "color: #00d4aa" if val is True or val == True else "color: #ff4b6e"

    st.dataframe(
        display.style.applymap(color_won, subset=["won"]),
        use_container_width=True, height=500,
    )

    # Download button
    csv = df_bt.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Full Backtest CSV",
        data=csv,
        file_name="backtest_results.csv",
        mime="text/csv",
    )


# ══════════════════════════════════════════════════════
# TAB 4: CALIBRATION CHART (Reliability Diagram)
# ══════════════════════════════════════════════════════

def render_calibration_tab(df_bt: pd.DataFrame):
    st.header("🎛️ Model Calibration — Reliability Diagram")

    if df_bt.empty or "prob" not in df_bt.columns:
        st.warning("Chưa có dữ liệu backtest với xác suất.")
        return

    st.markdown("""
    **Reliability Diagram** cho thấy model có *well-calibrated* không:
    - Đường **diagonal** = perfect calibration (prob = actual freq)
    - Đường **trên** = underconfident (model underbets)
    - Đường **dưới** = overconfident (model overbets)
    """)

    df = df_bt.dropna(subset=["prob", "won"]).copy()

    # Bin probabilities
    n_bins   = 10
    df["prob_bin"] = pd.cut(df["prob"], bins=n_bins, labels=False, include_lowest=True)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    bin_stats = df.groupby("prob_bin").agg(
        mean_prob=("prob", "mean"),
        actual_freq=("won", "mean"),
        count=("won", "count"),
    ).reset_index()

    fig = go.Figure()

    # Perfect calibration line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines", name="Perfect Calibration",
        line=dict(color="#888", dash="dash", width=1.5),
    ))

    # Calibration curve
    fig.add_trace(go.Scatter(
        x=bin_stats["mean_prob"],
        y=bin_stats["actual_freq"],
        mode="lines+markers",
        name="Model Calibration",
        line=dict(color="#00d4aa", width=2.5),
        marker=dict(
            size=bin_stats["count"] / bin_stats["count"].max() * 20 + 5,
            color="#00d4aa",
            line=dict(color="white", width=1),
        ),
        text=[f"n={int(c)}" for c in bin_stats["count"]],
        hovertemplate="Pred: %{x:.2f}<br>Actual: %{y:.2f}<br>%{text}<extra></extra>",
    ))

    # Confidence band
    fig.add_trace(go.Scatter(
        x=np.concatenate([bin_stats["mean_prob"], bin_stats["mean_prob"][::-1]]),
        y=np.concatenate([
            bin_stats["actual_freq"] + 0.05,
            (bin_stats["actual_freq"] - 0.05)[::-1],
        ]),
        fill="toself",
        fillcolor="rgba(0,212,170,0.08)",
        line=dict(color="rgba(0,0,0,0)"),
        name="±5% band",
        showlegend=True,
    ))

    fig.update_layout(
        height=500,
        xaxis_title="Predicted Probability",
        yaxis_title="Actual Win Frequency",
        xaxis=dict(range=[0, 1], tickformat=".0%"),
        yaxis=dict(range=[0, 1], tickformat=".0%"),
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="white"),
        legend=dict(x=0.02, y=0.98),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Brier Score
    from sklearn.metrics import brier_score_loss
    brier = brier_score_loss(df["won"].astype(int), df["prob"])
    col1, col2 = st.columns(2)
    col1.metric("Brier Score", f"{brier:.4f}",
                help="Thấp hơn = tốt hơn. Perfect = 0.0, Random = 0.25")
    col2.metric("Total Calibration Bets", f"{len(df):,}")

    # Histogram phân phối prob
    st.subheader("📊 Probability Distribution")
    fig2 = go.Figure()
    for won, color, name in [(True, "#00d4aa", "Won"), (False, "#ff4b6e", "Lost")]:
        subset = df[df["won"] == won]["prob"]
        fig2.add_trace(go.Histogram(
            x=subset, name=name, marker_color=color,
            opacity=0.7, nbinsx=20,
        ))
    fig2.update_layout(
        barmode="overlay", height=300,
        xaxis_title="Predicted Probability",
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="white"),
    )
    st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════
# TAB 5: DATA OVERVIEW
# ══════════════════════════════════════════════════════

def render_data_tab(df_matches: pd.DataFrame):
    st.header("🗄️ Database Overview")

    if df_matches.empty:
        st.warning("Chưa có dữ liệu. Chạy scraper.py trước.")
        st.code("python src/data/scraper.py")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Matches", f"{len(df_matches):,}")
    c2.metric("Leagues", df_matches["league"].nunique() if "league" in df_matches.columns else "—")
    c3.metric("Seasons",  df_matches["season"].nunique() if "season" in df_matches.columns else "—")
    c4.metric("Teams",    df_matches["home_team"].nunique() if "home_team" in df_matches.columns else "—")

    if "league" in df_matches.columns:
        fig = px.bar(
            df_matches.groupby(["league","season"]).size().reset_index(name="count"),
            x="season", y="count", color="league",
            title="Matches per Season",
            color_discrete_sequence=["#00d4aa", "#ffa94d"],
        )
        fig.update_layout(
            plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
            font=dict(color="white"), height=350,
        )
        st.plotly_chart(fig, use_container_width=True)

    if "result" in df_matches.columns:
        dist = df_matches["result"].value_counts().reset_index()
        dist.columns = ["Result", "Count"]
        dist["Result"] = dist["Result"].map({"H": "Home Win", "D": "Draw", "A": "Away Win"})
        fig2 = px.pie(dist, values="Count", names="Result",
                      color_discrete_sequence=["#00d4aa", "#ffa94d", "#ff4b6e"],
                      title="Result Distribution")
        fig2.update_layout(
            paper_bgcolor="#0e1117", font=dict(color="white"),
        )
        st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════

def main():
    # Load data
    df_bt      = load_backtest_results()
    df_signals = load_live_signals()
    df_matches = load_db_matches()

    # Sidebar
    filters = render_sidebar(df_bt)

    # Header
    st.title("⚽ Football Betting Analytics Dashboard")
    st.caption("Walk-Forward Backtest | Live EV+ Signals | Calibration | P&L Tracking")

    if df_bt.empty and df_signals.empty and df_matches.empty:
        st.info("""
        **🚀 Pipeline chưa được chạy.** Thực hiện theo thứ tự:
        ```
        # 1. Init DB
        python src/data/database.py

        # 2. Scrape historical data
        python src/data/scraper.py

        # 3. Train models
        python src/models/trainer.py

        # 4. Backtest
        python src/betting/backtest.py

        # 5. Generate live signals
        python src/signals/live_signals.py

        # 6. Chạy dashboard
        streamlit run src/dashboard/app.py
        ```
        """)
        return

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Live Signals",
        "💰 Bankroll Curve",
        "📊 P&L Detail",
        "🎛️ Calibration",
        "🗄️ Data",
    ])

    with tab1:
        render_signals_tab(df_signals)
    with tab2:
        render_bankroll_tab(df_bt, filters)
    with tab3:
        render_pnl_tab(df_bt, filters)
    with tab4:
        render_calibration_tab(df_bt)
    with tab5:
        render_data_tab(df_matches)


if __name__ == "__main__":
    main()
