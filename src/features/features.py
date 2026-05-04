"""
features.py — Feature engineering
Tính rolling stats, Elo, xG proxy, H2H cho mỗi trận
"""
import pandas as pd
import numpy as np
from loguru import logger
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import ROLLING_WINDOWS
from data.database import get_session, Match, MatchFeature


# ── Elo rating ─────────────────────────────────────────
ELO_K      = 32
ELO_START  = 1500

def expected_elo(ra, rb):
    return 1 / (1 + 10 ** ((rb - ra) / 400))

def update_elo(ra, rb, score_a):
    """score_a: 1=thắng, 0.5=hòa, 0=thua"""
    exp = expected_elo(ra, rb)
    new_ra = ra + ELO_K * (score_a - exp)
    new_rb = rb + ELO_K * ((1 - score_a) - (1 - exp))
    return round(new_ra, 2), round(new_rb, 2)


def compute_elo_ratings(df: pd.DataFrame) -> pd.DataFrame:
    """Tính Elo theo thứ tự thời gian, trả về df với 2 cột mới"""
    df = df.sort_values("date").copy()
    elo = {}
    home_elo_list, away_elo_list = [], []

    for _, row in df.iterrows():
        h, a = row["home_team"], row["away_team"]
        rh = elo.get(h, ELO_START)
        ra = elo.get(a, ELO_START)
        home_elo_list.append(rh)
        away_elo_list.append(ra)
        # Cập nhật sau trận (nếu có kết quả)
        if pd.notna(row.get("result")):
            score = 1.0 if row["result"] == "H" else (0.5 if row["result"] == "D" else 0.0)
            elo[h], elo[a] = update_elo(rh, ra, score)

    df["home_elo"] = home_elo_list
    df["away_elo"] = away_elo_list
    df["elo_diff"] = df["home_elo"] - df["away_elo"]
    return df

def compute_rolling_stats(df: pd.DataFrame, windows=None) -> pd.DataFrame:
    """
    Tính rolling stats PER-MATCH cho từng đội theo thứ tự thời gian.
    Trả về df gốc với các cột rolling được gắn vào từng trận (không leakage).
    Dùng shift(1) để đảm bảo features chỉ dùng dữ liệu TRƯỚC trận đó.
    """
    windows = windows or ROLLING_WINDOWS
    df = df.sort_values("date").reset_index(drop=True).copy()

    # Stack home + away rows, giữ match_idx để join lại
    home_rows = df[["date", "home_team", "home_goals", "away_goals",
                    "home_shots_target", "away_shots_target", "result"]].copy()
    home_rows["team"]      = home_rows["home_team"]
    home_rows["scored"]    = home_rows["home_goals"]
    home_rows["conceded"]  = home_rows["away_goals"]
    home_rows["sot_for"]   = home_rows["home_shots_target"]
    home_rows["sot_against"] = home_rows["away_shots_target"]
    home_rows["is_home"]   = True
    home_rows["match_idx"] = home_rows.index

    away_rows = df[["date", "away_team", "away_goals", "home_goals",
                    "away_shots_target", "home_shots_target", "result"]].copy()
    away_rows["team"]      = away_rows["away_team"]
    away_rows["scored"]    = away_rows["away_goals"]
    away_rows["conceded"]  = away_rows["home_goals"]
    away_rows["sot_for"]   = away_rows["away_shots_target"]
    away_rows["sot_against"] = away_rows["home_shots_target"]
    away_rows["is_home"]   = False
    away_rows["match_idx"] = away_rows.index

    keep_cols = ["date", "team", "scored", "conceded", "sot_for",
                 "sot_against", "result", "is_home", "match_idx"]
    teams = pd.concat([home_rows[keep_cols], away_rows[keep_cols]]).sort_values(
        ["date", "match_idx"]
    ).reset_index(drop=True)

    # Points: 3 nếu thắng, 1 hòa, 0 thua — dùng is_home thay vì lookup
    def calc_points(r):
        if r["result"] == "D":
            return 1
        win_result = "H" if r["is_home"] else "A"
        return 3 if r["result"] == win_result else 0

    teams["points"]   = teams.apply(calc_points, axis=1)
    teams["xg_proxy"] = teams["sot_for"] / (teams["sot_for"] + teams["sot_against"] + 1e-5)

    # Tính rolling per-team, shift(1) để tránh leakage
    rolling_records = []  # list of (match_idx, is_home, feat_dict)
    for team, grp in teams.groupby("team"):
        grp = grp.sort_values("date").copy()
        for w in windows:
            grp[f"_form_{w}"]     = grp["points"].shift(1).rolling(w, min_periods=1).mean()
            grp[f"_xg_{w}"]       = grp["xg_proxy"].shift(1).rolling(w, min_periods=1).mean()
            grp[f"_scored_{w}"]   = grp["scored"].shift(1).rolling(w, min_periods=1).mean()
            grp[f"_conceded_{w}"] = grp["conceded"].shift(1).rolling(w, min_periods=1).mean()

        for _, row in grp.iterrows():
            feat = {"match_idx": row["match_idx"], "is_home": row["is_home"]}
            for w in windows:
                feat[f"form_{w}"]     = row[f"_form_{w}"]
                feat[f"xg_{w}"]       = row[f"_xg_{w}"]
                feat[f"scored_{w}"]   = row[f"_scored_{w}"]
                feat[f"conceded_{w}"] = row[f"_conceded_{w}"]
            rolling_records.append(feat)

    roll_df = pd.DataFrame(rolling_records)
    home_roll = roll_df[roll_df["is_home"]].drop(columns="is_home").set_index("match_idx")
    away_roll = roll_df[~roll_df["is_home"]].drop(columns="is_home").set_index("match_idx")

    # Đặt prefix home_ / away_ rồi join vào df gốc
    home_roll.columns = ["home_" + c for c in home_roll.columns]
    away_roll.columns = ["away_" + c for c in away_roll.columns]

    df = df.join(home_roll).join(away_roll)
    return df


def compute_h2h(df: pd.DataFrame) -> pd.DataFrame:
    """Head-to-head winrate cho mỗi cặp đội (5 trận gần nhất trước trận đó)"""
    df = df.sort_values("date").reset_index(drop=True).copy()
    h2h_home, h2h_draw, h2h_away = [], [], []

    for pos in range(len(df)):
        row = df.iloc[pos]
        h, a = row["home_team"], row["away_team"]
        row_date = row["date"]
        # Dùng date để lọc, không dùng index (tránh bug sau reset_index)
        past = df[
            (df["date"] < row_date) &
            (((df["home_team"] == h) & (df["away_team"] == a)) |
             ((df["home_team"] == a) & (df["away_team"] == h)))
        ].tail(5)
        if len(past) == 0:
            h2h_home.append(0.4); h2h_draw.append(0.3); h2h_away.append(0.3)
            continue
        wins_h  = ((past["home_team"] == h) & (past["result"] == "H")).sum() + \
                  ((past["away_team"] == h) & (past["result"] == "A")).sum()
        draws   = (past["result"] == "D").sum()
        wins_a  = len(past) - wins_h - draws
        n = len(past)
        h2h_home.append(wins_h / n); h2h_draw.append(draws / n); h2h_away.append(wins_a / n)

    df["h2h_home"] = h2h_home
    df["h2h_draw"] = h2h_draw
    df["h2h_away"] = h2h_away
    return df


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Master function: nhận df lịch sử, trả về X (feature matrix) và y (labels).
    Tất cả rolling stats được tính per-match (không leakage).
    """
    df = compute_elo_ratings(df)
    df = compute_h2h(df)
    df = compute_rolling_stats(df)   # join trực tiếp vào df

    feat_cols = (
        ["elo_diff", "home_elo", "away_elo", "h2h_home", "h2h_draw", "h2h_away"]
        + [f"home_form_{w}" for w in ROLLING_WINDOWS]
        + [f"home_xg_{w}"   for w in ROLLING_WINDOWS]
        + [f"home_scored_{w}" for w in ROLLING_WINDOWS]
        + [f"home_conceded_{w}" for w in ROLLING_WINDOWS]
        + [f"away_form_{w}" for w in ROLLING_WINDOWS]
        + [f"away_xg_{w}"   for w in ROLLING_WINDOWS]
        + [f"away_scored_{w}" for w in ROLLING_WINDOWS]
        + [f"away_conceded_{w}" for w in ROLLING_WINDOWS]
    )

    # Chỉ lấy cột tồn tại (phòng trường hợp shots_target bị null)
    available = [c for c in feat_cols if c in df.columns]
    X = df[available].copy()
    y = df["result"].map({"H": 0, "D": 1, "A": 2})
    return X, y, df
