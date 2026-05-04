"""
database.py — SQLAlchemy schema (v2)
Thêm: OddsSnapshot, LiveBet, Pinnacle odds columns
"""
from sqlalchemy import (
    create_engine, Column, Integer, Float, String,
    DateTime, Boolean, ForeignKey, Text, Index
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from sqlalchemy.sql import func
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DB_PATH

Base = declarative_base()


class Match(Base):
    """Kết quả lịch sử các trận đấu"""
    __tablename__ = "matches"

    id          = Column(Integer, primary_key=True)
    league      = Column(String(50), nullable=False)
    season      = Column(String(10), nullable=False)
    date        = Column(DateTime, nullable=False)
    home_team   = Column(String(100), nullable=False)
    away_team   = Column(String(100), nullable=False)
    home_goals  = Column(Integer)
    away_goals  = Column(Integer)
    result      = Column(String(1))
    # Bet365 opening odds
    odds_h      = Column(Float)
    odds_d      = Column(Float)
    odds_a      = Column(Float)
    # Bet365 closing odds
    close_h     = Column(Float)
    close_d     = Column(Float)
    close_a     = Column(Float)
    # Pinnacle opening odds (sharper market, lower vig ~2-3%)
    pin_h       = Column(Float)
    pin_d       = Column(Float)
    pin_a       = Column(Float)
    # Pinnacle closing odds
    pin_close_h = Column(Float)
    pin_close_d = Column(Float)
    pin_close_a = Column(Float)
    # Match stats
    home_shots        = Column(Integer)
    away_shots        = Column(Integer)
    home_shots_target = Column(Integer)
    away_shots_target = Column(Integer)
    home_corners      = Column(Integer)
    away_corners      = Column(Integer)
    created_at  = Column(DateTime, server_default=func.now())

    features    = relationship("MatchFeature", back_populates="match", uselist=False)
    predictions = relationship("Prediction", back_populates="match")

    __table_args__ = (
        Index("ix_matches_league_date", "league", "date"),
        Index("ix_matches_teams", "home_team", "away_team"),
    )


class OddsSnapshot(Base):
    """
    Snapshot odds theo thời gian — dùng để detect intraday movement.
    Poll mỗi 60 phút từ The Odds API cho upcoming matches.
    is_baseline=True là snapshot đầu tiên khi match được publish.
    """
    __tablename__ = "odds_snapshots"

    id          = Column(Integer, primary_key=True)
    match_key   = Column(String(300), nullable=False)  # "Arsenal__Chelsea__2025-05-10"
    league      = Column(String(50))
    home_team   = Column(String(100))
    away_team   = Column(String(100))
    match_date  = Column(DateTime)
    snapshot_at = Column(DateTime, server_default=func.now())
    bookmaker   = Column(String(50), default="median")
    odds_h      = Column(Float)
    odds_d      = Column(Float)
    odds_a      = Column(Float)
    is_baseline = Column(Boolean, default=False)  # True = snapshot đầu tiên

    __table_args__ = (
        Index("ix_snapshots_match_key", "match_key"),
        Index("ix_snapshots_league_date", "league", "match_date"),
    )


class LiveBet(Base):
    """
    Log các bet thực tế (hoặc paper) từ live signals.
    Dùng để track P&L thực, so sánh với backtest.
    """
    __tablename__ = "live_bets"

    id              = Column(Integer, primary_key=True)
    match_key       = Column(String(300))
    league          = Column(String(50))
    season          = Column(String(10))
    home_team       = Column(String(100))
    away_team       = Column(String(100))
    match_date      = Column(DateTime)
    signal_side     = Column(String(1))     # H / D / A
    movement        = Column(Float)         # detected movement size
    baseline_odds   = Column(Float)         # odds tại snapshot baseline
    bet_odds        = Column(Float)         # odds tại thời điểm đặt
    stake           = Column(Float)
    bankroll_before = Column(Float)
    # Điền sau khi trận kết thúc
    result          = Column(String(1))     # H / D / A
    won             = Column(Boolean)
    profit          = Column(Float)
    bankroll_after  = Column(Float)
    # CLV: nếu bet_odds > close_odds => đặt tốt hơn closing (sharp-like)
    close_odds      = Column(Float)
    clv             = Column(Float)         # close_prob - 1/bet_odds (dương = good)
    signal_fired_at = Column(DateTime)
    bet_placed_at   = Column(DateTime)
    settled         = Column(Boolean, default=False)
    is_paper        = Column(Boolean, default=True)  # True = paper, False = real
    notes           = Column(Text)

    __table_args__ = (
        Index("ix_live_bets_match", "match_key"),
        Index("ix_live_bets_settled", "settled"),
    )


class MatchFeature(Base):
    __tablename__ = "match_features"
    id        = Column(Integer, primary_key=True)
    match_id  = Column(Integer, ForeignKey("matches.id"), unique=True)
    match     = relationship("Match", back_populates="features")
    home_xg_3   = Column(Float)
    home_xg_5   = Column(Float)
    home_xg_10  = Column(Float)
    away_xg_3   = Column(Float)
    away_xg_5   = Column(Float)
    away_xg_10  = Column(Float)
    home_form_3   = Column(Float)
    home_form_5   = Column(Float)
    home_form_10  = Column(Float)
    away_form_3   = Column(Float)
    away_form_5   = Column(Float)
    away_form_10  = Column(Float)
    home_scored_5    = Column(Float)
    home_conceded_5  = Column(Float)
    away_scored_5    = Column(Float)
    away_conceded_5  = Column(Float)
    home_elo  = Column(Float)
    away_elo  = Column(Float)
    elo_diff  = Column(Float)
    h2h_home_wins   = Column(Float)
    h2h_draws       = Column(Float)
    h2h_away_wins   = Column(Float)
    home_advantage  = Column(Float)
    home_rest_days  = Column(Integer)
    away_rest_days  = Column(Integer)


class Prediction(Base):
    __tablename__ = "predictions"
    id        = Column(Integer, primary_key=True)
    match_id  = Column(Integer, ForeignKey("matches.id"))
    match     = relationship("Match", back_populates="predictions")
    model_name   = Column(String(50))
    prob_home    = Column(Float)
    prob_draw    = Column(Float)
    prob_away    = Column(Float)
    cal_prob_home = Column(Float)
    cal_prob_draw = Column(Float)
    cal_prob_away = Column(Float)
    ev_home   = Column(Float)
    ev_draw   = Column(Float)
    ev_away   = Column(Float)
    kelly_home = Column(Float)
    kelly_draw = Column(Float)
    kelly_away = Column(Float)
    signal_home  = Column(Boolean, default=False)
    signal_draw  = Column(Boolean, default=False)
    signal_away  = Column(Boolean, default=False)
    created_at   = Column(DateTime, server_default=func.now())


class BettingRecord(Base):
    __tablename__ = "betting_records"
    id           = Column(Integer, primary_key=True)
    match_id     = Column(Integer, ForeignKey("matches.id"))
    bet_side     = Column(String(5))
    odds         = Column(Float)
    stake        = Column(Float)
    ev           = Column(Float)
    kelly_frac   = Column(Float)
    bankroll_before = Column(Float)
    bankroll_after  = Column(Float)
    profit       = Column(Float)
    won          = Column(Boolean)
    settled      = Column(Boolean, default=False)
    created_at   = Column(DateTime, server_default=func.now())


class LiveOdds(Base):
    __tablename__ = "live_odds"
    id          = Column(Integer, primary_key=True)
    league      = Column(String(50))
    match_key   = Column(String(200))
    bookmaker   = Column(String(50))
    home_team   = Column(String(100))
    away_team   = Column(String(100))
    commence    = Column(DateTime)
    odds_h      = Column(Float)
    odds_d      = Column(Float)
    odds_a      = Column(Float)
    fetched_at  = Column(DateTime, server_default=func.now())
    __table_args__ = (
        Index("ix_live_odds_match", "match_key", "bookmaker"),
    )


def get_engine():
    return create_engine(f"sqlite:///{DB_PATH}", echo=False)


def get_session():
    engine = get_engine()
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return Session()


def init_db():
    engine = get_engine()
    Base.metadata.create_all(engine)
    print(f"[DB] Database initialized at {DB_PATH}")


if __name__ == "__main__":
    init_db()
