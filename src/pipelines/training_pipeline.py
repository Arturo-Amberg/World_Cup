"""
training_pipeline.py — Full stacking ensemble trained on 49k historical
international matches (martj42/international_results).

Architecture
============
Level-0  base models (12):
  LightGBM, XGBoost, CatBoost, RandomForest, ExtraTrees,
  GradientBoosting, LogisticRegression, MLP, KNN, LinearSVC,
  PoissonModel (formula-based), RollingMeanModel (naive baseline)

Level-1  meta-learners (4, trained on out-of-fold base predictions):
  LogisticRegression, LightGBM, MLP, RidgeLogistic

Final output: mean of all meta-learner probability vectors.

Features (~88 per row):
  Rolling 5 / 10 / 15 / 30-match stats per team:
    GF_avg, GA_avg, win_rate, draw_rate, pts_avg, GD_avg, clean_sheet_rate
  EWMA (α=0.3): GF_ewma, GA_ewma
  n_prior (number of prior matches)
  H2H win rate, draw rate, games count
  ELO_a, ELO_b, ELO_diff
  is_neutral, tournament_type
  Difference features (gf_diff, ga_diff, pts_diff, win_diff × 4 windows + ewma)

Sample weights:
  recency  = exp((year - 1994) / 20)        # recent games count more
  t_weight = 3.0 for WC, 2.5 for Euro/Copa, 0.7 for friendlies, etc.
  final_w  = recency × t_weight  (normalised to mean=1)
"""

import json
import math
import os
import pickle
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson as sp_poisson
import argparse
import optuna
import math
import csv

# ── sklearn ────────────────────────────────────────────────────────────────────
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (ExtraTreesClassifier,
                               RandomForestClassifier,
                               HistGradientBoostingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import log_loss, accuracy_score

# ── Third-party ML ─────────────────────────────────────────────────────────────
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent.parent.parent
DATA_DIR   = BASE_DIR / "data"
MODELS_DIR = DATA_DIR / "models"
CSV_PATH   = DATA_DIR / "intl_results.csv"
ELOS_PATH  = DATA_DIR / "elos.json"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────────────
WC_AVG_GOALS       = 1.30
BASELINE_ELO       = 1750
MIN_YEAR           = 1994
MIN_PRIOR_MATCHES  = 10
N_FOLDS            = 5
MIN_PRIOR_MATCHES = 10
ROLLING_WINDOWS   = [5, 15, 30]
EWMA_ALPHA        = 0.153

TOURNAMENT_WEIGHTS = {
    "FIFA World Cup":                      3.0,
    "UEFA Euro":                           2.5,
    "Copa América":                        2.5,
    "African Cup of Nations":              2.0,
    "AFC Asian Cup":                       2.0,
    "Gold Cup":                            1.8,
    "UEFA Nations League":                 1.5,
    "CONCACAF Nations League":             1.5,
    "FIFA World Cup qualification":        1.4,
    "UEFA Euro qualification":             1.3,
    "African Cup of Nations qualification":1.2,
    "AFC Asian Cup qualification":         1.2,
    "Friendly":                            0.7,
}

TOURNAMENT_TYPE_MAP = {
    "FIFA World Cup":                      5,
    "UEFA Euro":                           4,
    "Copa América":                        4,
    "African Cup of Nations":              4,
    "AFC Asian Cup":                       4,
    "Gold Cup":                            4,
    "UEFA Nations League":                 3,
    "CONCACAF Nations League":             3,
    "FIFA World Cup qualification":        2,
    "UEFA Euro qualification":             2,
    "African Cup of Nations qualification":2,
    "AFC Asian Cup qualification":         2,
    "Friendly":                            0,
}

_DATASET_TO_ELO = {
    "United States": "USA",
    "South Korea":   "South Korea",
    "Ivory Coast":   "Ivory Coast",
    "DR Congo":      "DR Congo",
    "Bosnia and Herzegovina": "Bosnia & Herzegovina",
    "Curaçao":       "Curaçao",
    "Cape Verde":    "Cape Verde",
    "Turkey":        "Turkey",
    "Iran":          "Iran",
    "North Macedonia": "North Macedonia",
}


# ══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_data() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH, parse_dates=["date"])
    df = df[df["home_score"].notna() & df["away_score"].notna()].copy()
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    df = df.sort_values("date").reset_index(drop=True)
    return df

def load_elos() -> dict:
    with open(ELOS_PATH) as f:
        return json.load(f).get("data", {})

def team_elo(name: str, elos: dict) -> float:
    canonical = _DATASET_TO_ELO.get(name, name)
    if canonical in elos:
        return float(elos[canonical])
    low = canonical.lower()
    for k, v in elos.items():
        if low in k.lower() or k.lower() in low:
            return float(v)
    return BASELINE_ELO


# ══════════════════════════════════════════════════════════════════════════════
#  ROLLING FEATURE ENGINE
# ══════════════════════════════════════════════════════════════════════════════

_DEFAULT_STATS = {w: {
    "gf_avg": WC_AVG_GOALS, "ga_avg": WC_AVG_GOALS,
    "win_rate": 0.33, "draw_rate": 0.25,
    "pts_avg": 1.0, "gd_avg": 0.0, "cs_rate": 0.25,
} for w in ROLLING_WINDOWS}

def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1-a))

def load_country_coords(csv_path):
    coords = {}
    if not csv_path.exists():
        return coords
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get('Latitude') or not row.get('Longitude'): continue
            country = row['Country'].strip()
            lat, lon = float(row['Latitude']), float(row['Longitude'])
            elev = float(row.get('elevation', 0.0))
            coords[country] = (lat, lon, elev)
            
            # Map UK nations
            if country == "United Kingdom":
                coords["England"] = (lat, lon, elev)
                coords["Scotland"] = (lat+3, lon, elev)
                coords["Wales"] = (lat, lon-3, elev)
                coords["Northern Ireland"] = (lat+1, lon-4, elev)
    # Manual fixes for common football names
    coords["USA"] = (37.0902, -95.7129, 413.0)
    coords["United States"] = (37.0902, -95.7129, 413.0)
    coords["South Korea"] = (35.9078, 127.7669, 89.0)
    coords["Republic of Ireland"] = (53.1424, -7.6921, 130.0)
    coords["Ivory Coast"] = (7.54, -5.5471, 279.0)
    return coords


def _rolling_stats(vals_gf: np.ndarray, vals_ga: np.ndarray,
                   vals_pts: np.ndarray, window: int) -> dict:
    n = len(vals_gf)
    if n == 0:
        return dict(_DEFAULT_STATS[window])
    s = max(0, n - window)
    gf = vals_gf[s:]
    ga = vals_ga[s:]
    pts = vals_pts[s:]
    return {
        "gf_avg":   float(gf.mean()),
        "ga_avg":   float(ga.mean()),
        "win_rate": float((pts == 3).mean()),
        "draw_rate":float((pts == 1).mean()),
        "pts_avg":  float(pts.mean()),
        "gd_avg":   float((gf - ga).mean()),
        "cs_rate":  float((ga == 0).mean()),
    }


def _ewma(vals: np.ndarray, alpha: float) -> float:
    if len(vals) == 0:
        return WC_AVG_GOALS
    w = (1 - alpha) ** np.arange(len(vals) - 1, -1, -1)
    return float((vals * w).sum() / w.sum())


class TacticalStatsComputer:
    def __init__(self, df_tactical: pd.DataFrame):
        self._df = df_tactical
        self._cache = {}
        
    def _build(self, team: str):
        if team in self._cache:
            return self._cache[team]
        
        home = self._df[self._df["home_team"] == team][["date", "home_shots", "home_corners", "home_possession", "away_shots", "away_corners"]].copy()
        home.rename(columns={
            "home_shots": "shots_for",
            "home_corners": "corners_for",
            "home_possession": "poss",
            "away_shots": "shots_against",
            "away_corners": "corners_against"
        }, inplace=True)
        away = self._df[self._df["away_team"] == team][["date", "away_shots", "away_corners", "away_possession", "home_shots", "home_corners"]].copy()
        away.rename(columns={
            "away_shots": "shots_for",
            "away_corners": "corners_for",
            "away_possession": "poss",
            "home_shots": "shots_against",
            "home_corners": "corners_against"
        }, inplace=True)
        
        hist = pd.concat([home, away]).sort_values("date").reset_index(drop=True)
        dates_ns = pd.to_datetime(hist["date"]).values.astype("int64")
        
        self._cache[team] = (dates_ns, hist)
        return self._cache[team]
        
    def get_features(self, team: str, as_of_date, suffix: str) -> dict:
        as_of_ns = np.int64(pd.Timestamp(as_of_date).value)
        dates_ns, hist = self._build(team)
        idx = int(np.searchsorted(dates_ns, as_of_ns, side="left"))
        
        prior = hist.iloc[:idx]
        
        # Default neutral stats if no tactical data
        res = {
            f"tactical_shots_for{suffix}": 12.0,
            f"tactical_shots_against{suffix}": 12.0,
            f"tactical_corners_for{suffix}": 4.5,
            f"tactical_corners_against{suffix}": 4.5,
            f"tactical_poss{suffix}": 50.0,
        }
        
        if len(prior) > 0:
            recent = prior.tail(10) # last 10 tactical matches
            res[f"tactical_shots_for{suffix}"] = recent["shots_for"].mean()
            res[f"tactical_shots_against{suffix}"] = recent["shots_against"].mean()
            res[f"tactical_corners_for{suffix}"] = recent["corners_for"].mean()
            res[f"tactical_corners_against{suffix}"] = recent["corners_against"].mean()
            res[f"tactical_poss{suffix}"] = recent["poss"].mean()
            
        return res

class RollingStatsComputer:
    """Caches per-team sorted histories for fast rolling feature extraction."""

    def __init__(self, df: pd.DataFrame):
        self._df = df
        self._cache: dict[str, tuple] = {}  # team -> (dates_ns, gf, ga, pts)

    def _build(self, team: str):
        if team in self._cache:
            return self._cache[team]

        home = self._df[self._df["home_team"] == team][["date", "home_score", "away_score"]].copy()
        home.columns = ["date", "gf", "ga"]
        away = self._df[self._df["away_team"] == team][["date", "home_score", "away_score"]].copy()
        away.columns = ["date", "ga", "gf"]

        hist = pd.concat([home, away]).sort_values("date").reset_index(drop=True)
        gf  = hist["gf"].values.astype(float)
        ga  = hist["ga"].values.astype(float)
        pts = np.where(gf > ga, 3.0, np.where(gf == ga, 1.0, 0.0))
        dates_ns = hist["date"].values.astype("int64")

        self._cache[team] = (dates_ns, gf, ga, pts)
        return self._cache[team]

    def _prior_slice(self, team: str, as_of_ns: int) -> tuple:
        """Return (gf, ga, pts) arrays for all matches strictly before as_of_ns."""
        dates_ns, gf, ga, pts = self._build(team)
        idx = int(np.searchsorted(dates_ns, as_of_ns, side="left"))
        return gf[:idx], ga[:idx], pts[:idx]

    def get_features(self, team: str, as_of_date, suffix: str) -> dict:
        as_of_ns = np.int64(pd.Timestamp(as_of_date).value)
        gf, ga, pts = self._prior_slice(team, as_of_ns)

        feat: dict = {}
        for w in ROLLING_WINDOWS:
            stats = _rolling_stats(gf, ga, pts, w)
            for k, v in stats.items():
                feat[f"{k}_{w}{suffix}"] = v

        feat[f"gf_ewma{suffix}"] = _ewma(gf[-30:], EWMA_ALPHA)
        feat[f"ga_ewma{suffix}"] = _ewma(ga[-30:], EWMA_ALPHA)
        feat[f"n_prior{suffix}"] = len(gf)
        
        # Streaks
        win_streak = 0
        unbeaten_streak = 0
        for p in reversed(pts):
            if p == 3:
                win_streak += 1
                unbeaten_streak += 1
            elif p == 1:
                win_streak = 0
                unbeaten_streak += 1
            else:
                break
                
        feat[f"win_streak{suffix}"] = float(win_streak)
        feat[f"unbeaten_streak{suffix}"] = float(unbeaten_streak)
        return feat

    def get_current_features(self, team: str, suffix: str) -> dict:
        """All data (for inference, not training)."""
        dates_ns, gf, ga, pts = self._build(team)
        feat: dict = {}
        for w in ROLLING_WINDOWS:
            stats = _rolling_stats(gf, ga, pts, w)
            for k, v in stats.items():
                feat[f"{k}_{w}{suffix}"] = v
        feat[f"gf_ewma{suffix}"] = _ewma(gf[-30:], EWMA_ALPHA)
        feat[f"ga_ewma{suffix}"] = _ewma(ga[-30:], EWMA_ALPHA)
        feat[f"n_prior{suffix}"] = len(gf)
        
        win_streak = 0
        unbeaten_streak = 0
        for p in reversed(pts):
            if p == 3: win_streak += 1; unbeaten_streak += 1
            elif p == 1: win_streak = 0; unbeaten_streak += 1
            else: break
        feat[f"win_streak{suffix}"] = float(win_streak)
        feat[f"unbeaten_streak{suffix}"] = float(unbeaten_streak)
        return feat

    def n_prior(self, team: str, as_of_date) -> int:
        as_of_ns = np.int64(pd.Timestamp(as_of_date).value)
        gf, _, _ = self._prior_slice(team, as_of_ns)
        return len(gf)

    def rest_days(self, team: str, as_of_date) -> int:
        as_of_ns = np.int64(pd.Timestamp(as_of_date).value)
        dates_ns, _, _, _ = self._build(team)
        idx = int(np.searchsorted(dates_ns, as_of_ns, side="left"))
        if idx == 0:
            return 30 # Default max if no prior games
        prev_ns = dates_ns[idx - 1]
        days = (as_of_ns - prev_ns) / (1e9 * 86400)
        return min(int(days), 30)


class H2HComputer:
    """Caches per-pair H2H histories."""

    def __init__(self, df: pd.DataFrame):
        self._df = df
        self._cache: dict = {}

    def _pair_df(self, a: str, b: str) -> pd.DataFrame:
        key = frozenset([a, b])
        if key not in self._cache:
            mask = (
                ((self._df["home_team"] == a) & (self._df["away_team"] == b)) |
                ((self._df["home_team"] == b) & (self._df["away_team"] == a))
            )
            self._cache[key] = self._df[mask].copy()
        return self._cache[key]

    def get_h2h(self, a: str, b: str, as_of_date=None, n: int = 20) -> dict:
        h2h = self._pair_df(a, b)
        if as_of_date is not None:
            h2h = h2h[h2h["date"] < as_of_date]
        h2h = h2h.tail(n)
        ng = len(h2h)
        if ng == 0:
            return {"h2h_win_rate_a": 0.33, "h2h_draw_rate": 0.25, "h2h_gd_avg_a": 0.0, "h2h_games": 0}

        wins_a = draws = gd_a = 0
        for _, row in h2h.iterrows():
            hs = row["home_score"] if row["home_team"] == a else row["away_score"]
            as_ = row["away_score"] if row["home_team"] == a else row["home_score"]
            gd_a += (hs - as_)
            if hs > as_: wins_a += 1
            elif hs == as_: draws += 1

        return {
            "h2h_win_rate_a": wins_a / ng,
            "h2h_draw_rate":  draws  / ng,
            "h2h_gd_avg_a":   gd_a / ng,
            "h2h_games":      min(ng, 20),
        }


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL-BASED BASE MODELS  (sklearn-compatible, no training needed)
# ══════════════════════════════════════════════════════════════════════════════

def _poisson_3way(lam_a: float, lam_b: float, max_g: int = 9) -> list:
    pa = pd = pb = 0.0
    rho = -0.12  # Dixon-Coles
    for i in range(max_g + 1):
        pi = sp_poisson.pmf(i, lam_a)
        for j in range(max_g + 1):
            p = pi * sp_poisson.pmf(j, lam_b)
            if   i == 0 and j == 0: tau = 1 - lam_a * lam_b * rho
            elif i == 1 and j == 0: tau = 1 + lam_b * rho
            elif i == 0 and j == 1: tau = 1 + lam_a * rho
            elif i == 1 and j == 1: tau = 1 - rho
            else:                   tau = 1.0
            p *= max(0.01, tau)
            if   i > j: pa += p
            elif i == j: pd += p
            else:        pb += p
    total = pa + pd + pb or 1.0
    return [pa / total, pd / total, pb / total]


class PoissonBaseModel:
    """Formula-based Poisson (Dixon-Coles). No fitting needed."""

    def fit(self, X, y=None, sample_weight=None):
        return self

    def predict_proba(self, X):
        out = []
        gf_a = X["gf_avg_15_a"].values
        ga_a = X["ga_avg_15_a"].values
        gf_b = X["gf_avg_15_b"].values
        ga_b = X["ga_avg_15_b"].values
        mu = WC_AVG_GOALS
        for i in range(len(X)):
            la = max(0.3, gf_a[i]) * max(0.3, ga_b[i]) / mu
            lb = max(0.3, gf_b[i]) * max(0.3, ga_a[i]) / mu
            out.append(_poisson_3way(la, lb))
        return np.array(out)

    def get_params(self, deep=True): return {}
    def set_params(self, **p):       return self


class RollingMeanBaseModel:
    """Naive baseline: converts rolling win/draw rates directly to probabilities."""

    def fit(self, X, y=None, sample_weight=None):
        return self

    def predict_proba(self, X):
        out = []
        wr_a = X["win_rate_15_a"].values
        dr_a = X["draw_rate_15_a"].values
        wr_b = X["win_rate_15_b"].values
        dr_b = X["draw_rate_15_b"].values
        for i in range(len(X)):
            draw = (dr_a[i] + dr_b[i]) / 2
            total = wr_a[i] + wr_b[i] + draw
            if total <= 0:
                out.append([0.33, 0.34, 0.33])
            else:
                out.append([wr_a[i] / total, draw / total, wr_b[i] / total])
        return np.array(out)

    def get_params(self, deep=True): return {}
    def set_params(self, **p):       return self


# ══════════════════════════════════════════════════════════════════════════════
#  DATASET CONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════════

def _tournament_type(t: str) -> int:
    for k, v in TOURNAMENT_TYPE_MAP.items():
        if k.lower() in t.lower():
            return v
    return 1  # qualifier-like default

def _sample_weight(year: int, tournament: str) -> float:
    recency = math.exp((year - MIN_YEAR) / 20.0)
    tw = 1.0
    for k, v in TOURNAMENT_WEIGHTS.items():
        if k.lower() in tournament.lower():
            tw = v
            break
    return recency * tw


def build_training_dataset(
    df: pd.DataFrame,
    elos: dict,
    rolling: RollingStatsComputer,
    h2h: H2HComputer,
    tactical: TacticalStatsComputer = None,
) -> tuple:
    """
    Returns X_df (DataFrame), y (ndarray), weights (ndarray).
    y: 0=team_a wins, 1=draw, 2=team_b wins  (team_a = home team in dataset)
    """
    candidates = df[df["date"].dt.year >= MIN_YEAR].copy()
    print(f"Building features for {len(candidates):,} candidate matches...")
    t0 = time.time()

    rows, targets, weights, valid_indices = [], [], [], []

    coords_map = load_country_coords(DATA_DIR / "country_coords.csv")

    for i, (original_idx, row) in enumerate(candidates.iterrows()):
        if i % 10_000 == 0 and i > 0:
            elapsed = time.time() - t0
            pct = i / len(candidates)
            eta = elapsed / pct * (1 - pct)
            print(f"  {i:,}/{len(candidates):,}  ({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

        home = str(row["home_team"])
        away = str(row["away_team"])
        match_country = str(row["country"])
        date = row["date"]

        if rolling.n_prior(home, date) < MIN_PRIOR_MATCHES:
            continue
        if rolling.n_prior(away, date) < MIN_PRIOR_MATCHES:
            continue

        feat: dict = {}
        feat.update(rolling.get_features(home, date, "_a"))
        feat.update(rolling.get_features(away, date, "_b"))
        feat.update(h2h.get_h2h(home, away, as_of_date=date))
        
        if tactical is not None:
            feat.update(tactical.get_features(home, date, "_a"))
            feat.update(tactical.get_features(away, date, "_b"))
            
            # Differentials for tactical stats
            feat["tactical_shots_diff"] = feat["tactical_shots_for_a"] - feat["tactical_shots_for_b"]
            feat["tactical_corners_diff"] = feat["tactical_corners_for_a"] - feat["tactical_corners_for_b"]
            feat["tactical_poss_diff"] = feat["tactical_poss_a"] - feat["tactical_poss_b"]

        elo_a = team_elo(home, elos)
        elo_b = team_elo(away, elos)
        feat["elo_a"]    = elo_a
        feat["elo_b"]    = elo_b
        feat["elo_diff"] = elo_a - elo_b
        
        # Proxy Squad Rating (Exponential ELO mapping mimicking FIFA Ratings)
        feat["squad_rating_a"] = 50.0 + 49.0 * max(0, min(1, (elo_a - 1000) / 1100)) ** 1.5
        feat["squad_rating_b"] = 50.0 + 49.0 * max(0, min(1, (elo_b - 1000) / 1100)) ** 1.5
        feat["squad_rating_diff"] = feat["squad_rating_a"] - feat["squad_rating_b"]

        is_neutral = bool(row.get("neutral", False))
        feat["is_neutral"]       = float(is_neutral)
        feat["is_home_a"]        = float(not is_neutral)
        feat["is_home_b"]        = 0.0 # B is always away in the dataset layout
        feat["tournament_type"]  = _tournament_type(str(row.get("tournament", "")))
        
        # Month Seasonality
        feat["match_month"] = date.month

        # Rest Days
        r_a = rolling.rest_days(home, date)
        r_b = rolling.rest_days(away, date)
        feat["rest_days_a"] = float(r_a)
        feat["rest_days_b"] = float(r_b)
        feat["rest_days_diff"] = float(r_a - r_b)

        # Distance & Altitude Shock
        dist_a = 0.0
        dist_b = 0.0
        stadium_elev = 0.0
        if match_country in coords_map:
            c_lat, c_lon, c_elev = coords_map[match_country]
            stadium_elev = c_elev
            if home in coords_map:
                dist_a = haversine(c_lat, c_lon, coords_map[home][0], coords_map[home][1])
            if away in coords_map:
                dist_b = haversine(c_lat, c_lon, coords_map[away][0], coords_map[away][1])
        feat["travel_dist_a"] = dist_a
        feat["travel_dist_b"] = dist_b
        
        home_elev = coords_map.get(home, (0, 0, 0))[2]
        away_elev = coords_map.get(away, (0, 0, 0))[2]
        feat["altitude_shock_a"] = max(0.0, stadium_elev - home_elev)
        feat["altitude_shock_b"] = max(0.0, stadium_elev - away_elev)

        for w in ROLLING_WINDOWS:
            feat[f"gf_diff_{w}"]  = feat[f"gf_avg_{w}_a"]  - feat[f"gf_avg_{w}_b"]
            feat[f"ga_diff_{w}"]  = feat[f"ga_avg_{w}_a"]  - feat[f"ga_avg_{w}_b"]
            feat[f"pts_diff_{w}"] = feat[f"pts_avg_{w}_a"] - feat[f"pts_avg_{w}_b"]
            feat[f"win_diff_{w}"] = feat[f"win_rate_{w}_a"]- feat[f"win_rate_{w}_b"]
        feat["gf_ewma_diff"] = feat["gf_ewma_a"] - feat["gf_ewma_b"]
        feat["ga_ewma_diff"] = feat["ga_ewma_a"] - feat["ga_ewma_b"]

        # Momentum (Form vs Long term)
        feat["momentum_pts_a"] = feat["pts_avg_5_a"] - feat["pts_avg_30_a"]
        feat["momentum_pts_b"] = feat["pts_avg_5_b"] - feat["pts_avg_30_b"]
        feat["momentum_gd_a"] = (feat["gf_avg_5_a"] - feat["ga_avg_5_a"]) - (feat["gf_avg_30_a"] - feat["ga_avg_30_a"])
        feat["momentum_gd_b"] = (feat["gf_avg_5_b"] - feat["ga_avg_5_b"]) - (feat["gf_avg_30_b"] - feat["ga_avg_30_b"])

        # Streak differentials
        feat["win_streak_diff"] = feat["win_streak_a"] - feat["win_streak_b"]
        feat["unbeaten_streak_diff"] = feat["unbeaten_streak_a"] - feat["unbeaten_streak_b"]

        rows.append(feat)

        hs, as_ = int(row["home_score"]), int(row["away_score"])
        targets.append(0 if hs > as_ else (1 if hs == as_ else 2))

        year  = date.year
        tourn = str(row.get("tournament", "Friendly"))
        weights.append(_sample_weight(year, tourn))
        valid_indices.append(original_idx)

    print(f"  Done: {len(rows):,} training rows in {time.time() - t0:.0f}s")

    X_df = pd.DataFrame(rows)
    X_df.index = valid_indices
    y    = np.array(targets, dtype=int)
    w    = np.array(weights, dtype=float)
    w   /= w.mean()  # normalise to mean = 1
    return X_df, y, w


# ══════════════════════════════════════════════════════════════════════════════
#  BASE MODEL REGISTRY
# ══════════════════════════════════════════════════════════════════════════════

def get_base_models(best_params: dict = None) -> dict:
    if best_params is None:
        best_params = {}

    lgbm_p = best_params.get("lgbm", {"n_estimators": 600, "max_depth": 6, "learning_rate": 0.04, "num_leaves": 31, "subsample": 0.8, "colsample_bytree": 0.8})
    xgb_p = best_params.get("xgb", {"n_estimators": 600, "max_depth": 5, "learning_rate": 0.04, "subsample": 0.8, "colsample_bytree": 0.8})
    catboost_p = best_params.get("catboost", {"iterations": 500, "depth": 6, "learning_rate": 0.04})
    rf_p = best_params.get("rf", {"n_estimators": 400, "max_depth": 12, "min_samples_leaf": 4})
    et_p = best_params.get("et", {"n_estimators": 400, "max_depth": 12, "min_samples_leaf": 4})
    gb_p = best_params.get("gb", {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.05, "subsample": 0.8})
    lr_p = best_params.get("lr", {"C": 0.5})
    mlp_p = best_params.get("mlp", {"alpha": 0.0001, "learning_rate_init": 0.001})
    svm_p = best_params.get("svm", {"C": 0.5})
    knn_p = best_params.get("knn", {"n_neighbors": 21})

    return {
        "lgbm": lgb.LGBMClassifier(**lgbm_p, class_weight="balanced", random_state=42, verbose=-1),
        "xgb": xgb.XGBClassifier(**xgb_p, eval_metric="mlogloss", random_state=42, verbosity=0),
        "catboost": CatBoostClassifier(**catboost_p, random_seed=42, verbose=0, class_weights=[1.0, 1.2, 1.0]),
        "rf": RandomForestClassifier(**rf_p, random_state=42, n_jobs=-1, class_weight="balanced"),
        "et": ExtraTreesClassifier(**et_p, random_state=42, n_jobs=-1, class_weight="balanced"),
        "gb": HistGradientBoostingClassifier(**gb_p, random_state=42),
        "lr": LogisticRegression(**lr_p, solver="lbfgs", max_iter=1000, random_state=42),
        "mlp": MLPClassifier(hidden_layer_sizes=(256, 128, 64), activation="relu", max_iter=500, early_stopping=True, random_state=42, **mlp_p),
        "knn": KNeighborsClassifier(**knn_p, weights="distance", n_jobs=-1),
        "svm": CalibratedClassifierCV(LinearSVC(C=svm_p["C"], max_iter=2000, random_state=42, dual=False), cv=3),
        "poisson":      PoissonBaseModel(),
        "rolling_mean": RollingMeanBaseModel(),
    }


def get_meta_learners() -> dict:
    return {
        "meta_lr": LogisticRegression(
            C=0.5, solver="lbfgs",
            max_iter=1000, random_state=42,
        ),
        "meta_lgbm": lgb.LGBMClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.04,
            num_leaves=15, verbose=-1, random_state=42,
        ),
        "meta_mlp": MLPClassifier(
            hidden_layer_sizes=(128, 64), activation="relu",
            max_iter=400, early_stopping=True, random_state=42,
        ),
        "meta_ridge": LogisticRegression(
            C=0.1, solver="lbfgs",
            max_iter=1000, random_state=99,
        ),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  CROSS-VALIDATED OOF STACKING
# ══════════════════════════════════════════════════════════════════════════════

_NO_SAMPLE_WEIGHT = {"knn", "poisson", "rolling_mean", "mlp"}
_FORMULA_MODELS   = {"poisson", "rolling_mean"}


def cross_val_oof(
    models: dict,
    X: np.ndarray,          # scaled — for tree/linear/SVM/MLP models
    X_orig: pd.DataFrame,   # unscaled — for formula models that use raw feature values
    y: np.ndarray,
    weights: np.ndarray,
    feature_names: list,
    n_folds: int = N_FOLDS,
) -> np.ndarray:
    """
    Returns OOF prediction matrix of shape (n_samples, n_models * 3).
    Each block of 3 columns = [p_win_a, p_draw, p_win_b] from one base model.
    Formula models (Poisson, RollingMean) receive the original unscaled DataFrame
    so their feature lookups (win_rate, gf_avg, etc.) are in natural ranges.
    """
    n        = len(y)
    n_models = len(models)
    oof      = np.zeros((n, n_models * 3), dtype=float)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"  Fold {fold_idx + 1}/{n_folds}  "
              f"(train={len(train_idx):,}  val={len(val_idx):,})")

        X_tr, X_val          = X[train_idx], X[val_idx]
        X_orig_tr = X_orig.iloc[train_idx].reset_index(drop=True)
        X_orig_val= X_orig.iloc[val_idx].reset_index(drop=True)
        y_tr                 = y[train_idx]
        w_tr                 = weights[train_idx]

        for mi, (name, model) in enumerate(models.items()):
            col = mi * 3
            try:
                if name in _FORMULA_MODELS:
                    model.fit(X_orig_tr, y_tr)
                    proba = model.predict_proba(X_orig_val)
                elif name in _NO_SAMPLE_WEIGHT:
                    model.fit(X_tr, y_tr)
                    proba = model.predict_proba(X_val)
                else:
                    model.fit(X_tr, y_tr, sample_weight=w_tr)
                    proba = model.predict_proba(X_val)
                oof[val_idx, col:col + 3] = proba
            except Exception as e:
                print(f"    [{name}] fold {fold_idx+1} error: {e}")
                oof[val_idx, col:col + 3] = 1.0 / 3.0

    return oof


# ══════════════════════════════════════════════════════════════════════════════
#  EVALUATION HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(y_true: np.ndarray, proba: np.ndarray, name: str = "") -> dict:
    pred = proba.argmax(axis=1)
    ll   = log_loss(y_true, proba, labels=[0, 1, 2])
    acc  = accuracy_score(y_true, pred)
    # Brier score (multiclass): mean squared error of probabilities
    n = len(y_true)
    oh = np.zeros((n, 3))
    oh[np.arange(n), y_true] = 1
    brier = float(np.mean((proba - oh) ** 2))
    metrics = {"log_loss": round(ll, 4), "accuracy": round(acc, 4), "brier": round(brier, 4)}
    if name:
        print(f"  {name:20s}  log_loss={ll:.4f}  acc={acc:.4f}  brier={brier:.4f}")
    return metrics


# ══════════════════════════════════════════════════════════════════════════════
#  OPTUNA TUNING
# ══════════════════════════════════════════════════════════════════════════════

def run_optuna_tuning(X, y, weights, n_trials=20) -> dict:
    print(f"\n{'='*60}\nRunning Optuna Hyperparameter Tuning ({n_trials} trials/model)\n{'='*60}")
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    def cv_score(model) -> float:
        losses = []
        for tr_idx, val_idx in skf.split(X, y):
            try:
                model.fit(X[tr_idx], y[tr_idx], sample_weight=weights[tr_idx])
            except TypeError:
                model.fit(X[tr_idx], y[tr_idx])
            proba = model.predict_proba(X[val_idx])
            losses.append(log_loss(y[val_idx], proba, labels=[0, 1, 2]))
        return float(np.mean(losses))

    def obj_lgbm(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }
        m = lgb.LGBMClassifier(**params, class_weight="balanced", random_state=42, verbose=-1)
        return cv_score(m)

    def obj_xgb(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }
        m = xgb.XGBClassifier(**params, eval_metric="mlogloss", random_state=42, verbosity=0)
        return cv_score(m)

    def obj_catboost(trial):
        params = {
            "iterations": trial.suggest_int("iterations", 300, 800, step=100),
            "depth": trial.suggest_int("depth", 4, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        }
        m = CatBoostClassifier(**params, random_seed=42, verbose=0, class_weights=[1.0, 1.2, 1.0])
        return cv_score(m)

    def obj_rf(trial):
        params = {"n_estimators": trial.suggest_int("n_estimators", 200, 600, step=100), "max_depth": trial.suggest_int("max_depth", 6, 20), "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 10)}
        m = RandomForestClassifier(**params, random_state=42, n_jobs=-1, class_weight="balanced")
        return cv_score(m)

    def obj_et(trial):
        params = {"n_estimators": trial.suggest_int("n_estimators", 200, 600, step=100), "max_depth": trial.suggest_int("max_depth", 6, 20), "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 10)}
        m = ExtraTreesClassifier(**params, random_state=42, n_jobs=-1, class_weight="balanced")
        return cv_score(m)

    def obj_gb(trial):
        params = {"max_iter": trial.suggest_int("max_iter", 100, 500, step=100), "max_depth": trial.suggest_int("max_depth", 3, 7), "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True)}
        m = HistGradientBoostingClassifier(**params, random_state=42)
        return cv_score(m)

    def obj_lr(trial):
        params = {"C": trial.suggest_float("C", 0.01, 10.0, log=True)}
        m = LogisticRegression(**params, solver="lbfgs", max_iter=1000, random_state=42)
        return cv_score(m)

    def obj_mlp(trial):
        params = {"alpha": trial.suggest_float("alpha", 1e-5, 1e-1, log=True), "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True)}
        m = MLPClassifier(hidden_layer_sizes=(256, 128, 64), activation="relu", max_iter=500, early_stopping=True, random_state=42, **params)
        return cv_score(m)

    def obj_svm(trial):
        params = {"C": trial.suggest_float("C", 0.01, 10.0, log=True)}
        m = CalibratedClassifierCV(LinearSVC(**params, max_iter=2000, random_state=42), cv=3)
        return cv_score(m)

    def obj_knn(trial):
        params = {"n_neighbors": trial.suggest_int("n_neighbors", 5, 50)}
        m = KNeighborsClassifier(**params, weights="distance", n_jobs=-1)
        return cv_score(m)

    best_params = {}
    models_to_tune = [
        ("lgbm", obj_lgbm), ("xgb", obj_xgb), ("catboost", obj_catboost),
        ("rf", obj_rf), ("et", obj_et), ("gb", obj_gb),
        ("lr", obj_lr), ("mlp", obj_mlp), ("svm", obj_svm), ("knn", obj_knn)
    ]
    for name, obj in models_to_tune:
        print(f"\nTuning {name}...")
        study = optuna.create_study(direction="minimize")
        study.optimize(obj, n_trials=n_trials)
        best_params[name] = study.best_params
        print(f"Best {name} log_loss: {study.best_value:.4f}")
        print(f"Best {name} params: {study.best_params}")

    params_path = MODELS_DIR / "best_params.json"
    with open(params_path, "w") as f:
        json.dump(best_params, f, indent=2)
    print(f"\\nTuned parameters saved to {params_path}")
    return best_params


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN TRAINING ORCHESTRATION
# ══════════════════════════════════════════════════════════════════════════════

def train_and_save(optuna_trials: int = 0, shap_prune: int = 0):
    print("=" * 70)
    print("TRAINING PIPELINE — World Cup 2026 stacking ensemble")
    print("=" * 70)

    # 1. Load data
    print("\n[1/6] Loading data...")
    df   = load_data()
    elos = load_elos()
    print(f"  Dataset: {len(df):,} completed matches  |  ELOs: {len(elos)} teams")

    # 2. Feature engineering
    print("Loading data...")
    df = load_data()
    elos = load_elos()
    
    tactical = None
    if os.path.exists("data/statsbomb_match_stats.csv"):
        print("Loading StatsBomb tactical data...")
        df_tactical = pd.read_csv("data/statsbomb_match_stats.csv")
        tactical = TacticalStatsComputer(df_tactical)

    rolling = RollingStatsComputer(df)
    h2h = H2HComputer(df)

    # 3. Build training dataset
    print("\n[3/6] Building training dataset...")
    X_df, y, weights = build_training_dataset(df, elos, rolling, h2h, tactical=tactical)
    print(f"  Shape: {X_df.shape}  |  Class distribution: "
          f"W={int((y==0).sum())}  D={int((y==1).sum())}  L={int((y==2).sum())}")
          
    if shap_prune > 0:
        print(f"\n[3.5] Pruning features to top {shap_prune} using SHAP...")
        import lightgbm as lgb
        import shap
        
        prune_model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, random_state=42, verbose=-1)
        prune_model.fit(X_df, y)
        
        explainer = shap.TreeExplainer(prune_model)
        shap_values = explainer.shap_values(X_df)
        
        if isinstance(shap_values, list):
            mean_abs_shap = np.abs(np.array(shap_values)).mean(axis=0).mean(axis=0)
        else:
            if len(shap_values.shape) == 3:
                mean_abs_shap = np.abs(shap_values).mean(axis=0).mean(axis=1)
            else:
                mean_abs_shap = np.abs(shap_values).mean(axis=0)
            
        feature_importance = pd.DataFrame({
            'feature': X_df.columns,
            'importance': mean_abs_shap
        }).sort_values('importance', ascending=False)
        
        top_features = feature_importance.head(shap_prune)['feature'].tolist()
        print(f"  Selected top {shap_prune} features.")
        print(f"  Top 5: {top_features[:5]}")
        
        X_df = X_df[top_features]
        print(f"  New Shape: {X_df.shape}")

    feature_names = list(X_df.columns)
    with open(MODELS_DIR / "feature_names.json", "w") as f:
        json.dump(feature_names, f)
    print(f"  Features: {len(feature_names)}")

    # 4. Scale
    print("\n[4/6] Scaling features & running OOF cross-validation...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_df.values)
    pickle.dump(scaler, open(MODELS_DIR / "scaler.pkl", "wb"))

    # 4.5 Optuna Tuning (Optional)
    best_params = None
    if optuna_trials > 0:
        best_params = run_optuna_tuning(X_scaled, y, weights, optuna_trials)
    else:
        params_path = MODELS_DIR / "best_params.json"
        if params_path.exists():
            print(f"\\n[4.5] Loading tuned hyperparameters from {params_path}")
            with open(params_path) as f:
                best_params = json.load(f)

    # 5. OOF stacking
    base_models  = get_base_models(best_params)
    model_names  = list(base_models.keys())

    print(f"  Base models: {model_names}")
    oof_matrix = cross_val_oof(
        base_models, X_scaled, X_df, y, weights, feature_names, N_FOLDS
    )

    # Evaluate individual OOF performance
    print("\n  OOF base model performance:")
    oof_metrics = {}
    for mi, name in enumerate(model_names):
        col = mi * 3
        proba = oof_matrix[:, col:col + 3]
        oof_metrics[name] = evaluate(y, proba, name)

    # 6. Retrain base models on full data, train meta-learners
    print("\n[5/6] Retraining base models on full dataset...")
    trained_base = {}
    for name, model in base_models.items():
        print(f"  Fitting {name}...")
        try:
            if name in _FORMULA_MODELS:
                model.fit(X_df, y)
            elif name in _NO_SAMPLE_WEIGHT:
                model.fit(X_scaled, y)
            else:
                model.fit(X_scaled, y, sample_weight=weights)
            trained_base[name] = model
            pickle.dump(model, open(MODELS_DIR / f"base_{name}.pkl", "wb"))
        except Exception as e:
            print(f"  [{name}] failed: {e}")

    print("\n[6/6] Training meta-learners on OOF predictions...")
    meta_learners = get_meta_learners()
    trained_meta  = {}

    # Meta-learner input: OOF matrix (n_samples, n_models*3)
    meta_scaler = StandardScaler()
    oof_scaled  = meta_scaler.fit_transform(oof_matrix)
    pickle.dump(meta_scaler, open(MODELS_DIR / "meta_scaler.pkl", "wb"))

    meta_eval = {}
    for name, model in meta_learners.items():
        print(f"  Fitting {name}...")
        try:
            model.fit(oof_scaled, y)
            proba = model.predict_proba(oof_scaled)
            meta_eval[name] = evaluate(y, proba, name)
            trained_meta[name] = model
            pickle.dump(model, open(MODELS_DIR / f"{name}.pkl", "wb"))
        except Exception as e:
            print(f"  [{name}] failed: {e}")

    # Ensemble of meta-learners
    meta_probas = [m.predict_proba(oof_scaled) for m in trained_meta.values()]
    ensemble_proba = np.mean(meta_probas, axis=0)
    print("\n  Final ensemble OOF:")
    ensemble_metrics = evaluate(y, ensemble_proba, "ensemble (mean)")

    # Save training report
    report = {
        "n_training_rows":  int(len(y)),
        "n_features":       len(feature_names),
        "class_counts":     {"win": int((y==0).sum()), "draw": int((y==1).sum()), "loss": int((y==2).sum())},
        "base_model_names": model_names,
        "meta_model_names": list(meta_learners.keys()),
        "oof_metrics":      oof_metrics,
        "meta_metrics":     meta_eval,
        "ensemble_metrics": ensemble_metrics,
    }
    with open(MODELS_DIR / "training_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nAll models saved to {MODELS_DIR}/")
    print(f"Training report: {MODELS_DIR}/training_report.json")
    print("=" * 70)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="World Cup 2026 Stacking Ensemble Pipeline")
    parser.add_argument("--optuna-trials", type=int, default=0, help="Number of optuna trials per model to run before training")
    parser.add_argument("--shap-prune", type=int, default=0, help="Number of top features to keep using SHAP importance (0 to disable)")
    args = parser.parse_args()

    report = train_and_save(optuna_trials=args.optuna_trials, shap_prune=args.shap_prune)
    print("\nFinal ensemble OOF metrics:")
    print(json.dumps(report["ensemble_metrics"], indent=2))
