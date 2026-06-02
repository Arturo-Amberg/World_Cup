"""
ensemble_predictor.py — Inference wrapper for the trained stacking ensemble.

Loads all base models + meta-learners saved by training_pipeline.py and
provides a predict() interface compatible with stacked_predictor.py.

Extra components at inference time:
  - SARIMAX(1,0,1): per-team time-series forecast of GF/GA trend → Poisson probs
  - Neutral-venue correction: predictions are symmetrised (predict both directions
    and average) since all WC matches are played at neutral venues.

Final blend (configurable):
  85 % stacking ensemble (mean of meta-learners)
  15 % SARIMAX Poisson  (if SARIMAX converges for both teams, else 0 %)
"""

import json
import math
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson as sp_poisson

warnings.filterwarnings("ignore")

BASE_DIR   = Path(__file__).parent.parent.parent
DATA_DIR   = BASE_DIR / "data"
MODELS_DIR = DATA_DIR / "models"
CSV_PATH   = DATA_DIR / "intl_results.csv"
ELOS_PATH  = DATA_DIR / "elos.json"

WC_AVG_GOALS  = 1.30
BASELINE_ELO  = 1750
SARIMAX_BLEND = 0.15   # weight given to SARIMAX at prediction time

_DATASET_TO_ELO = {
    "United States": "USA", "South Korea": "South Korea",
    "Ivory Coast": "Ivory Coast", "DR Congo": "DR Congo",
    "Bosnia and Herzegovina": "Bosnia & Herzegovina",
    "Curaçao": "Curaçao", "Cape Verde": "Cape Verde",
    "Turkey": "Turkey", "Iran": "Iran",
}

# Our internal names → CSV dataset names (for rolling feature lookup)
_INTERNAL_TO_DATASET = {
    "USA": "United States", "Bosnia & Herzegovina": "Bosnia and Herzegovina",
}


def _team_elo(name: str, elos: dict) -> float:
    canonical = _DATASET_TO_ELO.get(name, name)
    if canonical in elos:
        return float(elos[canonical])
    low = canonical.lower()
    for k, v in elos.items():
        if low in k.lower() or k.lower() in low:
            return float(v)
    return BASELINE_ELO


def _poisson_3way(lam_a: float, lam_b: float, max_g: int = 9) -> np.ndarray:
    pa = pd_ = pb = 0.0
    rho = -0.12
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
            if   i > j: pa  += p
            elif i == j: pd_ += p
            else:        pb  += p
    tot = pa + pd_ + pb or 1.0
    return np.array([pa / tot, pd_ / tot, pb / tot])


# ══════════════════════════════════════════════════════════════════════════════
#  SARIMAX GOAL FORECASTER
# ══════════════════════════════════════════════════════════════════════════════

class SARIMAXForecaster:
    """
    Fits ARIMA(1,0,1) on a team's recent GF and GA series to forecast
    their expected goals in the next match.
    """

    def __init__(self, df: pd.DataFrame):
        self._df = df
        self._cache: dict = {}

    def _team_series(self, team: str) -> tuple:
        """Returns (gf_series, ga_series) as numpy arrays sorted by date."""
        if team in self._cache:
            return self._cache[team]

        home = self._df[self._df["home_team"] == team][["date","home_score","away_score"]].copy()
        home.columns = ["date","gf","ga"]
        away = self._df[self._df["away_team"] == team][["date","home_score","away_score"]].copy()
        away.columns = ["date","ga","gf"]

        hist = pd.concat([home, away]).sort_values("date").tail(30)
        gf = hist["gf"].values.astype(float)
        ga = hist["ga"].values.astype(float)
        self._cache[team] = (gf, ga)
        return gf, ga

    def forecast_goals(self, team: str) -> tuple[float, float]:
        """
        Returns (forecast_gf, forecast_ga). Falls back to rolling average
        if ARIMA fails to converge.
        """
        from statsmodels.tsa.arima.model import ARIMA

        gf, ga = self._team_series(team)
        if len(gf) < 8:
            return float(gf.mean()) if len(gf) else WC_AVG_GOALS, \
                   float(ga.mean()) if len(ga) else WC_AVG_GOALS

        def _fit_arima(series: np.ndarray) -> float:
            try:
                res = ARIMA(series, order=(1, 0, 1)).fit()
                fc = res.forecast(steps=1)[0]
                return max(0.2, float(fc))
            except Exception:
                return float(series.mean())

        return _fit_arima(gf), _fit_arima(ga)

    def predict_proba(self, team_a: str, team_b: str) -> np.ndarray:
        """Returns [p_win_a, p_draw, p_win_b] using ARIMA forecasts + Poisson."""
        gf_a, ga_a = self.forecast_goals(team_a)
        gf_b, ga_b = self.forecast_goals(team_b)

        mu   = WC_AVG_GOALS
        la   = max(0.3, gf_a) * max(0.3, ga_b) / mu
        lb   = max(0.3, gf_b) * max(0.3, ga_a) / mu
        return _poisson_3way(la, lb)


# ══════════════════════════════════════════════════════════════════════════════
#  ROLLING FEATURE BUILDER  (mirrors training_pipeline.py)
# ══════════════════════════════════════════════════════════════════════════════

ROLLING_WINDOWS = [5, 10, 15, 30]
EWMA_ALPHA      = 0.3


def _ewma(vals: np.ndarray, alpha: float) -> float:
    if len(vals) == 0:
        return WC_AVG_GOALS
    w = (1 - alpha) ** np.arange(len(vals) - 1, -1, -1)
    return float((vals * w).sum() / w.sum())


def _rolling_stats(gf: np.ndarray, ga: np.ndarray, pts: np.ndarray,
                   window: int, suffix: str) -> dict:
    n  = len(gf)
    s  = max(0, n - window)
    gf_, ga_, pts_ = gf[s:], ga[s:], pts[s:]
    if len(gf_) == 0:
        return {
            f"gf_avg_{window}{suffix}": WC_AVG_GOALS,
            f"ga_avg_{window}{suffix}": WC_AVG_GOALS,
            f"win_rate_{window}{suffix}": 0.33,
            f"draw_rate_{window}{suffix}": 0.25,
            f"pts_avg_{window}{suffix}": 1.0,
            f"gd_avg_{window}{suffix}": 0.0,
            f"cs_rate_{window}{suffix}": 0.25,
        }
    return {
        f"gf_avg_{window}{suffix}": float(gf_.mean()),
        f"ga_avg_{window}{suffix}": float(ga_.mean()),
        f"win_rate_{window}{suffix}": float((pts_ == 3).mean()),
        f"draw_rate_{window}{suffix}": float((pts_ == 1).mean()),
        f"pts_avg_{window}{suffix}": float(pts_.mean()),
        f"gd_avg_{window}{suffix}": float((gf_ - ga_).mean()),
        f"cs_rate_{window}{suffix}": float((ga_ == 0).mean()),
    }


def _build_team_arrays(df: pd.DataFrame, dataset_name: str) -> tuple:
    home = df[df["home_team"] == dataset_name][["date","home_score","away_score"]].copy()
    home.columns = ["date","gf","ga"]
    away = df[df["away_team"] == dataset_name][["date","home_score","away_score"]].copy()
    away.columns = ["date","ga","gf"]
    hist = pd.concat([home, away]).sort_values("date").reset_index(drop=True)
    gf   = hist["gf"].values.astype(float)
    ga   = hist["ga"].values.astype(float)
    pts  = np.where(gf > ga, 3.0, np.where(gf == ga, 1.0, 0.0))
    return gf, ga, pts


def _team_rolling_features(df: pd.DataFrame, internal_name: str, suffix: str) -> dict:
    dataset_name = _INTERNAL_TO_DATASET.get(internal_name, internal_name)
    gf, ga, pts  = _build_team_arrays(df, dataset_name)

    feat: dict = {}
    for w in ROLLING_WINDOWS:
        feat.update(_rolling_stats(gf, ga, pts, w, suffix))
    feat[f"gf_ewma{suffix}"] = _ewma(gf[-30:], EWMA_ALPHA)
    feat[f"ga_ewma{suffix}"] = _ewma(ga[-30:], EWMA_ALPHA)
    feat[f"n_prior{suffix}"] = len(gf)
    return feat


def _h2h_features(df: pd.DataFrame, a_internal: str, b_internal: str) -> dict:
    a_ds = _INTERNAL_TO_DATASET.get(a_internal, a_internal)
    b_ds = _INTERNAL_TO_DATASET.get(b_internal, b_internal)

    mask = (
        ((df["home_team"] == a_ds) & (df["away_team"] == b_ds)) |
        ((df["home_team"] == b_ds) & (df["away_team"] == a_ds))
    )
    h2h = df[mask].tail(20)
    ng  = len(h2h)
    if ng == 0:
        return {"h2h_win_rate_a": 0.33, "h2h_draw_rate": 0.25, "h2h_games": 0}

    wins_a = draws = 0
    for _, row in h2h.iterrows():
        hs  = row["home_score"] if row["home_team"] == a_ds else row["away_score"]
        as_ = row["away_score"] if row["home_team"] == a_ds else row["home_score"]
        if hs > as_: wins_a += 1
        elif hs == as_: draws += 1

    return {
        "h2h_win_rate_a": wins_a / ng,
        "h2h_draw_rate":  draws  / ng,
        "h2h_games":      min(ng, 20),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  ENSEMBLE PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════

class EnsemblePredictor:
    """
    Loads the trained stacking ensemble and provides match outcome predictions.

    Usage:
        ep = EnsemblePredictor()
        ep.load()
        p_win_a, p_draw, p_win_b = ep.predict(team_a_dict, team_b_dict)
    """

    def __init__(self):
        self.feature_names: list = []
        self.scaler       = None
        self.meta_scaler  = None
        self.base_models  : dict = {}
        self.meta_models  : dict = {}
        self._df          = None
        self._elos        : dict = {}
        self._sarimax     = None
        self._loaded      = False

    # ── Loading ──────────────────────────────────────────────────────────────

    def load(self) -> "EnsemblePredictor":
        if self._loaded:
            return self

        if not MODELS_DIR.exists():
            raise FileNotFoundError(
                f"Models not found at {MODELS_DIR}. Run training_pipeline.py first."
            )

        # The base model pickles were saved with __main__ as the module path
        # (training_pipeline.py was run directly). Remap to the real module so
        # unpickling works regardless of how this code is invoked.
        import sys
        import src.pipelines.training_pipeline as _tp
        for _attr in dir(_tp):
            if not _attr.startswith("_"):
                setattr(sys.modules.get("__main__", _tp), _attr, getattr(_tp, _attr))

        def _safe_load(path):
            with open(path, "rb") as fh:
                return pickle.load(fh)

        self.feature_names = json.loads(
            (MODELS_DIR / "feature_names.json").read_text()
        )
        self.scaler      = _safe_load(MODELS_DIR / "scaler.pkl")
        self.meta_scaler = _safe_load(MODELS_DIR / "meta_scaler.pkl")

        for p in MODELS_DIR.glob("base_*.pkl"):
            name = p.stem[5:]  # strip "base_"
            self.base_models[name] = _safe_load(p)

        for p in MODELS_DIR.glob("meta_*.pkl"):
            name = p.stem        # e.g. "meta_lr"
            self.meta_models[name] = _safe_load(p)

        # Data for rolling features
        self._df   = pd.read_csv(CSV_PATH, parse_dates=["date"])
        self._df   = self._df[self._df["home_score"].notna()].copy()
        self._df["home_score"] = self._df["home_score"].astype(int)
        self._df["away_score"] = self._df["away_score"].astype(int)

        self._elos = json.loads(ELOS_PATH.read_text()).get("data", {})

        self._sarimax = SARIMAXForecaster(self._df)
        self._loaded  = True
        print(f"[EnsemblePredictor] Loaded {len(self.base_models)} base models, "
              f"{len(self.meta_models)} meta-learners, "
              f"{len(self.feature_names)} features.")
        return self

    # ── Feature construction ─────────────────────────────────────────────────

    def _build_feature_row(
        self,
        name_a: str,
        name_b: str,
        elo_a: float,
        elo_b: float,
        is_neutral: float = 1.0,
        tournament_type: int = 5,
    ) -> pd.DataFrame:
        feat: dict = {}
        feat.update(_team_rolling_features(self._df, name_a, "_a"))
        feat.update(_team_rolling_features(self._df, name_b, "_b"))
        feat.update(_h2h_features(self._df, name_a, name_b))

        feat["elo_a"]    = elo_a
        feat["elo_b"]    = elo_b
        feat["elo_diff"] = elo_a - elo_b

        feat["is_neutral"]      = is_neutral
        feat["tournament_type"] = tournament_type

        for w in ROLLING_WINDOWS:
            feat[f"gf_diff_{w}"]  = feat[f"gf_avg_{w}_a"]  - feat[f"gf_avg_{w}_b"]
            feat[f"ga_diff_{w}"]  = feat[f"ga_avg_{w}_a"]  - feat[f"ga_avg_{w}_b"]
            feat[f"pts_diff_{w}"] = feat[f"pts_avg_{w}_a"] - feat[f"pts_avg_{w}_b"]
            feat[f"win_diff_{w}"] = feat[f"win_rate_{w}_a"]- feat[f"win_rate_{w}_b"]
        feat["gf_ewma_diff"] = feat["gf_ewma_a"] - feat["gf_ewma_b"]
        feat["ga_ewma_diff"] = feat["ga_ewma_a"] - feat["ga_ewma_b"]

        # Ensure correct column order and fill any missing with 0
        row = {fn: feat.get(fn, 0.0) for fn in self.feature_names}
        return pd.DataFrame([row])

    # ── Base model inference ─────────────────────────────────────────────────

    def _base_predictions(
        self, X_df: pd.DataFrame, X_scaled: np.ndarray
    ) -> np.ndarray:
        """Returns (1, n_models * 3) array of base model probabilities."""
        parts = []
        for name, model in self.base_models.items():
            try:
                if name in ("poisson", "rolling_mean"):
                    p = model.predict_proba(X_df)
                else:
                    p = model.predict_proba(X_scaled)
                parts.append(p[0])
            except Exception:
                parts.append(np.array([1/3, 1/3, 1/3]))
        return np.array(parts).flatten().reshape(1, -1)

    # ── Main prediction ──────────────────────────────────────────────────────

    def predict(
        self,
        team_a: dict,
        team_b: dict,
        home_team: str | None = None,
        is_neutral: bool = True,
        tournament_type: int = 5,
    ) -> tuple[float, float, float]:
        """
        Returns (p_win_a, p_draw, p_win_b).

        team_a / team_b dicts must have at minimum: {"name": ..., "ELO": ...}
        """
        if not self._loaded:
            self.load()

        name_a = team_a["name"]
        name_b = team_b["name"]
        elo_a  = float(team_a.get("ELO", _team_elo(name_a, self._elos)))
        elo_b  = float(team_b.get("ELO", _team_elo(name_b, self._elos)))
        neutral_f = 1.0 if is_neutral else 0.0

        def _predict_single(na, nb, ea, eb, h2h_dir=1):
            """Predict with h2h_dir=1 (a=home side) or h2h_dir=-1 (swapped)."""
            row_df     = self._build_feature_row(na, nb, ea, eb, neutral_f, tournament_type)
            row_scaled = self.scaler.transform(row_df.values)

            base_vec   = self._base_predictions(row_df, row_scaled)
            meta_input = self.meta_scaler.transform(base_vec)

            meta_probas = []
            for model in self.meta_models.values():
                try:
                    meta_probas.append(model.predict_proba(meta_input)[0])
                except Exception:
                    meta_probas.append(np.array([1/3, 1/3, 1/3]))

            return np.mean(meta_probas, axis=0)  # [p_win_a, p_draw, p_win_b]

        # Neutral-venue symmetrisation: predict A-vs-B and swapped B-vs-A
        # then average, flipping win probs back to A's perspective
        p_fwd = _predict_single(name_a, name_b, elo_a, elo_b)
        p_rev = _predict_single(name_b, name_a, elo_b, elo_a)
        p_rev_flipped = np.array([p_rev[2], p_rev[1], p_rev[0]])

        ensemble_p = 0.5 * p_fwd + 0.5 * p_rev_flipped

        # SARIMAX blend
        try:
            sarimax_p = self._sarimax.predict_proba(name_a, name_b)
            final_p   = (1 - SARIMAX_BLEND) * ensemble_p + SARIMAX_BLEND * sarimax_p
        except Exception:
            final_p = ensemble_p

        # Normalise and floor
        final_p = np.maximum(final_p, 0.02)
        final_p /= final_p.sum()

        return float(final_p[0]), float(final_p[1]), float(final_p[2])

    def precompute_matchups(self, team_names: list) -> dict:
        """
        Pre-compute ensemble win probabilities for every ordered pair of teams.
        Returns {(team_a, team_b): (p_win_a, p_draw, p_win_b), ...}.
        Both directions are stored so lookup is O(1) during simulation.
        """
        if not self._loaded:
            self.load()

        cache: dict = {}
        names = list(team_names)
        total = len(names) * (len(names) - 1) // 2
        done = 0
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                ta = {"name": a, "ELO": _team_elo(a, self._elos)}
                tb = {"name": b, "ELO": _team_elo(b, self._elos)}
                pa, pd, pb = self.predict(ta, tb)
                cache[(a, b)] = (pa, pd, pb)
                cache[(b, a)] = (pb, pd, pa)
                done += 1
                if done % 100 == 0:
                    print(f"  [ensemble] pre-computed {done}/{total} matchups …", flush=True)

        print(f"  [ensemble] done — {len(cache)} matchup entries cached.", flush=True)
        return cache

    def predict_with_breakdown(
        self,
        team_a: dict,
        team_b: dict,
        **kwargs,
    ) -> dict:
        """Full breakdown including per-model probabilities."""
        if not self._loaded:
            self.load()

        name_a = team_a["name"]
        name_b = team_b["name"]
        elo_a  = float(team_a.get("ELO", _team_elo(name_a, self._elos)))
        elo_b  = float(team_b.get("ELO", _team_elo(name_b, self._elos)))
        neutral_f = 1.0

        row_df     = self._build_feature_row(name_a, name_b, elo_a, elo_b)
        row_scaled = self.scaler.transform(row_df.values)

        base_breakdown = {}
        base_vec_parts = []
        for name, model in self.base_models.items():
            try:
                if name in ("poisson", "rolling_mean"):
                    p = model.predict_proba(row_df)[0]
                else:
                    p = model.predict_proba(row_scaled)[0]
                base_breakdown[name] = {
                    "win_a": round(float(p[0]), 3),
                    "draw":  round(float(p[1]), 3),
                    "win_b": round(float(p[2]), 3),
                }
                base_vec_parts.append(p)
            except Exception as e:
                base_breakdown[name] = {"error": str(e)}
                base_vec_parts.append(np.array([1/3, 1/3, 1/3]))

        base_vec   = np.array(base_vec_parts).flatten().reshape(1, -1)
        meta_input = self.meta_scaler.transform(base_vec)

        meta_breakdown = {}
        meta_probas    = []
        for name, model in self.meta_models.items():
            try:
                p = model.predict_proba(meta_input)[0]
                meta_breakdown[name] = {
                    "win_a": round(float(p[0]), 3),
                    "draw":  round(float(p[1]), 3),
                    "win_b": round(float(p[2]), 3),
                }
                meta_probas.append(p)
            except Exception as e:
                meta_breakdown[name] = {"error": str(e)}

        ensemble_p = np.mean(meta_probas, axis=0) if meta_probas else np.array([1/3, 1/3, 1/3])

        try:
            sarimax_p = self._sarimax.predict_proba(name_a, name_b)
            final_p   = (1 - SARIMAX_BLEND) * ensemble_p + SARIMAX_BLEND * sarimax_p
            sarimax_d = {
                "win_a": round(float(sarimax_p[0]), 3),
                "draw":  round(float(sarimax_p[1]), 3),
                "win_b": round(float(sarimax_p[2]), 3),
            }
        except Exception:
            final_p   = ensemble_p
            sarimax_d = None

        final_p = np.maximum(final_p, 0.02)
        final_p /= final_p.sum()

        return {
            "p_win_a": round(float(final_p[0]), 4),
            "p_draw":  round(float(final_p[1]), 4),
            "p_win_b": round(float(final_p[2]), 4),
            "base_models":    base_breakdown,
            "meta_learners":  meta_breakdown,
            "sarimax":        sarimax_d,
            "ensemble_raw": {
                "win_a": round(float(ensemble_p[0]), 4),
                "draw":  round(float(ensemble_p[1]), 4),
                "win_b": round(float(ensemble_p[2]), 4),
            },
        }


# ══════════════════════════════════════════════════════════════════════════════
#  Quick test
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import json as _json

    ep = EnsemblePredictor()
    ep.load()

    test_matches = [
        ({"name": "Spain",   "ELO": 2165}, {"name": "France",  "ELO": 2081}),
        ({"name": "Brazil",  "ELO": 1984}, {"name": "Argentina","ELO": 2095}),
        ({"name": "USA",     "ELO": 1721}, {"name": "England",  "ELO": 2020}),
        ({"name": "Japan",   "ELO": 1904}, {"name": "Germany",  "ELO": 1969}),
        ({"name": "Morocco", "ELO": 1861}, {"name": "Netherlands","ELO": 1961}),
    ]

    print("\n=== Ensemble predictions ===\n")
    for ta, tb in test_matches:
        result = ep.predict_with_breakdown(ta, tb)
        print(f"{ta['name']} vs {tb['name']}")
        print(f"  Win A: {result['p_win_a']:.3f}  Draw: {result['p_draw']:.3f}  Win B: {result['p_win_b']:.3f}")
        print(f"  Base models:")
        for m, v in result["base_models"].items():
            if "error" not in v:
                print(f"    {m:15s} {v['win_a']:.3f} / {v['draw']:.3f} / {v['win_b']:.3f}")
        print(f"  Meta-learners:")
        for m, v in result["meta_learners"].items():
            if "error" not in v:
                print(f"    {m:15s} {v['win_a']:.3f} / {v['draw']:.3f} / {v['win_b']:.3f}")
        if result["sarimax"]:
            s = result["sarimax"]
            print(f"  SARIMAX:        {s['win_a']:.3f} / {s['draw']:.3f} / {s['win_b']:.3f}")
        print()
