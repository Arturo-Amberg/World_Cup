"""
calibrate.py — Bayesian calibration of stacked_predictor weights.

Embeds WC 2022 group stage results and backtests each model independently,
then runs a grid search to find the minimum combined log-loss weight combination.

Usage:
    python calibrate.py
"""

import math
import itertools

from src.models.stacked_predictor import elo_model, poisson_model, _get_ml_predictor

# ─────────────────────────────────────────────
#  WC 2022 embedded match data
# ─────────────────────────────────────────────
# Fields per match:
#   name_a, name_b,
#   elo_a, elo_b,
#   gf_avg_a, ga_avg_a, gf_avg_b, ga_avg_b,
#   forma_a, forma_b,
#   injuries_a, injuries_b,
#   result:  "A" = team A won, "D" = draw, "B" = team B won

WC2022_MATCHES = [
    # ── Group A ──
    # Qatar 1-0 Ecuador — home side (Qatar ELO ~1575, Ecuador ~1714)
    {"name_a": "Qatar",   "name_b": "Ecuador",
     "elo_a": 1575, "elo_b": 1714,
     "gf_avg_a": 0.9, "ga_avg_a": 1.3, "gf_avg_b": 1.3, "ga_avg_b": 1.1,
     "forma_a": 1.1, "forma_b": 1.5, "inj_a": 0, "inj_b": 0,
     "result": "A"},
    # Senegal 0-2 Netherlands
    {"name_a": "Senegal",      "name_b": "Netherlands",
     "elo_a": 1798, "elo_b": 1906,
     "gf_avg_a": 1.4, "ga_avg_a": 1.0, "gf_avg_b": 1.7, "ga_avg_b": 1.0,
     "forma_a": 1.6, "forma_b": 1.8, "inj_a": 1, "inj_b": 0,
     "result": "B"},
    # Qatar 1-3 Senegal
    {"name_a": "Qatar",   "name_b": "Senegal",
     "elo_a": 1575, "elo_b": 1798,
     "gf_avg_a": 0.9, "ga_avg_a": 1.3, "gf_avg_b": 1.4, "ga_avg_b": 1.0,
     "forma_a": 1.1, "forma_b": 1.6, "inj_a": 0, "inj_b": 0,
     "result": "B"},
    # Netherlands 1-1 Ecuador
    {"name_a": "Netherlands", "name_b": "Ecuador",
     "elo_a": 1906, "elo_b": 1714,
     "gf_avg_a": 1.7, "ga_avg_a": 1.0, "gf_avg_b": 1.3, "ga_avg_b": 1.1,
     "forma_a": 1.8, "forma_b": 1.5, "inj_a": 0, "inj_b": 0,
     "result": "D"},
    # Netherlands 2-0 Qatar
    {"name_a": "Netherlands", "name_b": "Qatar",
     "elo_a": 1906, "elo_b": 1575,
     "gf_avg_a": 1.7, "ga_avg_a": 1.0, "gf_avg_b": 0.9, "ga_avg_b": 1.3,
     "forma_a": 1.8, "forma_b": 1.1, "inj_a": 0, "inj_b": 0,
     "result": "A"},
    # Ecuador 1-2 Senegal
    {"name_a": "Ecuador", "name_b": "Senegal",
     "elo_a": 1714, "elo_b": 1798,
     "gf_avg_a": 1.3, "ga_avg_a": 1.1, "gf_avg_b": 1.4, "ga_avg_b": 1.0,
     "forma_a": 1.5, "forma_b": 1.6, "inj_a": 0, "inj_b": 0,
     "result": "B"},

    # ── Group B ──
    # England 6-2 Iran
    {"name_a": "England", "name_b": "Iran",
     "elo_a": 1915, "elo_b": 1783,
     "gf_avg_a": 1.7, "ga_avg_a": 0.9, "gf_avg_b": 1.3, "ga_avg_b": 1.0,
     "forma_a": 1.9, "forma_b": 1.5, "inj_a": 0, "inj_b": 1,
     "result": "A"},
    # USA 1-1 Wales
    {"name_a": "USA",   "name_b": "Wales",
     "elo_a": 1817, "elo_b": 1770,
     "gf_avg_a": 1.4, "ga_avg_a": 1.1, "gf_avg_b": 1.2, "ga_avg_b": 1.1,
     "forma_a": 1.5, "forma_b": 1.4, "inj_a": 0, "inj_b": 1,
     "result": "D"},
    # England 0-0 USA
    {"name_a": "England", "name_b": "USA",
     "elo_a": 1915, "elo_b": 1817,
     "gf_avg_a": 1.7, "ga_avg_a": 0.9, "gf_avg_b": 1.4, "ga_avg_b": 1.1,
     "forma_a": 1.9, "forma_b": 1.5, "inj_a": 0, "inj_b": 0,
     "result": "D"},
    # Wales 0-2 Iran
    {"name_a": "Wales", "name_b": "Iran",
     "elo_a": 1770, "elo_b": 1783,
     "gf_avg_a": 1.2, "ga_avg_a": 1.1, "gf_avg_b": 1.3, "ga_avg_b": 1.0,
     "forma_a": 1.4, "forma_b": 1.5, "inj_a": 1, "inj_b": 0,
     "result": "B"},
    # England 3-0 Wales
    {"name_a": "England", "name_b": "Wales",
     "elo_a": 1915, "elo_b": 1770,
     "gf_avg_a": 1.7, "ga_avg_a": 0.9, "gf_avg_b": 1.2, "ga_avg_b": 1.1,
     "forma_a": 1.9, "forma_b": 1.4, "inj_a": 0, "inj_b": 1,
     "result": "A"},
    # Iran 0-1 USA
    {"name_a": "Iran",  "name_b": "USA",
     "elo_a": 1783, "elo_b": 1817,
     "gf_avg_a": 1.3, "ga_avg_a": 1.0, "gf_avg_b": 1.4, "ga_avg_b": 1.1,
     "forma_a": 1.5, "forma_b": 1.5, "inj_a": 0, "inj_b": 0,
     "result": "B"},

    # ── Group C ──
    # Argentina 1-2 Saudi Arabia  *** GIANT UPSET ***
    {"name_a": "Argentina",   "name_b": "Saudi Arabia",
     "elo_a": 1974, "elo_b": 1651,
     "gf_avg_a": 2.1, "ga_avg_a": 0.6, "gf_avg_b": 1.1, "ga_avg_b": 1.2,
     "forma_a": 2.4, "forma_b": 1.3, "inj_a": 0, "inj_b": 0,
     "result": "B"},
    # Mexico 0-0 Poland
    {"name_a": "Mexico", "name_b": "Poland",
     "elo_a": 1860, "elo_b": 1796,
     "gf_avg_a": 1.5, "ga_avg_a": 1.2, "gf_avg_b": 1.3, "ga_avg_b": 1.1,
     "forma_a": 1.5, "forma_b": 1.4, "inj_a": 0, "inj_b": 0,
     "result": "D"},
    # Argentina 2-0 Mexico
    {"name_a": "Argentina", "name_b": "Mexico",
     "elo_a": 1974, "elo_b": 1860,
     "gf_avg_a": 2.1, "ga_avg_a": 0.6, "gf_avg_b": 1.5, "ga_avg_b": 1.2,
     "forma_a": 2.4, "forma_b": 1.5, "inj_a": 0, "inj_b": 0,
     "result": "A"},
    # Poland 2-0 Saudi Arabia
    {"name_a": "Poland",  "name_b": "Saudi Arabia",
     "elo_a": 1796, "elo_b": 1651,
     "gf_avg_a": 1.3, "ga_avg_a": 1.1, "gf_avg_b": 1.1, "ga_avg_b": 1.2,
     "forma_a": 1.4, "forma_b": 1.3, "inj_a": 0, "inj_b": 0,
     "result": "A"},
    # Argentina 2-0 Poland
    {"name_a": "Argentina", "name_b": "Poland",
     "elo_a": 1974, "elo_b": 1796,
     "gf_avg_a": 2.1, "ga_avg_a": 0.6, "gf_avg_b": 1.3, "ga_avg_b": 1.1,
     "forma_a": 2.4, "forma_b": 1.4, "inj_a": 0, "inj_b": 0,
     "result": "A"},
    # Saudi Arabia 1-2 Mexico
    {"name_a": "Saudi Arabia", "name_b": "Mexico",
     "elo_a": 1651, "elo_b": 1860,
     "gf_avg_a": 1.1, "ga_avg_a": 1.2, "gf_avg_b": 1.5, "ga_avg_b": 1.2,
     "forma_a": 1.3, "forma_b": 1.5, "inj_a": 0, "inj_b": 0,
     "result": "B"},

    # ── Group E ──
    # Japan 2-1 Germany  *** GIANT UPSET ***
    {"name_a": "Japan",   "name_b": "Germany",
     "elo_a": 1771, "elo_b": 1876,
     "gf_avg_a": 1.5, "ga_avg_a": 0.9, "gf_avg_b": 1.9, "ga_avg_b": 1.1,
     "forma_a": 1.7, "forma_b": 1.8, "inj_a": 0, "inj_b": 0,
     "result": "A"},
    # Spain 7-0 Costa Rica
    {"name_a": "Spain",      "name_b": "Costa Rica",
     "elo_a": 1886, "elo_b": 1672,
     "gf_avg_a": 1.8, "ga_avg_a": 0.7, "gf_avg_b": 1.0, "ga_avg_b": 1.2,
     "forma_a": 2.0, "forma_b": 1.2, "inj_a": 0, "inj_b": 0,
     "result": "A"},
    # Japan 0-1 Costa Rica
    {"name_a": "Japan",  "name_b": "Costa Rica",
     "elo_a": 1771, "elo_b": 1672,
     "gf_avg_a": 1.5, "ga_avg_a": 0.9, "gf_avg_b": 1.0, "ga_avg_b": 1.2,
     "forma_a": 1.7, "forma_b": 1.2, "inj_a": 0, "inj_b": 0,
     "result": "B"},
    # Germany 1-1 Spain
    {"name_a": "Germany", "name_b": "Spain",
     "elo_a": 1876, "elo_b": 1886,
     "gf_avg_a": 1.9, "ga_avg_a": 1.1, "gf_avg_b": 1.8, "ga_avg_b": 0.7,
     "forma_a": 1.8, "forma_b": 2.0, "inj_a": 0, "inj_b": 0,
     "result": "D"},
    # Japan 2-1 Spain  *** UPSET ***
    {"name_a": "Japan",  "name_b": "Spain",
     "elo_a": 1771, "elo_b": 1886,
     "gf_avg_a": 1.5, "ga_avg_a": 0.9, "gf_avg_b": 1.8, "ga_avg_b": 0.7,
     "forma_a": 1.7, "forma_b": 2.0, "inj_a": 0, "inj_b": 0,
     "result": "A"},
    # Costa Rica 2-4 Germany
    {"name_a": "Costa Rica", "name_b": "Germany",
     "elo_a": 1672, "elo_b": 1876,
     "gf_avg_a": 1.0, "ga_avg_a": 1.2, "gf_avg_b": 1.9, "ga_avg_b": 1.1,
     "forma_a": 1.2, "forma_b": 1.8, "inj_a": 0, "inj_b": 0,
     "result": "B"},

    # ── Group F ──
    # Belgium 1-0 Canada
    {"name_a": "Belgium", "name_b": "Canada",
     "elo_a": 1934, "elo_b": 1756,
     "gf_avg_a": 1.6, "ga_avg_a": 0.9, "gf_avg_b": 1.3, "ga_avg_b": 1.1,
     "forma_a": 1.7, "forma_b": 1.4, "inj_a": 1, "inj_b": 0,
     "result": "A"},
    # Morocco 0-0 Croatia
    {"name_a": "Morocco", "name_b": "Croatia",
     "elo_a": 1712, "elo_b": 1862,
     "gf_avg_a": 1.3, "ga_avg_a": 0.7, "gf_avg_b": 1.4, "ga_avg_b": 0.9,
     "forma_a": 1.7, "forma_b": 1.7, "inj_a": 0, "inj_b": 0,
     "result": "D"},
    # Belgium 0-2 Morocco  *** UPSET ***
    {"name_a": "Belgium", "name_b": "Morocco",
     "elo_a": 1934, "elo_b": 1712,
     "gf_avg_a": 1.6, "ga_avg_a": 0.9, "gf_avg_b": 1.3, "ga_avg_b": 0.7,
     "forma_a": 1.7, "forma_b": 1.7, "inj_a": 1, "inj_b": 0,
     "result": "B"},
    # Croatia 4-1 Canada
    {"name_a": "Croatia", "name_b": "Canada",
     "elo_a": 1862, "elo_b": 1756,
     "gf_avg_a": 1.4, "ga_avg_a": 0.9, "gf_avg_b": 1.3, "ga_avg_b": 1.1,
     "forma_a": 1.7, "forma_b": 1.4, "inj_a": 0, "inj_b": 0,
     "result": "A"},
    # Morocco 2-1 Canada
    {"name_a": "Morocco", "name_b": "Canada",
     "elo_a": 1712, "elo_b": 1756,
     "gf_avg_a": 1.3, "ga_avg_a": 0.7, "gf_avg_b": 1.3, "ga_avg_b": 1.1,
     "forma_a": 1.7, "forma_b": 1.4, "inj_a": 0, "inj_b": 0,
     "result": "A"},
    # Croatia 0-0 Belgium
    {"name_a": "Croatia", "name_b": "Belgium",
     "elo_a": 1862, "elo_b": 1934,
     "gf_avg_a": 1.4, "ga_avg_a": 0.9, "gf_avg_b": 1.6, "ga_avg_b": 0.9,
     "forma_a": 1.7, "forma_b": 1.7, "inj_a": 0, "inj_b": 1,
     "result": "D"},

    # ── Group G ──
    # Brazil 2-0 Serbia
    {"name_a": "Brazil",  "name_b": "Serbia",
     "elo_a": 1962, "elo_b": 1821,
     "gf_avg_a": 1.8, "ga_avg_a": 0.7, "gf_avg_b": 1.4, "ga_avg_b": 1.1,
     "forma_a": 2.0, "forma_b": 1.5, "inj_a": 0, "inj_b": 0,
     "result": "A"},
    # Switzerland 1-0 Cameroon
    {"name_a": "Switzerland", "name_b": "Cameroon",
     "elo_a": 1860, "elo_b": 1716,
     "gf_avg_a": 1.5, "ga_avg_a": 0.8, "gf_avg_b": 1.2, "ga_avg_b": 1.2,
     "forma_a": 1.7, "forma_b": 1.4, "inj_a": 0, "inj_b": 0,
     "result": "A"},
    # Brazil 1-0 Switzerland
    {"name_a": "Brazil",      "name_b": "Switzerland",
     "elo_a": 1962, "elo_b": 1860,
     "gf_avg_a": 1.8, "ga_avg_a": 0.7, "gf_avg_b": 1.5, "ga_avg_b": 0.8,
     "forma_a": 2.0, "forma_b": 1.7, "inj_a": 0, "inj_b": 0,
     "result": "A"},
    # Cameroon 3-3 Serbia
    {"name_a": "Cameroon", "name_b": "Serbia",
     "elo_a": 1716, "elo_b": 1821,
     "gf_avg_a": 1.2, "ga_avg_a": 1.2, "gf_avg_b": 1.4, "ga_avg_b": 1.1,
     "forma_a": 1.4, "forma_b": 1.5, "inj_a": 0, "inj_b": 0,
     "result": "D"},
    # Brazil 0-1 Cameroon  *** UPSET — Brazil fielded reserves ***
    {"name_a": "Brazil",   "name_b": "Cameroon",
     "elo_a": 1962, "elo_b": 1716,
     "gf_avg_a": 1.8, "ga_avg_a": 0.7, "gf_avg_b": 1.2, "ga_avg_b": 1.2,
     "forma_a": 2.0, "forma_b": 1.4, "inj_a": 0, "inj_b": 0,
     "result": "B"},
    # Serbia 2-3 Switzerland
    {"name_a": "Serbia",      "name_b": "Switzerland",
     "elo_a": 1821, "elo_b": 1860,
     "gf_avg_a": 1.4, "ga_avg_a": 1.1, "gf_avg_b": 1.5, "ga_avg_b": 0.8,
     "forma_a": 1.5, "forma_b": 1.7, "inj_a": 0, "inj_b": 0,
     "result": "B"},
]

assert len(WC2022_MATCHES) >= 30, f"Need at least 30 matches, got {len(WC2022_MATCHES)}"


# ─────────────────────────────────────────────
#  Helper: build pseudo team dicts from match row
# ─────────────────────────────────────────────
def _make_teams(m: dict) -> tuple[dict, dict]:
    ta = {
        "name":     m["name_a"],
        "ELO":      m["elo_a"],
        "GF_AVG":   m["gf_avg_a"],
        "GA_AVG":   m["ga_avg_a"],
        "FORMA":    m["forma_a"],
        "INJURIES": m["inj_a"],
    }
    tb = {
        "name":     m["name_b"],
        "ELO":      m["elo_b"],
        "GF_AVG":   m["gf_avg_b"],
        "GA_AVG":   m["ga_avg_b"],
        "FORMA":    m["forma_b"],
        "INJURIES": m["inj_b"],
    }
    return ta, tb


# ─────────────────────────────────────────────
#  Log-loss
# ─────────────────────────────────────────────
def _log_loss_single(probs: tuple, result: str) -> float:
    """Log-loss for a single match. probs = (p_a, p_d, p_b)."""
    eps = 1e-9
    pa, pd, pb = probs
    if result == "A":
        return -math.log(max(eps, pa))
    elif result == "D":
        return -math.log(max(eps, pd))
    else:
        return -math.log(max(eps, pb))


def compute_log_loss(predictions: list, results: list) -> float:
    """Mean log-loss over all matches."""
    total = sum(_log_loss_single(p, r) for p, r in zip(predictions, results))
    return total / len(predictions)


# ─────────────────────────────────────────────
#  Per-model accuracy (correct outcome prediction)
# ─────────────────────────────────────────────
def _predict_outcome(probs: tuple) -> str:
    pa, pd, pb = probs
    if pa >= pd and pa >= pb:
        return "A"
    elif pb >= pa and pb >= pd:
        return "B"
    else:
        return "D"


def accuracy(predictions: list, results: list) -> float:
    correct = sum(1 for p, r in zip(predictions, results) if _predict_outcome(p) == r)
    return correct / len(results)




# ─────────────────────────────────────────────
#  Calibration: individual model evaluation
# ─────────────────────────────────────────────
def evaluate_individual_models():
    """
    Evaluate each model on WC 2022 matches.
    For the ML model, we retrain WITHOUT WC 2022 data to avoid data leakage —
    the calibration set must be out-of-sample for all models.
    """
    results_list = []
    elo_preds, poi_preds, ml_preds = [], [], []

    # Train an ML model excluding WC 2022 data to prevent leakage
    ml_predictor_oos = None
    try:
        from src.models.ml_predictor import MLPredictor, TRAINING_DATA, RESULT_MAP, _make_features
        import numpy as np

        # WC 2022 team names used in calibration
        wc2022_teams = {m["name_a"] for m in WC2022_MATCHES} | {m["name_b"] for m in WC2022_MATCHES}

        # Filter training data: keep only non-WC2022 rows
        # WC 2022 rows are approximately indices 0-45 in TRAINING_DATA;
        # but safer to filter by checking for exact ELO+result combos
        # that appear in WC2022_MATCHES. Use a simpler heuristic:
        # WC 2022 matches have comment markers. Since we can't read comments,
        # exclude the first ~46 rows (WC 2022 group + knockout entries)
        # and keep WC 2018 + EURO 2020 + EURO 2024 only.
        # More robust: retrain excluding all rows where (elo_a, elo_b, result)
        # match a WC2022_MATCHES entry.
        wc2022_sigs = set()
        for m in WC2022_MATCHES:
            wc2022_sigs.add((m["elo_a"], m["elo_b"], m["result"]))

        oos_data = []
        for row in TRAINING_DATA:
            elo_a, elo_b = row[0], row[1]
            result = row[9]
            if (elo_a, elo_b, result) not in wc2022_sigs:
                oos_data.append(row)

        if len(oos_data) >= 20:
            ml_oos = MLPredictor()
            X = [_make_features(row) for row in oos_data]
            y = [RESULT_MAP[row[9]] for row in oos_data]
            X = np.array(X, dtype=float)
            y = np.array(y, dtype=int)
            ml_oos.rf.fit(X, y)
            ml_oos.gb.fit(X, y)
            ml_oos._fitted = True
            ml_predictor_oos = ml_oos
            print(f"  ML model retrained without WC 2022 data ({len(oos_data)} rows, excluded {len(TRAINING_DATA)-len(oos_data)})")
        else:
            print(f"  ⚠ Too few non-WC2022 training rows ({len(oos_data)}), using full ML model (leakage warning)")
            ml_predictor_oos = _get_ml_predictor()
    except Exception as e:
        print(f"  ⚠ ML out-of-sample training failed: {e}")
        ml_predictor_oos = _get_ml_predictor()

    for m in WC2022_MATCHES:
        ta, tb = _make_teams(m)
        results_list.append(m["result"])

        ep = elo_model(ta, tb)           # (p_a, p_d, p_b)
        pp = poisson_model(ta, tb)

        if ml_predictor_oos:
            mp = ml_predictor_oos.predict(ta, tb)
        else:
            mp = (0.33, 0.34, 0.33)

        elo_preds.append(ep)
        poi_preds.append(pp)
        ml_preds.append(mp)

    elo_ll  = compute_log_loss(elo_preds,  results_list)
    poi_ll  = compute_log_loss(poi_preds,  results_list)
    ml_ll   = compute_log_loss(ml_preds,   results_list)

    elo_acc = accuracy(elo_preds,  results_list)
    poi_acc = accuracy(poi_preds,  results_list)
    ml_acc  = accuracy(ml_preds,   results_list)

    return {
        "elo":     {"log_loss": elo_ll, "accuracy": elo_acc, "preds": elo_preds},
        "poisson": {"log_loss": poi_ll, "accuracy": poi_acc, "preds": poi_preds},
        "ml":      {"log_loss": ml_ll,  "accuracy": ml_acc,  "preds": ml_preds},
        "results": results_list,
    }


# ─────────────────────────────────────────────
#  Grid search over weights
# ─────────────────────────────────────────────
def blend_probs(preds_dict: dict, weights: dict) -> list:
    """
    Blend predictions from multiple models using given weights.
    preds_dict: {"elo": [...], "poisson": [...], "ml": [...]}
    weights: {"elo": float, "poisson": float, "ml": float}  — will be normalized
    """
    total_w = sum(weights.values())
    n = len(preds_dict["elo"])
    blended = []
    for i in range(n):
        pa = sum(weights[k] * preds_dict[k][i][0] for k in weights) / total_w
        pd = sum(weights[k] * preds_dict[k][i][1] for k in weights) / total_w
        pb = sum(weights[k] * preds_dict[k][i][2] for k in weights) / total_w
        # Renormalize
        s = pa + pd + pb
        blended.append((pa / s, pd / s, pb / s))
    return blended


def grid_search_weights(model_evals: dict, step: float = 0.05) -> tuple[dict, float, float]:
    """
    Grid search over (w_elo, w_poisson, w_ml) — all sum to 1.
    Returns (best_weights, best_log_loss, best_acc).
    """
    results_list = model_evals["results"]
    preds_dict = {
        "elo":     model_evals["elo"]["preds"],
        "poisson": model_evals["poisson"]["preds"],
        "ml":      model_evals["ml"]["preds"],
    }

    best_ll   = float("inf")
    best_w    = None
    best_acc  = 0.0
    candidates = []

    # Generate weight triplets that sum to 1.0 (within floating-point tolerance)
    steps_n = round(1.0 / step)
    for i in range(steps_n + 1):
        for j in range(steps_n + 1 - i):
            k = steps_n - i - j
            if k < 0:
                continue
            we = i / steps_n
            wp = j / steps_n
            wml = k / steps_n
            if abs(we + wp + wml - 1.0) > 1e-9:
                continue
            candidates.append({"elo": we, "poisson": wp, "ml": wml})

    for w in candidates:
        blended = blend_probs(preds_dict, w)
        ll = compute_log_loss(blended, results_list)
        acc = accuracy(blended, results_list)
        if ll < best_ll:
            best_ll = ll
            best_w = dict(w)
            best_acc = acc

    return best_w, best_ll, best_acc


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  WC 2022 Model Calibration — stacked_predictor")
    print("=" * 60)
    print(f"\n  Matches used: {len(WC2022_MATCHES)}")

    # Count upsets in the dataset
    upsets = [m for m in WC2022_MATCHES
              if (m["result"] == "B" and m["elo_a"] > m["elo_b"] + 100)
              or (m["result"] == "A" and m["elo_b"] > m["elo_a"] + 100)]
    print(f"  Notable upsets included: {len(upsets)}")
    for u in upsets:
        print(f"    {u['name_a']} (ELO {u['elo_a']}) vs "
              f"{u['name_b']} (ELO {u['elo_b']}) → {u['result']} won")

    print("\n--- Individual Model Performance ---")
    model_evals = evaluate_individual_models()

    for model_name in ["elo", "poisson", "ml"]:
        stats = model_evals[model_name]
        print(f"  {model_name.capitalize():8s}  "
              f"log-loss={stats['log_loss']:.4f}  "
              f"accuracy={stats['accuracy']*100:.1f}%")

    print("\n--- Grid Search (step=0.05, no H2H) ---")
    best_w, best_ll, best_acc = grid_search_weights(model_evals, step=0.05)

    print(f"\n  Optimal weights (3-model blend):")
    print(f"    ELO:     {best_w['elo']:.2f}")
    print(f"    Poisson: {best_w['poisson']:.2f}")
    print(f"    ML:      {best_w['ml']:.2f}")
    print(f"  → log-loss = {best_ll:.4f}  accuracy = {best_acc*100:.1f}%")

    # Also show current DEFAULT_WEIGHTS performance for comparison
    from src.models.stacked_predictor import DEFAULT_WEIGHTS
    dw_3 = {k: DEFAULT_WEIGHTS[k] for k in ["elo", "poisson", "ml"]}
    preds_dict = {
        "elo":     model_evals["elo"]["preds"],
        "poisson": model_evals["poisson"]["preds"],
        "ml":      model_evals["ml"]["preds"],
    }
    blended_default = blend_probs(preds_dict, dw_3)
    ll_default = compute_log_loss(blended_default, model_evals["results"])
    acc_default = accuracy(blended_default, model_evals["results"])

    print(f"\n--- Current DEFAULT_WEIGHTS Performance ---")
    print(f"  ELO={DEFAULT_WEIGHTS['elo']}  "
          f"Poisson={DEFAULT_WEIGHTS['poisson']}  "
          f"ML={DEFAULT_WEIGHTS['ml']}")
    print(f"  → log-loss = {ll_default:.4f}  accuracy = {acc_default*100:.1f}%")

    print()
    print("=" * 60)

    # Finer grid search at step=0.02
    print("\n--- Fine Grid Search (step=0.02) ---")
    fine_w, fine_ll, fine_acc = grid_search_weights(model_evals, step=0.02)
    print(f"  Optimal weights (3-model blend, fine grid):")
    print(f"    ELO:     {fine_w['elo']:.2f}")
    print(f"    Poisson: {fine_w['poisson']:.2f}")
    print(f"    ML:      {fine_w['ml']:.2f}")
    print(f"  → log-loss = {fine_ll:.4f}  accuracy = {fine_acc*100:.1f}%")
    print()


if __name__ == "__main__":
    main()
