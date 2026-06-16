"""
Guessing Game Points Optimizer for WC 2026.

Scoring rules:
  5 pts — exact scoreline
  3 pts — correct outcome (win or draw) + correct goal difference
  2 pts — correct outcome only
  0 pts — wrong outcome

Key insight for draws: any draw prediction earns 3 pts for *any* draw result
because the goal difference is always 0 → a draw prediction clusters 3-pt
probability across all possible draws, making it often higher EV than the
bare highest-probability scoreline.

Usage:
    from src.models.points_optimizer import find_optimal_pick
    result = find_optimal_pick(lam_a=1.3, lam_b=0.9)
"""

import math
from src.models.stacked_predictor import DIXON_COLES_RHO, _poisson_pmf

# ── Calibration constants ─────────────────────────────────────────────────────
# Calibrated 2026-06-16: 8 draws / 16 games (50% observed vs 29.8% model mean).
# Lambda scaling alone can't reach 50% (binary search hits lower bound at 0.5,
# achieves only 44.7%). Using 0.82 gives ~36% draw rate in the scoreline matrix
# — a meaningful boost toward reality without making 0-0 dominate every pick.
# The DRAW_BOOST in stacked_predictor.py handles the 1X2 probability side.
# Lambda scale: keep at 1.0. Reducing it shifts draw picks from 1-1 to 0-0,
# which is wrong — 1-1 is by far the most common draw in WC 2026 (5/8 draws).
# The DRAW_BOOST in stacked_predictor.py handles the 1X2 probability calibration.
WC2026_LAMBDA_SCALE: float = 1.0

# rho=-0.30 lifts exact low-score draws (0-0, 1-1) relative to adjacent cells.
# Stronger than the default -0.15, calibrated on WC 2026 where 1-1 dominates.
WC2026_DC_RHO: float = -0.30


# ── Core functions ────────────────────────────────────────────────────────────

def dc_scoreline_matrix(
    lam_a: float,
    lam_b: float,
    max_goals: int = 8,
    rho: float = WC2026_DC_RHO,
) -> dict:
    """
    Build a full Dixon-Coles-corrected scoreline probability matrix.

    Returns a dict keyed by (i, j) tuples where i = goals_a, j = goals_b.
    All cells are normalized to sum to 1.0.

    max_goals=8 (vs 6 in the display scoreline_matrix) avoids truncating tail
    events that accumulate in the 3-pt tier for the optimizer.
    """
    matrix = {}
    total = 0.0

    for i in range(max_goals + 1):
        pa_i = _poisson_pmf(lam_a, i)
        for j in range(max_goals + 1):
            p = pa_i * _poisson_pmf(lam_b, j)

            # Dixon-Coles tau correction for low-scoring outcomes
            if i == 0 and j == 0:
                tau = 1.0 - lam_a * lam_b * rho
            elif i == 1 and j == 0:
                tau = 1.0 + lam_b * rho
            elif i == 0 and j == 1:
                tau = 1.0 + lam_a * rho
            elif i == 1 and j == 1:
                tau = 1.0 - rho
            else:
                tau = 1.0

            tau = max(0.01, tau)
            p *= tau
            matrix[(i, j)] = p
            total += p

    if total <= 0:
        total = 1.0
    for key in matrix:
        matrix[key] /= total

    return matrix


def _outcome(i: int, j: int) -> int:
    """Return +1 if i > j (A wins), 0 if draw, -1 if j > i (B wins)."""
    if i > j:
        return 1
    elif i == j:
        return 0
    else:
        return -1


def expected_points(
    pred_a: int,
    pred_b: int,
    score_matrix: dict,
) -> float:
    """
    Compute expected points for predicting scoreline (pred_a, pred_b),
    given a scoreline probability matrix from dc_scoreline_matrix().

    Scoring:
      5 pts → exact (i == pred_a and j == pred_b)
      3 pts → same outcome AND same goal difference (but different scoreline)
      2 pts → same outcome only
      0 pts → wrong outcome
    """
    pred_outcome = _outcome(pred_a, pred_b)
    pred_gd = pred_a - pred_b
    pts = 0.0

    for (i, j), p in score_matrix.items():
        if i == pred_a and j == pred_b:
            pts += 5 * p
        elif _outcome(i, j) == pred_outcome:
            if (i - j) == pred_gd:
                pts += 3 * p
            else:
                pts += 2 * p
        # else: 0 pts, wrong outcome — no contribution

    return pts


def find_optimal_pick(
    lam_a: float,
    lam_b: float,
    rho: float = WC2026_DC_RHO,
    lambda_scale: float = WC2026_LAMBDA_SCALE,
    max_goals: int = 8,
) -> dict:
    """
    Find the scoreline prediction that maximizes expected points in the
    guessing game (5/3/2/0 scoring).

    Args:
        lam_a: Expected goals for team A (from expected_goals())
        lam_b: Expected goals for team B
        rho: Dixon-Coles correlation parameter
        lambda_scale: Multiplier applied to both lambdas (< 1.0 boosts draws)
        max_goals: Max goals per team to consider (8 recommended for optimizer)

    Returns a dict with:
        max_ev_pick    — scoreline with highest expected points
        max_prob_pick  — scoreline with highest raw probability (current model behaviour)
        top_candidates — top 5 picks by expected points
        pick_delta_ev  — EV gain of optimal pick vs max-prob pick
        draw_probability — total draw probability from the matrix
    """
    lam_a_adj = lam_a * lambda_scale
    lam_b_adj = lam_b * lambda_scale

    matrix = dc_scoreline_matrix(lam_a_adj, lam_b_adj, max_goals, rho)

    best_ev_pick = None
    best_ev = -1.0
    best_prob_pick = None
    best_prob = -1.0
    candidates = []

    for (a, b), prob in matrix.items():
        ev = expected_points(a, b, matrix)
        score = f"{a}-{b}"
        candidates.append({
            "score":        score,
            "gf_a":         a,
            "gf_b":         b,
            "expected_pts": round(ev, 4),
            "prob":         round(prob, 5),
        })
        if ev > best_ev:
            best_ev = ev
            best_ev_pick = {"score": score, "gf_a": a, "gf_b": b,
                            "expected_pts": round(ev, 4), "prob": round(prob, 5)}
        if prob > best_prob:
            best_prob = prob
            best_prob_pick = {"score": score, "gf_a": a, "gf_b": b,
                              "prob": round(prob, 5), "expected_pts": round(
                                  expected_points(a, b, matrix), 4)}

    # Sort by EV descending; top 5 for display
    candidates.sort(key=lambda x: -x["expected_pts"])
    top_candidates = candidates[:5]

    # Total draw probability
    draw_prob = sum(p for (i, j), p in matrix.items() if i == j)

    # EV delta between optimal and max-prob picks
    pick_delta_ev = round(
        best_ev_pick["expected_pts"] - best_prob_pick["expected_pts"], 4
    ) if best_ev_pick and best_prob_pick else 0.0

    return {
        "max_ev_pick":    best_ev_pick,
        "max_prob_pick":  best_prob_pick,
        "top_candidates": top_candidates,
        "pick_delta_ev":  pick_delta_ev,
        "draw_probability": round(draw_prob, 4),
        "lam_a":          round(lam_a_adj, 3),
        "lam_b":          round(lam_b_adj, 3),
    }
