"""
Draw Rate Calibration — WC 2026

Computes observed draw rate from actual WC 2026 results and compares it against
the model's predicted draw rate. Outputs recommended values for:

  DRAW_BOOST        → src/models/stacked_predictor.py
  WC2026_LAMBDA_SCALE → src/models/points_optimizer.py

Run after each matchday:
  python scripts/calibrate_draw_rate.py

Hold off on updating constants until ≥12 completed matches are available.
"""

import sys
import os
import csv
import math

# Ensure project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.stacked_predictor import DIXON_COLES_RHO, _poisson_pmf
from src.models.tournament import load_team_db
from src.models.stacked_predictor import expected_goals

WC2026_START = "2026-06-11"
CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "intl_results.csv")


def poisson_draw_prob(lam_a: float, lam_b: float, max_goals: int = 9) -> float:
    """Compute draw probability from a Dixon-Coles Poisson model."""
    rho = DIXON_COLES_RHO
    p_draw = 0.0
    total = 0.0
    for i in range(max_goals + 1):
        pa_i = _poisson_pmf(lam_a, i)
        for j in range(max_goals + 1):
            p = pa_i * _poisson_pmf(lam_b, j)
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
            total += p
            if i == j:
                p_draw += p
    return p_draw / total if total > 0 else 0.0


def scaled_draw_prob(lam_a: float, lam_b: float, scale: float) -> float:
    return poisson_draw_prob(lam_a * scale, lam_b * scale)


def find_lambda_scale(lam_pairs: list, target_draw_rate: float) -> float:
    """
    Binary-search for a lambda_scale s such that the mean draw probability
    across all matches (with lambdas multiplied by s) equals target_draw_rate.
    """
    lo, hi = 0.50, 1.20
    for _ in range(40):
        mid = (lo + hi) / 2
        mean_draw = sum(scaled_draw_prob(a, b, mid) for a, b in lam_pairs) / len(lam_pairs)
        if mean_draw < target_draw_rate:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def main():
    # ── Load WC 2026 completed matches from CSV ────────────────────────────────
    matches = []
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("tournament", "") != "FIFA World Cup":
                continue
            if row.get("date", "") < WC2026_START:
                continue
            try:
                hs = int(row["home_score"])
                as_ = int(row["away_score"])
            except (ValueError, KeyError):
                continue  # skip rows with missing/NA scores
            matches.append({
                "home": row["home_team"],
                "away": row["away_team"],
                "home_score": hs,
                "away_score": as_,
                "date": row["date"],
            })

    if not matches:
        print("No completed WC 2026 matches found in intl_results.csv.")
        print(f"Checked: {CSV_PATH}")
        return

    n = len(matches)
    n_draws = sum(1 for m in matches if m["home_score"] == m["away_score"])
    observed_draw_rate = n_draws / n

    print(f"\n{'─'*55}")
    print(f"  WC 2026 Completed Matches : {n}")
    print(f"  Draws observed            : {n_draws}")
    print(f"  Observed draw rate        : {observed_draw_rate:.1%}")
    print(f"{'─'*55}")

    if n < 12:
        print(f"\n  ⚠  Only {n} matches — wait until ≥12 before updating constants.")
        print(f"     Current recommendation: keep DRAW_BOOST = 1.0 (preliminary data only)\n")

    # ── Compute model draw probability for each match ──────────────────────────
    team_db = load_team_db()

    def team_dict(name):
        t = dict(team_db.get(name, {
            "ELO": 1650, "FORMA": 1.2, "GF_AVG": 1.0, "GA_AVG": 1.2, "INJURIES": 0
        }))
        t["name"] = name
        return t

    model_draw_probs = []
    lam_pairs = []
    skipped = 0

    for m in matches:
        ta = team_dict(m["home"])
        tb = team_dict(m["away"])
        # Determine host nation
        home_team = None
        for hn in ("USA", "Mexico", "Canada"):
            if m["home"] == hn or m["away"] == hn:
                home_team = hn
                break
        try:
            lam_a, lam_b = expected_goals(ta, tb, home_team=home_team)
            dp = poisson_draw_prob(lam_a, lam_b)
            model_draw_probs.append(dp)
            lam_pairs.append((lam_a, lam_b))
        except Exception as e:
            skipped += 1

    if not model_draw_probs:
        print("Could not compute model draw probabilities (team data missing?).")
        return

    mean_model_draw = sum(model_draw_probs) / len(model_draw_probs)
    draw_boost = observed_draw_rate / mean_model_draw if mean_model_draw > 0 else 1.0
    draw_boost = max(0.70, min(2.50, draw_boost))   # safety clamp

    print(f"\n  Model mean draw rate      : {mean_model_draw:.1%}")
    print(f"  Ratio (observed/model)    : {draw_boost:.3f}")
    if skipped:
        print(f"  Skipped (missing data)    : {skipped}")

    # ── Find optimal lambda_scale ──────────────────────────────────────────────
    lambda_scale = find_lambda_scale(lam_pairs, observed_draw_rate)

    print(f"\n{'─'*55}")
    print(f"  Recommended constants:")
    print(f"{'─'*55}")
    print(f"\n  In src/models/stacked_predictor.py:")
    print(f"    DRAW_BOOST: float = {draw_boost:.2f}")
    print(f"\n  In src/models/points_optimizer.py:")
    print(f"    WC2026_LAMBDA_SCALE: float = {lambda_scale:.3f}")
    print(f"\n  Verification — scaled model draw rate: "
          f"{sum(scaled_draw_prob(a, b, lambda_scale) for a, b in lam_pairs) / len(lam_pairs):.1%}"
          f"  (target: {observed_draw_rate:.1%})")

    print(f"\n  Per-match breakdown:")
    print(f"  {'Date':<12} {'Home':<25} {'Away':<25} {'Score':<8} {'Model Draw%'}")
    print(f"  {'─'*12} {'─'*25} {'─'*25} {'─'*8} {'─'*12}")
    for m, dp in zip(matches, model_draw_probs):
        score = f"{m['home_score']}-{m['away_score']}"
        marker = " ← draw" if m["home_score"] == m["away_score"] else ""
        print(f"  {m['date']:<12} {m['home']:<25} {m['away']:<25} {score:<8} {dp:.1%}{marker}")

    print()


if __name__ == "__main__":
    main()
