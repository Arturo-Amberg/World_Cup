#!/usr/bin/env python3
"""
Porra Points Optimizer — calibrate WC2026_LAMBDA_SCALE and WC2026_DC_RHO
by maximizing total porra points on a historical WC group stage dataset.

Usage:
    python scripts/optimize_porra.py              # train on WC 2022 group stage
    python scripts/optimize_porra.py --year 2026  # train on WC 2026 played so far

The optimizer grid-searches (lambda_scale × rho) and picks the combo that
would have scored the most points if applied to each completed match.

After finding the best params it prints the recommended constants to paste
into src/models/points_optimizer.py.
"""

import sys
import os
import csv
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.tournament import load_team_db
from src.models.stacked_predictor import expected_goals
from src.models.points_optimizer import dc_scoreline_matrix, expected_points, find_optimal_pick

CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "intl_results.csv")

# WC group stage date ranges
WC_RANGES = {
    2022: ("2022-11-20", "2022-12-02"),
    2026: ("2026-06-11", "2026-12-31"),
}

HOST_NATIONS = {
    2022: {"Qatar"},
    2026: {"USA", "Mexico", "Canada"},
}

# Top-tier competitive tournaments to include in the "pro" training set
COMPETITIVE_TOURNAMENTS = {
    "FIFA World Cup",
    "FIFA World Cup qualification",
    "African Cup of Nations",
    "Gold Cup",
    "CONCACAF Series",
    "FIFA Series",
    "AFC Asian Cup qualification",
    "UEFA Nations League",
    "Copa America",
}

# Dataset name → internal name (mirrors github_stats_client)
NAME_MAP = {
    "United States":          "USA",
    "Bosnia and Herzegovina": "Bosnia & Herzegovina",
    "South Korea":            "South Korea",
    "Ivory Coast":            "Ivory Coast",
    "DR Congo":               "DR Congo",
    "Curaçao":                "Curaçao",
    "Cape Verde":             "Cape Verde",
}

CURRENT = {"lambda_scale": 1.03, "rho": -0.42}


def load_matches(year: int | None = None, since: str | None = None) -> list[dict]:
    """
    Load completed match data.
    - year: load a specific WC group stage (2022 or 2026)
    - since: load all competitive matches from this date onwards (YYYY-MM-DD)
    """
    matches = []
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            d = row.get("date", "")
            hs = row.get("home_score", "")
            as_ = row.get("away_score", "")
            if hs in ("", "NA") or as_ in ("", "NA"):
                continue

            if year is not None:
                start, end = WC_RANGES[year]
                if row.get("tournament") != "FIFA World Cup":
                    continue
                if not (start <= d <= end):
                    continue
            elif since is not None:
                if d < since:
                    continue
                if row.get("tournament", "") not in COMPETITIVE_TOURNAMENTS:
                    continue
            else:
                continue

            matches.append({
                "date":       d,
                "home":       NAME_MAP.get(row["home_team"], row["home_team"]),
                "away":       NAME_MAP.get(row["away_team"], row["away_team"]),
                "home_score": int(float(hs)),
                "away_score": int(float(as_)),
                "tournament": row.get("tournament", ""),
            })
    return matches


def score_pick(pick_a: int, pick_b: int, actual_a: int, actual_b: int) -> int:
    def outcome(x, y): return 1 if x > y else (0 if x == y else -1)
    if pick_a == actual_a and pick_b == actual_b:
        return 5
    if outcome(pick_a, pick_b) == outcome(actual_a, actual_b):
        if (pick_a - pick_b) == (actual_a - actual_b):
            return 3
        return 2
    return 0


def compute_lambdas(matches: list[dict], year: int | None = None) -> list[tuple | None]:
    team_db = load_team_db()
    hosts = HOST_NATIONS.get(year, {"USA", "Mexico", "Canada"})

    def team_dict(name):
        t = dict(team_db.get(name, {
            "ELO": 1650, "FORMA": 1.2, "GF_AVG": 1.0, "GA_AVG": 1.2, "INJURIES": 0
        }))
        t["name"] = name
        return t

    result = []
    for m in matches:
        home_team = None
        for h in hosts:
            if m["home"] == h or m["away"] == h:
                home_team = h
                break
        try:
            lam_a, lam_b = expected_goals(
                team_dict(m["home"]), team_dict(m["away"]), home_team=home_team
            )
            result.append((lam_a, lam_b))
        except Exception:
            result.append(None)
    return result


def total_points(matches, lambdas, lambda_scale: float, rho: float) -> int:
    total = 0
    for m, lam in zip(matches, lambdas):
        if lam is None:
            continue
        lam_a, lam_b = lam
        opt = find_optimal_pick(lam_a, lam_b, rho=rho, lambda_scale=lambda_scale)
        pick = opt["max_ev_pick"]
        total += score_pick(pick["gf_a"], pick["gf_b"],
                            m["home_score"], m["away_score"])
    return total


def grid_search(matches, lambdas, scale_range, rho_range, step_s=0.01, step_r=0.02):
    """Coarse grid search over lambda_scale × rho."""
    import numpy as np
    scales = [round(s, 3) for s in list(
        map(lambda x: round(x, 3),
            [scale_range[0] + i * step_s
             for i in range(int((scale_range[1] - scale_range[0]) / step_s) + 1)]))]
    rhos = [round(r, 3) for r in
            [rho_range[0] + i * step_r
             for i in range(int((rho_range[1] - rho_range[0]) / step_r) + 1)]]

    best_pts = -1
    best_params = CURRENT.copy()
    results = []

    for s in scales:
        for r in rhos:
            pts = total_points(matches, lambdas, s, r)
            results.append((pts, s, r))
            if pts > best_pts:
                best_pts = pts
                best_params = {"lambda_scale": s, "rho": r}

    return best_params, best_pts, sorted(results, reverse=True)[:10]


def fine_search(matches, lambdas, center_s, center_r, radius_s=0.03, radius_r=0.06):
    """Fine grid search around a center point."""
    step_s = 0.005
    step_r = 0.01
    s0, s1 = center_s - radius_s, center_s + radius_s
    r0, r1 = center_r - radius_r, center_r + radius_r
    return grid_search(matches, lambdas, (s0, s1), (r0, r1), step_s, step_r)


def print_match_breakdown(matches, lambdas, lambda_scale, rho, year):
    print(f"\n  Per-match breakdown (scale={lambda_scale}, rho={rho}):")
    header = f"  {'Date':<12} {'Home':<25} {'Away':<20} {'Actual':<8} {'Pick':<6} {'Pts'}"
    print(header)
    print("  " + "─" * 80)
    total = 0
    for m, lam in zip(matches, lambdas):
        if lam is None:
            continue
        lam_a, lam_b = lam
        opt = find_optimal_pick(lam_a, lam_b, rho=rho, lambda_scale=lambda_scale)
        pick = opt["max_ev_pick"]
        pts = score_pick(pick["gf_a"], pick["gf_b"],
                         m["home_score"], m["away_score"])
        total += pts
        actual_str = f"{m['home_score']}-{m['away_score']}"
        pick_str = pick["score"]
        marker = " ✓" if pts >= 2 else ""
        print(f"  {m['date']:<12} {m['home']:<25} {m['away']:<20} "
              f"{actual_str:<8} {pick_str:<6} {pts}{marker}")
    print(f"  {'─'*80}")
    print(f"  Total: {total} pts  ({total/len([l for l in lambdas if l]):.2f} pts/game)")
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int,
                        help="WC year to use as training data (2022 or 2026)")
    parser.add_argument("--since", default="2025-06-01",
                        help="Use all competitive matches since YYYY-MM-DD (default: 2025-06-01)")
    parser.add_argument("--show-current", action="store_true",
                        help="Show how current params score on this dataset")
    args = parser.parse_args()

    year = args.year

    if year is not None:
        if year not in WC_RANGES:
            print(f"Year {year} not supported. Choose from: {list(WC_RANGES.keys())}")
            sys.exit(1)
        label = f"WC {year} group stage"
        matches = load_matches(year=year)
    else:
        label = f"competitive matches since {args.since}"
        matches = load_matches(since=args.since)

    print(f"\n{'═'*60}")
    print(f"  Porra Optimizer — {label}")
    print(f"{'═'*60}\n")

    if not matches:
        print(f"No matches found.")
        sys.exit(1)

    valid = matches
    n_draws = sum(1 for m in valid if m["home_score"] == m["away_score"])
    print(f"  Loaded {len(valid)} completed matches ({n_draws} draws = {n_draws/len(valid):.0%})")

    print("  Computing model lambdas …")
    lambdas = compute_lambdas(valid, year)
    n_ok = sum(1 for l in lambdas if l is not None)
    print(f"  Lambdas computed for {n_ok}/{len(valid)} matches\n")

    # Current params baseline
    current_pts = total_points(valid, lambdas,
                               CURRENT["lambda_scale"], CURRENT["rho"])
    n_valid = sum(1 for l in lambdas if l is not None)
    print(f"  Current params  (scale={CURRENT['lambda_scale']}, rho={CURRENT['rho']}): "
          f"{current_pts} pts  ({current_pts/n_valid:.2f}/game)")

    # Coarse grid search
    print("\n  Running coarse grid search …")
    best, best_pts, top10 = grid_search(
        valid, lambdas,
        scale_range=(0.75, 1.05),
        rho_range=(-0.60, 0.10),
        step_s=0.02,
        step_r=0.05,
    )
    print(f"  Coarse best: scale={best['lambda_scale']}, rho={best['rho']} → {best_pts} pts")

    # Fine grid search around coarse best
    print("  Running fine grid search …")
    best2, best_pts2, top10_fine = fine_search(
        valid, lambdas, best["lambda_scale"], best["rho"]
    )
    print(f"  Fine best:   scale={best2['lambda_scale']}, rho={best2['rho']} → {best_pts2} pts")

    delta = best_pts2 - current_pts
    sign = "+" if delta >= 0 else ""
    print(f"\n  Improvement vs current: {sign}{delta} pts  ({sign}{delta/n_valid:.2f}/game)")

    print(f"\n  Top 5 (scale, rho) by total points:")
    for rank, (pts, s, r) in enumerate(top10_fine[:5], 1):
        marker = " ← current" if abs(s - CURRENT["lambda_scale"]) < 0.001 and abs(r - CURRENT["rho"]) < 0.001 else ""
        print(f"    {rank}. scale={s}, rho={r} → {pts} pts{marker}")

    # Per-match breakdown for best params
    if args.show_current:
        print(f"\n  Current params breakdown:")
        print_match_breakdown(valid, lambdas, CURRENT["lambda_scale"], CURRENT["rho"], year)

    if len(valid) <= 50:
        print(f"\n  Best params breakdown:")
        print_match_breakdown(valid, lambdas, best2["lambda_scale"], best2["rho"], year)

    print(f"\n{'─'*60}")
    print(f"  Recommended constants for src/models/points_optimizer.py:")
    print(f"{'─'*60}")
    print(f"\n    WC2026_LAMBDA_SCALE: float = {best2['lambda_scale']}")
    print(f"    WC2026_DC_RHO: float       = {best2['rho']}")
    print(f"\n  (current: scale={CURRENT['lambda_scale']}, rho={CURRENT['rho']})")
    print()


if __name__ == "__main__":
    main()
