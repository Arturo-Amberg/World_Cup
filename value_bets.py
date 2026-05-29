"""
Value Bet Finder — FIFA World Cup 2026
Compares model probabilities (Monte Carlo + stack_predict) against
Coolbet odds to surface positive-EV betting opportunities.

Usage:
    python3 value_bets.py                  # 5 000 MC sims
    python3 value_bets.py --sims 20000     # more precise
    python3 value_bets.py --min-edge 0.05  # stricter edge filter
    python3 value_bets.py --no-mc          # skip MC, only match odds
"""

import json
import sys
import math
import re
from collections import defaultdict
from pathlib import Path

# ── config ────────────────────────────────────────────────────────────────────
ODDS_FILE    = Path("data/coolbet/latest.json")
N_SIMS       = 5_000
MIN_EDGE     = 0.04    # model_prob - implied_prob ≥ 4 %
MIN_EV       = 0.06    # EV ≥ 6 % (ROI per unit staked)
MIN_ODDS     = 1.30    # ignore sub-1.30 (no juice)
MAX_ODDS     = 80.0    # ignore extreme longshots
RUN_MC       = True

# ── CLI flags ─────────────────────────────────────────────────────────────────
for arg in sys.argv[1:]:
    if arg.startswith("--sims="): N_SIMS   = int(arg.split("=")[1])
    if arg.startswith("--min-edge="): MIN_EDGE = float(arg.split("=")[1])
    if arg == "--no-mc": RUN_MC = False

# ── helpers ───────────────────────────────────────────────────────────────────

def implied(odds: float) -> float:
    """Decimal odds → raw implied probability (no margin removal)."""
    return 1.0 / odds

def remove_margin(probs: list[float]) -> list[float]:
    """Remove bookmaker margin from a set of implied probabilities."""
    total = sum(probs)
    return [p / total for p in probs]

def ev(model_prob: float, odds: float) -> float:
    """Expected value per unit staked."""
    return model_prob * odds - 1.0

def kelly_fraction(model_prob: float, odds: float) -> float:
    """Kelly Criterion: f = (p*b - q) / b where b is net odds (odds - 1)."""
    if odds <= 1: return 0
    b = odds - 1
    p = model_prob
    q = 1.0 - p
    f = (p * b - q) / b
    return max(0, f)

def fmt_pct(p: float) -> str:
    return f"{p*100:.1f}%"

def normalise_name(name: str) -> str:
    """Loose normalisation for team name matching."""
    n = name.strip().lower()
    n = re.sub(r"[^a-z0-9 ]", "", n)
    n = re.sub(r"\s+", " ", n)
    return n

# ── load odds ──────────────────────────────────────────────────────────────────

def load_odds() -> list[dict]:
    with open(ODDS_FILE, encoding="utf-8") as f:
        return json.load(f)

# ── load model ────────────────────────────────────────────────────────────────

def load_model():
    from tournament import load_team_db, load_groups, sim_group_stage
    from stacked_predictor import stack_predict
    return load_team_db, load_groups, sim_group_stage, stack_predict

# ── Monte Carlo ───────────────────────────────────────────────────────────────

def run_monte_carlo(n: int) -> dict:
    from tournament import load_team_db, load_groups, sim_full_tournament, sim_group_stage
    import time

    team_db = load_team_db()
    groups  = load_groups()

    # Accumulators
    wins      = defaultdict(int)
    finals    = defaultdict(int)
    semis     = defaultdict(int)
    quarters  = defaultdict(int)
    qual      = defaultdict(int)   # qualified from group stage
    grp_win   = defaultdict(int)   # group winner
    grp_2nd   = defaultdict(int)   # group second

    t0 = time.time()
    print(f"\nRunning {n:,} Monte Carlo simulations...")
    for i in range(n):
        r = sim_full_tournament(groups, team_db, silent=True)

        wins[r["champion"]] += 1
        finals[r["finalist"]] += 1
        finals[r["champion"]] += 1

        for t in r["semifinalists"]:  semis[t] += 1
        for t in r["quarterfinalists"]: quarters[t] += 1

        for grp, ranking in r["group_results"].items():
            if len(ranking) >= 1: grp_win[ranking[0]] += 1
            if len(ranking) >= 2: grp_2nd[ranking[1]] += 1
            # Top 2 always qualify
            if len(ranking) >= 1: qual[ranking[0]] += 1
            if len(ranking) >= 2: qual[ranking[1]] += 1

        # Best 8 third-placed teams also qualify (2026 format: 48 teams)
        all_thirds = []
        for grp, ranking in r["group_results"].items():
            if len(ranking) >= 3:
                all_thirds.append(ranking[2])
        # In sim_full_tournament the best thirds are already selected;
        # approximate here by counting teams that advanced past groups
        # Use the semifinalists/quarterfinalists to infer who qualified
        # More directly: count the 8 best thirds from group results
        # We need the group stats to rank thirds, but sim_full_tournament
        # doesn't return them. Instead, use the knockout participants:
        # everyone in the R32 qualified. The R32 has 32 teams = 24 (top 2) + 8 (best 3rds)
        qualified_set = set()
        for grp, ranking in r["group_results"].items():
            if len(ranking) >= 1: qualified_set.add(ranking[0])
            if len(ranking) >= 2: qualified_set.add(ranking[1])
        # The quarterfinalists are a subset of qualified teams;
        # but we need all 32 R32 participants. Since sim_full_tournament
        # returns group_results (rankings), we know the 3rd-placed teams
        # but not which 8 advanced. Use a heuristic: count all thirds
        # that appear in the quarterfinalists or beyond as qualified.
        # Better approach: patch sim_full_tournament to return best_thirds.
        # For now, mark all 8 best thirds using the returned data.
        for t in r.get("best_thirds", []):
            qual[t] += 1

        if (i + 1) % 1_000 == 0:
            print(f"  {i+1:,}/{n:,}  ({time.time()-t0:.0f}s)")

    print(f"Done in {time.time()-t0:.1f}s\n")

    return {
        "n":        n,
        "champion": {t: c / n for t, c in wins.items()},
        "finalist": {t: c / n for t, c in finals.items()},
        "semi":     {t: c / n for t, c in semis.items()},
        "quarter":  {t: c / n for t, c in quarters.items()},
        "grp_win":  {t: c / n for t, c in grp_win.items()},
        "grp_2nd":  {t: c / n for t, c in grp_2nd.items()},
        "qual":     {t: c / n for t, c in qual.items()},
    }

# ── match-level probabilities ─────────────────────────────────────────────────

def blended_match_prob(team_db: dict, team_a: str, team_b: str,
                       venue_name: str = None, home_team: str = None) -> dict:
    """
    Run stack_predict but down-weight ML for large ELO mismatches (>300 pts).
    The ML component can be noisy on extreme matchups.
    """
    from stacked_predictor import stack_predict, HOST_NATIONS
    import math

    # Auto-detect home team if either team is a host nation
    if home_team is None:
        if team_a in HOST_NATIONS:
            home_team = team_a
        elif team_b in HOST_NATIONS:
            home_team = team_b

    pred = stack_predict(team_db[team_a], team_db[team_b],
                         venue_name=venue_name, home_team=home_team)

    elo_a = team_db[team_a].get("ELO", 1700)
    elo_b = team_db[team_b].get("ELO", 1700)
    elo_diff = abs(elo_a - elo_b)

    # When ELO diff > 300, blend toward pure ELO+Poisson (skip ML noise)
    if elo_diff > 300:
        mb   = pred["model_breakdown"]
        elo  = mb["ELO"]
        poi  = mb["Poisson"]
        w    = min(1.0, (elo_diff - 300) / 400)   # 0→1 as diff goes 300→700
        # Conservative blend: lerp from full-stack toward ELO-only
        def lerp(a, b, t): return a * (1 - t) + b * t
        p_h = lerp(pred["p_win_a"], lerp(elo["win_a"], poi["win_a"], 0.5), w * 0.6)
        p_a = lerp(pred["p_win_b"], lerp(elo["win_b"], poi["win_b"], 0.5), w * 0.6)
        p_d = 1.0 - p_h - p_a
        return {"p_win_a": p_h, "p_draw": max(0.05, p_d), "p_win_b": p_a}

    return pred


def compute_match_probs(team_db: dict, groups: dict) -> dict:
    """
    Run stack_predict for every group-stage match.
    Returns {frozenset({home, away}): {home, away, p_h, p_d, p_a}}
    """
    results = {}
    for grp, teams in groups.items():
        t = list(teams)
        for i in range(len(t)):
            for j in range(i + 1, len(t)):
                a, b = t[i], t[j]
                if a not in team_db or b not in team_db:
                    continue
                try:
                    pred = blended_match_prob(team_db, a, b,
                                             venue_name=None, home_team=None)
                    results[frozenset([a, b])] = {
                        "home": a, "away": b,
                        "p_h":  pred["p_win_a"],
                        "p_d":  pred["p_draw"],
                        "p_a":  pred["p_win_b"],
                    }
                except Exception:
                    pass
    return results

# ── name matching ──────────────────────────────────────────────────────────────

# A few aliases to bridge Coolbet names → model names
ALIAS = {
    "czechia": "czech republic",
    "south korea": "south korea",
    "ivory coast": "ivory coast",
    "congo dr": "dr congo",
    "dr congo": "dr congo",
    "bosniaand herzegovina": "bosnia & herzegovina",
    "bosnia and herzegovina": "bosnia & herzegovina",
    "bosnia  herzegovina": "bosnia & herzegovina",
    "united states": "usa",
    "curacao": "curaçao",
    "cape verde": "cape verde",
    "north macedonia": "north macedonia",
    "curaçao": "curaçao",
}

def canonical(name: str) -> str:
    n = normalise_name(name)
    return ALIAS.get(n, n)

def best_match(name: str, model_keys: list[str]) -> str | None:
    """Find the best matching model team name for a Coolbet selection."""
    c = canonical(name)
    for k in model_keys:
        if canonical(k) == c:
            return k
    # Partial match fallback
    for k in model_keys:
        ck = canonical(k)
        if c in ck or ck in c:
            return k
    return None

# ── main analysis ─────────────────────────────────────────────────────────────

def find_value_bets(odds_data: list[dict], mc: dict | None, team_db: dict) -> list[dict]:
    """
    Cross-reference Coolbet odds with model probabilities.
    Returns list of value bet candidates, sorted by EV descending.
    """
    from tournament import load_groups
    groups = load_groups()

    # Build match-level probs
    match_probs = compute_match_probs(team_db, groups)
    model_teams = list(team_db.keys())

    value_bets = []

    # ── Helper: record a value bet if it meets thresholds ────────────────────
    def check(page, match_name, market, line, selection, odds_val, model_prob, category,
              fair_imp=None):
        """
        fair_imp: margin-free implied probability for this outcome.
        If not provided, falls back to raw implied (1/odds) — less accurate
        but needed for outrights where we don't have all outcomes' odds.
        """
        if model_prob is None or model_prob <= 0:
            return
        if not (MIN_ODDS <= odds_val <= MAX_ODDS):
            return
        raw_imp = implied(odds_val)
        imp = fair_imp if fair_imp is not None else raw_imp
        edge = model_prob - imp
        exp_val = ev(model_prob, odds_val)
        kelly = kelly_fraction(model_prob, odds_val)
        if edge >= MIN_EDGE and exp_val >= MIN_EV:
            value_bets.append({
                "category":   category,
                "page":       page,
                "match":      match_name,
                "market":     market,
                "line":       line,
                "selection":  selection,
                "odds":       odds_val,
                "model_prob": round(model_prob, 4),
                "implied_prob": round(imp, 4),
                "edge":       round(edge, 4),
                "ev":         round(exp_val, 4),
                "kelly":      round(kelly, 4),
            })

    # ── 1. MATCH ODDS (1X2) ─────────────────────────────────────────────────
    # Group odds rows by match + market
    from itertools import groupby
    match_markets: dict[tuple, list] = defaultdict(list)
    for row in odds_data:
        if row["page"] == "matches" and row["market"] == "Match Result (1X2)":
            key = (row["match"], row["market"])
            match_markets[key].append(row)

    for (match_name, market), rows in match_markets.items():
        # Identify home/away from match name (format: "Home - Away")
        parts = match_name.split(" - ", 1)
        if len(parts) != 2:
            continue
        home_raw, away_raw = parts
        key = frozenset([home_raw, away_raw])

        # Find matching model pair
        matched_key = None
        for mkey in match_probs:
            mhome = match_probs[mkey]["home"]
            maway = match_probs[mkey]["away"]
            if (best_match(home_raw, model_teams) == mhome and
                best_match(away_raw, model_teams) == maway) or \
               (best_match(home_raw, model_teams) == maway and
                best_match(away_raw, model_teams) == mhome):
                matched_key = mkey
                break

        if matched_key is None:
            continue

        mp    = match_probs[matched_key]
        # Determine which model team is "home" in Coolbet
        flip = (best_match(home_raw, model_teams) == mp["away"])
        p_home = mp["p_a"] if flip else mp["p_h"]
        p_draw = mp["p_d"]
        p_away = mp["p_h"] if flip else mp["p_a"]

        # Remove bookmaker margin to get fair implied probabilities
        raw_implied_list = [implied(r["odds"]) for r in rows]
        fair_probs = remove_margin(raw_implied_list)
        # Map fair probs back to selections
        fair_by_sel = {rows[i]["selection"]: fair_probs[i] for i in range(len(rows))}

        for row in rows:
            sel  = row["selection"]
            odds = row["odds"]
            if sel == home_raw or sel == "1":
                mp_val = p_home
            elif sel == away_raw or sel == "2":
                mp_val = p_away
            elif sel.lower() in ("draw", "x"):
                mp_val = p_draw
            else:
                continue
            check("matches", match_name, market, "", sel, odds, mp_val, "Match 1X2",
                  fair_imp=fair_by_sel.get(sel))

    # ── 2. OUTRIGHT WINNER ──────────────────────────────────────────────────
    if mc:
        for row in odds_data:
            if row["page"] == "wc_specials" and "Outright Winner" in row["market"]:
                sel   = row["selection"]
                odds  = row["odds"]
                team  = best_match(sel, model_teams)
                if team is None:
                    continue
                mp_val = mc["champion"].get(team, 0)
                check("wc_specials", "WC 2026", "Outright Winner", "", sel, odds, mp_val, "Tournament Winner")

    # ── 3. GROUP WINNER ─────────────────────────────────────────────────────
    if mc:
        for row in odds_data:
            if row["page"] == "group_specials" and "Winner" in row["market"]:
                sel  = row["selection"]
                odds = row["odds"]
                team = best_match(sel, model_teams)
                if team is None:
                    continue
                mp_val = mc["grp_win"].get(team, 0)
                check("group_specials", row["match"], row["market"], "", sel, odds, mp_val, "Group Winner")

    # ── 4. GROUP QUALIFICATION (qualify from group stage) ───────────────────
    # market field = "{Team} to Qualify from Group Stage", selection = "Yes"/"No"
    if mc:
        qual_rows = [r for r in odds_data
                     if r["page"] == "group_specials"
                     and "Qualify" in r.get("market", "")]

        for row in qual_rows:
            market_name = row["market"]   # e.g. "Algeria to Qualify from Group Stage"
            sel   = row["selection"]      # "Yes" or "No"
            odds  = row["odds"]

            # Extract team name from market string
            team_raw = market_name.replace("to Qualify from Group Stage", "").strip()
            team = best_match(team_raw, model_teams)
            if team is None:
                continue

            qual_prob = mc["qual"].get(team, 0)

            if sel == "Yes":
                mp_val = qual_prob
                label  = f"{team} to Qualify"
            else:
                mp_val = 1.0 - qual_prob
                label  = f"{team} NOT to Qualify"

            check("group_specials", row["match"], market_name, "", label, odds, mp_val, "Group Qualification")

    # ── 5. TOP GOALS OVER/UNDER (Total Goals) ───────────────────────────────
    # Compare Coolbet's O/U 2.5 line with our Poisson model
    from stacked_predictor import stack_predict as sp
    groups = load_groups()

    match_ou: dict[str, dict] = {}  # match_name → {over_2.5, under_2.5, ...}
    for row in odds_data:
        if row["page"] == "matches" and "Total Goals" in row["market"]:
            line_str = str(row.get("line", "")).strip()
            try:
                line_val = float(line_str)
            except (ValueError, TypeError):
                continue
            mn = row["match"]
            if mn not in match_ou:
                match_ou[mn] = {}
            key = f"{row['selection']}_{line_val}"
            match_ou[mn][key] = row["odds"]

    def poisson_ou_probs(home_lam, away_lam, line):
        """P(over line) and P(under line) for total goals.
        Uses Dixon-Coles τ correction for low-score outcomes (consistent with match model).
        """
        from stacked_predictor import DIXON_COLES_RHO
        rho = DIXON_COLES_RHO
        max_g = 15
        p_over = 0.0
        p_total = 0.0
        for h in range(max_g + 1):
            ph = math.exp(-home_lam) * home_lam**h / math.factorial(h)
            for a in range(max_g + 1):
                pa = math.exp(-away_lam) * away_lam**a / math.factorial(a)
                p = ph * pa

                # Dixon-Coles τ adjustment for low-score outcomes
                if h == 0 and a == 0:
                    tau = 1.0 - home_lam * away_lam * rho
                elif h == 1 and a == 0:
                    tau = 1.0 + away_lam * rho
                elif h == 0 and a == 1:
                    tau = 1.0 + home_lam * rho
                elif h == 1 and a == 1:
                    tau = 1.0 - rho
                else:
                    tau = 1.0
                tau = max(0.01, tau)
                p *= tau

                p_total += p
                total = h + a
                if total > line:
                    p_over += p
        # Renormalize (τ adjustments may shift total slightly from 1.0)
        if p_total > 0:
            p_over /= p_total
        return p_over, 1.0 - p_over

    for match_name, ou_dict in match_ou.items():
        parts = match_name.split(" - ", 1)
        if len(parts) != 2:
            continue
        home_raw, away_raw = parts
        home_team = best_match(home_raw, model_teams)
        away_team = best_match(away_raw, model_teams)
        if home_team is None or away_team is None:
            continue
        if home_team not in team_db or away_team not in team_db:
            continue

        try:
            from stacked_predictor import expected_goals as eg
            home_lam, away_lam = eg(team_db[home_team], team_db[away_team])
        except Exception:
            pred = sp(team_db[home_team], team_db[away_team])
            home_lam = pred.get("exp_goals_a", 1.2)
            away_lam = pred.get("exp_goals_b", 1.0)

        for line in [0.5, 1.5, 2.5, 3.5, 4.5]:
            p_over, p_under = poisson_ou_probs(home_lam, away_lam, line)
            over_key  = f"Over_{line}"
            under_key = f"Under_{line}"

            # Compute fair implied if both sides of the line are available
            fair_over = fair_under = None
            if over_key in ou_dict and under_key in ou_dict:
                raw_o = implied(ou_dict[over_key])
                raw_u = implied(ou_dict[under_key])
                fair = remove_margin([raw_o, raw_u])
                fair_over, fair_under = fair[0], fair[1]

            if over_key in ou_dict:
                check("matches", match_name, f"Total Goals O/U", str(line),
                      f"Over {line}", ou_dict[over_key], p_over, "Total Goals",
                      fair_imp=fair_over)
            if under_key in ou_dict:
                check("matches", match_name, f"Total Goals O/U", str(line),
                      f"Under {line}", ou_dict[under_key], p_under, "Total Goals",
                      fair_imp=fair_under)

    # ── Sort by EV ───────────────────────────────────────────────────────────
    value_bets.sort(key=lambda x: -x["ev"])
    return value_bets


# ── display ───────────────────────────────────────────────────────────────────

def print_report(bets: list[dict]):
    # Group by category
    categories = {}
    for b in bets:
        cat = b["category"]
        categories.setdefault(cat, []).append(b)

    print("\n" + "="*80)
    print("  VALUE BETS REPORT — COOLBET WORLD CUP 2026")
    print(f"  min edge: {MIN_EDGE*100:.0f}%  |  min EV: {MIN_EV*100:.0f}%  |  sims: {N_SIMS:,}")
    print("="*80)

    if not bets:
        print("\nNo value bets found with current thresholds.")
        return

    HEADER = f"  {'Selection':<30} {'Odds':>6}  {'Model%':>7}  {'Impl%':>7}  {'EV':>7}  {'Kelly':>7}"
    SEP    = "  " + "-"*76

    for cat, cat_bets in categories.items():
        print(f"\n{'─'*80}")
        print(f"  ★  {cat.upper()}  ({len(cat_bets)} bets)")
        print(SEP)
        print(HEADER)
        print(SEP)
        for b in cat_bets:
            sel = b["selection"][:32]
            print(f"  {sel:<32} {b['odds']:>6.2f}  {b['model_prob']*100:>6.1f}%  {b['implied_prob']*100:>6.1f}%  {b['ev']*100:>+6.1f}%  {b['kelly']*100:>6.1f}%")
            ctx = b["match"][:45] if b["match"].strip() else b["market"][:45]
            line_s = f" [{b['line']}]" if b["line"] else ""
            print(f"    ↳ {ctx}{line_s}")

    print(f"\n{'='*80}")
    print(f"  TOTAL: {len(bets)} value bets found")
    top = bets[:5]
    print(f"\n  TOP 5 BY KELLY FRACTION (STAKE SIZE):")
    bets_sorted_kelly = sorted(bets, key=lambda x: -x["kelly"])
    for i, b in enumerate(bets_sorted_kelly[:5], 1):
        star = "★★★" if b["kelly"] >= 0.05 else "★★" if b["kelly"] >= 0.02 else "★"
        print(f"  {i}. {star} {b['selection']} ({b['match'][:40]})")
        print(f"     Kelly {b['kelly']*100:.1f}% | Odds {b['odds']:.2f} | Model {b['model_prob']*100:.1f}% | EV {b['ev']*100:+.1f}%")
    print("="*80)

    # Save to file
    out = Path("data/coolbet/value_bets.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(bets, f, ensure_ascii=False, indent=2)
    print(f"\n  Saved → {out}")


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    print("Loading Coolbet odds...")
    odds_data = load_odds()
    print(f"  {len(odds_data)} odds rows loaded")

    from tournament import load_team_db
    team_db = load_team_db()

    mc = None
    if RUN_MC:
        mc_file = Path("data/mc_results.json")
        if mc_file.exists():
            print(f"Loading existing Monte Carlo results from {mc_file}...")
            try:
                with open(mc_file, encoding="utf-8") as f:
                    mc_raw = json.load(f)
                    # Convert table back to the expected dict format
                    mc = {
                        "champion": {r["team"]: r["champion_%"]/100 for r in mc_raw["table"]},
                        "finalist": {r["team"]: r["final_%"]/100 for r in mc_raw["table"]},
                        "semi":     {r["team"]: r["semi_%"]/100 for r in mc_raw["table"]},
                        "quarter":  {r["team"]: r["quarter_%"]/100 for r in mc_raw["table"]},
                        "qual":     {r["team"]: r.get("qual_%", 0)/100 for r in mc_raw["table"]},
                        "grp_win":  {}, 
                    }
                    print("  Loaded successfully.")
            except Exception as e:
                print(f"  Error loading MC file: {e}. Running fresh simulation...")
                mc = run_monte_carlo(N_SIMS)
        else:
            mc = run_monte_carlo(N_SIMS)

    print("\nSearching for value bets...")
    bets = find_value_bets(odds_data, mc, team_db)
    print_report(bets)
    return bets


if __name__ == "__main__":
    main()
