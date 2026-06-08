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
    from src.models.tournament import load_team_db, load_groups, sim_group_stage
    from src.models.stacked_predictor import stack_predict
    return load_team_db, load_groups, sim_group_stage, stack_predict

# ── Monte Carlo ───────────────────────────────────────────────────────────────

def run_monte_carlo(n: int) -> dict:
    from src.models.tournament import load_team_db, load_groups, sim_full_tournament, sim_group_stage
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
    from src.models.stacked_predictor import stack_predict, HOST_NATIONS
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
        def lerp(a, b, t): return a * (1 - t) + b * t
        # For extreme mismatches the ML model is poorly calibrated (trained on
        # balanced datasets, rarely sees 400+ ELO gaps).  Use an ELO-dominant
        # target (80 % ELO / 20 % Poisson) and a stronger correction weight so
        # that at w=0.5 (ELO diff ~500) we are already 75 % of the way there.
        elo_w = 0.8   # ELO weight in the target blend
        target_h = elo_w * elo["win_a"] + (1 - elo_w) * poi["win_a"]
        target_a = elo_w * elo["win_b"] + (1 - elo_w) * poi["win_b"]
        blend = min(1.0, w * 1.5)    # 0→1 as w goes 0→0.67 (ELO diff ~567)
        p_h = lerp(pred["p_win_a"], target_h, blend)
        p_a = lerp(pred["p_win_b"], target_a, blend)
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
    from src.models.tournament import load_groups
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
        # Sanity check: reject likely inverted/corrupted Coolbet lines.
        # A model prob > 60% with odds > 12 (implied < 8%) is almost certainly
        # a pricing error (e.g. France at 31 to finish above Iraq in group).
        if model_prob > 0.60 and odds_val > 12:
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
    from src.models.stacked_predictor import stack_predict as sp
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
        from src.models.stacked_predictor import DIXON_COLES_RHO
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
            from src.models.stacked_predictor import expected_goals as eg
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

    # ── 6. CORNERS OVER/UNDER ────────────────────────────────────────────────
    from src.models.stacked_predictor import corners_model

    match_corners: dict[str, dict] = {}
    for row in odds_data:
        if row["page"] == "matches" and "Corner" in row.get("market", ""):
            line_str = str(row.get("line", "")).strip()
            try:
                line_val = float(line_str)
            except (ValueError, TypeError):
                continue
            mn = row["match"]
            if mn not in match_corners:
                match_corners[mn] = {}
            key = f"{row['selection']}_{line_val}"
            match_corners[mn][key] = row["odds"]

    for match_name, c_dict in match_corners.items():
        parts = match_name.split(" - ", 1)
        if len(parts) != 2:
            continue
        home_raw, away_raw = parts
        home_team = best_match(home_raw, model_teams)
        away_team = best_match(away_raw, model_teams)
        if not home_team or not away_team:
            continue
        if home_team not in team_db or away_team not in team_db:
            continue

        try:
            lam_c_h, lam_c_a = corners_model(
                team_db[home_team], team_db[away_team], home_team=home_team
            )
        except Exception:
            continue

        lam_c_total = lam_c_h + lam_c_a

        for line in [7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5]:
            p_over, p_under = poisson_ou_probs(lam_c_h, lam_c_a, line)
            over_key  = f"Over_{line}"
            under_key = f"Under_{line}"

            fair_over = fair_under = None
            if over_key in c_dict and under_key in c_dict:
                raw_o = implied(c_dict[over_key])
                raw_u = implied(c_dict[under_key])
                fair  = remove_margin([raw_o, raw_u])
                fair_over, fair_under = fair[0], fair[1]

            if over_key in c_dict:
                check("matches", match_name, "Corners O/U", str(line),
                      f"Over {line} Corners", c_dict[over_key], p_over, "Corners",
                      fair_imp=fair_over)
            if under_key in c_dict:
                check("matches", match_name, "Corners O/U", str(line),
                      f"Under {line} Corners", c_dict[under_key], p_under, "Corners",
                      fair_imp=fair_under)

    # ── 7. YELLOW CARDS OVER/UNDER ───────────────────────────────────────────
    from src.models.stacked_predictor import cards_model

    match_cards: dict[str, dict] = {}
    for row in odds_data:
        if row["page"] == "matches" and any(
            kw in row.get("market", "") for kw in ("Yellow Card", "Booking", "Cards")
        ):
            line_str = str(row.get("line", "")).strip()
            try:
                line_val = float(line_str)
            except (ValueError, TypeError):
                continue
            mn = row["match"]
            if mn not in match_cards:
                match_cards[mn] = {}
            key = f"{row['selection']}_{line_val}"
            match_cards[mn][key] = row["odds"]

    for match_name, cards_dict in match_cards.items():
        parts = match_name.split(" - ", 1)
        if len(parts) != 2:
            continue
        home_raw, away_raw = parts
        home_team = best_match(home_raw, model_teams)
        away_team = best_match(away_raw, model_teams)
        if not home_team or not away_team:
            continue
        if home_team not in team_db or away_team not in team_db:
            continue

        try:
            lam_yc_h, lam_yc_a = cards_model(team_db[home_team], team_db[away_team])
        except Exception:
            continue

        for line in [1.5, 2.5, 3.5, 4.5, 5.5]:
            p_over, p_under = poisson_ou_probs(lam_yc_h, lam_yc_a, line)
            over_key  = f"Over_{line}"
            under_key = f"Under_{line}"

            fair_over = fair_under = None
            if over_key in cards_dict and under_key in cards_dict:
                raw_o = implied(cards_dict[over_key])
                raw_u = implied(cards_dict[under_key])
                fair  = remove_margin([raw_o, raw_u])
                fair_over, fair_under = fair[0], fair[1]

            if over_key in cards_dict:
                check("matches", match_name, "Yellow Cards O/U", str(line),
                      f"Over {line} Yellow Cards", cards_dict[over_key], p_over, "Yellow Cards",
                      fair_imp=fair_over)
            if under_key in cards_dict:
                check("matches", match_name, "Yellow Cards O/U", str(line),
                      f"Under {line} Yellow Cards", cards_dict[under_key], p_under, "Yellow Cards",
                      fair_imp=fair_under)

    # ── 8. BOTH TEAMS TO SCORE (BTTS) ────────────────────────────────────────
    match_btts: dict[str, dict] = {}
    for row in odds_data:
        if row["page"] == "matches" and "Both Teams" in row.get("market", ""):
            mn = row["match"]
            if mn not in match_btts:
                match_btts[mn] = {}
            match_btts[mn][row["selection"].lower()] = row["odds"]

    for match_name, btts_dict in match_btts.items():
        parts = match_name.split(" - ", 1)
        if len(parts) != 2:
            continue
        home_raw, away_raw = parts
        home_team = best_match(home_raw, model_teams)
        away_team = best_match(away_raw, model_teams)
        if not home_team or not away_team:
            continue
        if home_team not in team_db or away_team not in team_db:
            continue

        try:
            from src.models.stacked_predictor import expected_goals as eg
            lam_h, lam_a = eg(team_db[home_team], team_db[away_team])
        except Exception:
            continue

        # P(BTTS Yes) = P(home scores ≥ 1) × P(away scores ≥ 1)
        p_home_scores = 1.0 - math.exp(-lam_h)
        p_away_scores = 1.0 - math.exp(-lam_a)
        p_btts_yes    = p_home_scores * p_away_scores
        p_btts_no     = 1.0 - p_btts_yes

        fair_yes = fair_no = None
        if "yes" in btts_dict and "no" in btts_dict:
            raw_y = implied(btts_dict["yes"])
            raw_n = implied(btts_dict["no"])
            fair  = remove_margin([raw_y, raw_n])
            fair_yes, fair_no = fair[0], fair[1]

        if "yes" in btts_dict:
            check("matches", match_name, "Both Teams to Score", "",
                  "BTTS Yes", btts_dict["yes"], p_btts_yes, "BTTS",
                  fair_imp=fair_yes)
        if "no" in btts_dict:
            check("matches", match_name, "Both Teams to Score", "",
                  "BTTS No", btts_dict["no"], p_btts_no, "BTTS",
                  fair_imp=fair_no)

    # ── 9. GROUP STAGE CORNER TOTALS ─────────────────────────────────────────
    # Coolbet: "Total Corners in the Group" O/U per group (A–L)
    # Model: sum corners_model(A,B) for all 6 group matchups → Poisson group total

    import itertools as _itools
    import math as _m

    def _poisson_cdf(k_max: int, lam: float) -> float:
        """P(X <= k_max) for Poisson(lam)."""
        total = 0.0
        for i in range(k_max + 1):
            total += _m.exp(-lam) * lam**i / _m.factorial(i)
        return min(1.0, total)

    # Collect Coolbet group corner O/U lines
    grp_corner_lines: dict[str, dict] = {}
    for row in odds_data:
        if row.get("market") == "Total Corners in the Group":
            grp_match = row.get("match", "")
            m_grp = re.match(r"Group ([A-L]) Specials", grp_match)
            if not m_grp:
                continue
            grp = m_grp.group(1)
            try:
                line_val = float(row.get("line", 0))
            except (TypeError, ValueError):
                continue
            if line_val == 0:
                continue
            sel = row.get("selection", "")
            if sel in ("Over", "Under"):
                grp_corner_lines.setdefault(grp, {})[sel] = (row["odds"], line_val)

    for grp_ltr, sides in grp_corner_lines.items():
        if "Over" not in sides or "Under" not in sides:
            continue
        over_odds, line_val = sides["Over"]
        under_odds, _       = sides["Under"]

        if grp_ltr not in groups:
            continue
        grp_teams = groups[grp_ltr]
        valid = [t for t in grp_teams if t in team_db]
        if len(valid) < 3:
            continue

        lam_group = 0.0
        n_matches = 0
        for ta, tb in _itools.combinations(valid, 2):
            try:
                lam_a, lam_b = corners_model(team_db[ta], team_db[tb])
                lam_group += lam_a + lam_b
                n_matches += 1
            except Exception:
                lam_group += 9.0
                n_matches += 1

        if n_matches == 0:
            continue

        k_floor = int(line_val)   # line is X.5, so P(over) = 1 - P(X <= floor)
        p_over  = 1.0 - _poisson_cdf(k_floor, lam_group)
        p_under = 1.0 - p_over

        raw_o = implied(over_odds)
        raw_u = implied(under_odds)
        fair  = remove_margin([raw_o, raw_u])
        fair_over, fair_under = fair[0], fair[1]

        label_ctx = f"Group {grp_ltr} Specials"
        check("group_specials", label_ctx, "Total Corners in the Group", str(line_val),
              f"Group {grp_ltr} Over {line_val} Corners", over_odds, p_over, "Group Corners",
              fair_imp=fair_over)
        check("group_specials", label_ctx, "Total Corners in the Group", str(line_val),
              f"Group {grp_ltr} Under {line_val} Corners", under_odds, p_under, "Group Corners",
              fair_imp=fair_under)

    # ── 10. HANDICAP (3-WAY) ─────────────────────────────────────────────────
    # line "h_cap - a_cap": effective score = actual + handicap; home win/draw/away win

    def _score_matrix(lam_h, lam_a, max_g=12):
        """Dixon-Coles corrected joint score probabilities."""
        from src.models.stacked_predictor import DIXON_COLES_RHO as _rho
        probs = {}
        total = 0.0
        for h in range(max_g + 1):
            ph = math.exp(-lam_h) * lam_h**h / math.factorial(h)
            for a in range(max_g + 1):
                pa = math.exp(-lam_a) * lam_a**a / math.factorial(a)
                p  = ph * pa
                if   h == 0 and a == 0: tau = 1.0 - lam_h * lam_a * _rho
                elif h == 1 and a == 0: tau = 1.0 + lam_a * _rho
                elif h == 0 and a == 1: tau = 1.0 + lam_h * _rho
                elif h == 1 and a == 1: tau = 1.0 - _rho
                else:                   tau = 1.0
                p *= max(0.01, tau)
                probs[(h, a)] = p
                total += p
        if total > 0:
            probs = {k: v / total for k, v in probs.items()}
        return probs

    def _handicap_probs(lam_h, lam_a, h_cap, a_cap):
        sp = _score_matrix(lam_h, lam_a)
        p_hw = p_d = p_aw = 0.0
        for (h, a), p in sp.items():
            eff_h, eff_a = h + h_cap, a + a_cap
            if   eff_h > eff_a: p_hw += p
            elif eff_h == eff_a: p_d  += p
            else:                p_aw += p
        return p_hw, p_d, p_aw

    match_hcp: dict[str, dict] = {}
    for row in odds_data:
        if row.get("page") == "matches" and row.get("market") == "Handicap (3 Way)":
            line_str = str(row.get("line", "")).strip()
            try:
                parts = line_str.split(" - ")
                h_cap, a_cap = int(parts[0]), int(parts[1])
            except Exception:
                continue
            mn = row["match"]
            if mn not in match_hcp:
                match_hcp[mn] = {}
            key = f"{h_cap}_{a_cap}"
            if key not in match_hcp[mn]:
                match_hcp[mn][key] = {"h_cap": h_cap, "a_cap": a_cap, "odds": {}}
            match_hcp[mn][key]["odds"][row["selection"]] = row["odds"]

    for match_name, lines_dict in match_hcp.items():
        parts = match_name.split(" - ", 1)
        if len(parts) != 2:
            continue
        home_raw, away_raw = parts
        home_team = best_match(home_raw, model_teams)
        away_team = best_match(away_raw, model_teams)
        if not home_team or not away_team:
            continue
        if home_team not in team_db or away_team not in team_db:
            continue
        try:
            from src.models.stacked_predictor import expected_goals as eg
            lam_h, lam_a = eg(team_db[home_team], team_db[away_team])
        except Exception:
            continue

        for key, ld in lines_dict.items():
            h_cap, a_cap = ld["h_cap"], ld["a_cap"]
            odds_map = ld["odds"]
            p_hw, p_d, p_aw = _handicap_probs(lam_h, lam_a, h_cap, a_cap)

            # Map selections to home/draw/away
            sel_map = {}
            for sel, ov in odds_map.items():
                sl = sel.strip().lower()
                if sl == home_raw.lower() or sl == home_team.lower():
                    sel_map["home"] = (sel, ov)
                elif sl == away_raw.lower() or sl == away_team.lower():
                    sel_map["away"] = (sel, ov)
                elif sl in ("draw", "x"):
                    sel_map["draw"] = (sel, ov)

            fair_probs = {}
            if len(sel_map) == 3:
                raw_list = [implied(sel_map[k][1]) for k in ("home", "draw", "away")]
                fair_list = remove_margin(raw_list)
                for i, k in enumerate(("home", "draw", "away")):
                    fair_probs[k] = fair_list[i]

            for side_key, model_p in (("home", p_hw), ("draw", p_d), ("away", p_aw)):
                if side_key not in sel_map:
                    continue
                sel, odds_val = sel_map[side_key]
                check("matches", match_name, f"Handicap ({h_cap}-{a_cap})", f"{h_cap}-{a_cap}",
                      f"{sel} HCP {h_cap}-{a_cap}", odds_val, model_p, "Handicap",
                      fair_imp=fair_probs.get(side_key))

    # ── 11. ADVANCEMENT (To Reach QF / SF / Final) ────────────────────────────
    if mc:
        _reach_map = {
            "Quarter": (mc.get("quarter", {}), "QF"),
            "Semi":    (mc.get("semi",    {}), "SF"),
            "Final":   (mc.get("finalist",{}), "Final"),
        }
        for row in odds_data:
            mkt = row.get("market", "")
            stage_key = None
            for kw, (prob_dict, lbl) in _reach_map.items():
                if f"to Reach the {kw}" in mkt or (kw == "Semi" and "Semi-final" in mkt):
                    stage_key = (prob_dict, lbl)
                    break
            if stage_key is None:
                continue
            prob_dict, lbl = stage_key
            # Extract team name from market string
            for kw in ("to Reach the Quarter-Final", "to Reach the Semi-final", "to Reach the Final"):
                mkt = mkt.replace(kw, "").strip()
            team_raw = mkt.strip()
            team = best_match(team_raw, model_teams)
            if team is None:
                continue
            p_yes = prob_dict.get(team, 0.0)
            p_no  = 1.0 - p_yes
            sel   = row.get("selection", "")
            odds  = row["odds"]
            if sel == "Yes":
                check(row["page"], row["match"], row["market"], "", f"{team} to Reach {lbl}",
                      odds, p_yes, "Advancement")
            elif sel == "No":
                check(row["page"], row["match"], row["market"], "", f"{team} NOT {lbl}",
                      odds, p_no, "Advancement")

    # ── 12. STAGE OF ELIMINATION ──────────────────────────────────────────────
    if mc:
        _qual    = mc.get("qual",     {})
        _quarter = mc.get("quarter",  {})
        _semi    = mc.get("semi",     {})
        _fin     = mc.get("finalist", {})
        _champ   = mc.get("champion", {})

        # Stage probabilities: P(eliminated AT each stage)
        def _stage_probs(team):
            q   = _qual.get(team, 0.0)
            qf  = _quarter.get(team, 0.0)
            sf  = _semi.get(team, 0.0)
            f   = _fin.get(team, 0.0)
            ch  = _champ.get(team, 0.0)
            return {
                "Group Stage":  1.0 - q,
                "Round of 32":  max(0, q - qf) * 0.5,   # split R32+R16 evenly
                "Round of 16":  max(0, q - qf) * 0.5,
                "1/4-Final":    max(0, qf - sf),
                "1/2-Final":    max(0, sf - f),
                "2nd Place":    max(0, f  - ch),
                "Winner":       ch,
            }

        for row in odds_data:
            if "Stage of Elimination" not in row.get("market", ""):
                continue
            mkt      = row["market"].replace("Stage of Elimination", "").strip()
            team_raw = mkt.strip()
            team     = best_match(team_raw, model_teams)
            if team is None:
                continue
            sp = _stage_probs(team)
            sel = row.get("selection", "")
            if sel not in sp:
                continue
            check(row["page"], row["match"], row["market"], "", f"{team} out {sel}",
                  row["odds"], sp[sel], "Stage of Elimination")

    # ── 13. TOURNAMENT H2H ────────────────────────────────────────────────────
    # "Argentina-Brazil Tournament H2H" → who finishes further in the tournament
    if mc:
        def _expected_progress(team):
            """Weighted expected 'level' reached, for H2H comparison."""
            q  = _qual.get(team, 0.0)
            qf = _quarter.get(team, 0.0)
            sf = _semi.get(team, 0.0)
            f  = _fin.get(team, 0.0)
            ch = _champ.get(team, 0.0)
            return ch*6 + (f-ch)*5 + (sf-f)*4 + (qf-sf)*3 + (q-qf)*2 + (1-q)*1

        for row in odds_data:
            mkt = row.get("market", "")
            if "Tournament H2H" not in mkt:
                continue
            # Market = "ArgentinaName-BrazilName Tournament H2H"
            h2h_part = mkt.replace("Tournament H2H", "").strip().rstrip("-").strip()
            sep_idx  = h2h_part.find("-")
            if sep_idx < 0:
                continue
            t_a_raw = h2h_part[:sep_idx].strip()
            t_b_raw = h2h_part[sep_idx+1:].strip()
            t_a = best_match(t_a_raw, model_teams)
            t_b = best_match(t_b_raw, model_teams)
            if not t_a or not t_b:
                continue

            ep_a = _expected_progress(t_a)
            ep_b = _expected_progress(t_b)
            total = ep_a + ep_b
            if total == 0:
                continue
            p_a = ep_a / total
            p_b = ep_b / total

            sel = row.get("selection", "")
            t_sel = best_match(sel, model_teams)
            if t_sel == t_a:
                check(row["page"], row["match"], mkt, "", f"{t_a} further than {t_b}",
                      row["odds"], p_a, "Tournament H2H")
            elif t_sel == t_b:
                check(row["page"], row["match"], mkt, "", f"{t_b} further than {t_a}",
                      row["odds"], p_b, "Tournament H2H")

    # ── 14. BOTH TO REACH THE FINAL ───────────────────────────────────────────
    if mc:
        _fin_dict = mc.get("finalist", {})
        for row in odds_data:
            mkt = row.get("market", "")
            if "Both to Reach the Final" not in mkt:
                continue
            # Market = "ArgentinaName & BrazilName Both to Reach the Final"
            teams_part = mkt.replace("Both to Reach the Final", "").strip().strip("&").strip()
            amp_idx    = teams_part.find(" & ")
            if amp_idx < 0:
                continue
            t_a_raw = teams_part[:amp_idx].strip()
            t_b_raw = teams_part[amp_idx+3:].strip()
            t_a = best_match(t_a_raw, model_teams)
            t_b = best_match(t_b_raw, model_teams)
            if not t_a or not t_b:
                continue
            p_a = _fin_dict.get(t_a, 0.0)
            p_b = _fin_dict.get(t_b, 0.0)
            # P(both in final) ≈ P(A) × P(B) — independent approximation
            # (slightly over-estimates when same bracket half, but adequate)
            p_both = min(p_a, p_b, p_a * p_b * 2.0)  # cap at min of individual probs
            p_not  = 1.0 - p_both
            sel    = row.get("selection", "")
            if sel == "Yes":
                check(row["page"], row["match"], mkt, "",
                      f"{t_a} & {t_b} both final", row["odds"], p_both, "Both to Final")
            elif sel == "No":
                check(row["page"], row["match"], mkt, "",
                      f"NOT both {t_a} & {t_b} final", row["odds"], p_not, "Both to Final")

    # ── 15. DOUBLE CHANCE — TOURNAMENT WINNER ─────────────────────────────────
    if mc:
        _champ_dict = mc.get("champion", {})
        for row in odds_data:
            if row.get("market") != "Winner Double Chance":
                continue
            sel = row.get("selection", "")
            # Format: "Spain or France"
            or_idx = sel.find(" or ")
            if or_idx < 0:
                continue
            t_a_raw = sel[:or_idx].strip()
            t_b_raw = sel[or_idx+4:].strip()
            t_a = best_match(t_a_raw, model_teams)
            t_b = best_match(t_b_raw, model_teams)
            if not t_a or not t_b:
                continue
            p_either = _champ_dict.get(t_a, 0.0) + _champ_dict.get(t_b, 0.0)
            check(row["page"], row["match"], row["market"], "",
                  f"{t_a} or {t_b} wins WC", row["odds"], p_either, "Double Chance Winner")

    # ── 16. TOTAL GOALS IN GROUP ──────────────────────────────────────────────
    from src.models.stacked_predictor import expected_goals as eg

    grp_goals_lines: dict[str, dict] = {}
    for row in odds_data:
        if row.get("market") == "Total Goals in the Group":
            grp_match = row.get("match", "")
            m_grp = re.match(r"Group ([A-L]) Specials", grp_match)
            if not m_grp:
                continue
            grp = m_grp.group(1)
            try:
                line_val = float(row.get("line", 0))
            except (TypeError, ValueError):
                continue
            if line_val == 0:
                continue
            sel = row.get("selection", "")
            if sel in ("Over", "Under"):
                grp_goals_lines.setdefault(grp, {})[sel] = (row["odds"], line_val)

    for grp_ltr, sides in grp_goals_lines.items():
        if "Over" not in sides or "Under" not in sides:
            continue
        over_odds, line_val = sides["Over"]
        under_odds, _       = sides["Under"]
        if grp_ltr not in groups:
            continue
        grp_teams = groups[grp_ltr]
        valid = [t for t in grp_teams if t in team_db]
        if len(valid) < 3:
            continue

        lam_goals = 0.0
        for ta, tb in _itools.combinations(valid, 2):
            try:
                lh, la = eg(team_db[ta], team_db[tb])
                lam_goals += lh + la
            except Exception:
                lam_goals += 2.5

        k_floor = int(line_val)
        p_over  = 1.0 - _poisson_cdf(k_floor, lam_goals)
        p_under = 1.0 - p_over

        raw_o = implied(over_odds)
        raw_u = implied(under_odds)
        fair  = remove_margin([raw_o, raw_u])
        fair_over, fair_under = fair[0], fair[1]

        lbl = f"Group {grp_ltr} Specials"
        check("group_specials", lbl, "Total Goals in the Group", str(line_val),
              f"Group {grp_ltr} Over {line_val} Goals", over_odds, p_over, "Group Goals",
              fair_imp=fair_over)
        check("group_specials", lbl, "Total Goals in the Group", str(line_val),
              f"Group {grp_ltr} Under {line_val} Goals", under_odds, p_under, "Group Goals",
              fair_imp=fair_under)

    # ── 17. GROUP STAGE H2H (who finishes higher in group standings) ───────────
    # Approximation: P(A above B) from direct match prob + relative qual strength
    for row in odds_data:
        mkt = row.get("market", "")
        if "Group Stage H2H" not in mkt:
            continue
        # Market = "TeamA - TeamB Group Stage H2H"
        h2h_part = mkt.replace("Group Stage H2H", "").strip().rstrip("-").strip()
        sep = h2h_part.rfind(" - ")
        if sep < 0:
            continue
        t_a_raw = h2h_part[:sep].strip()
        t_b_raw = h2h_part[sep+3:].strip()
        t_a = best_match(t_a_raw, model_teams)
        t_b = best_match(t_b_raw, model_teams)
        if not t_a or not t_b:
            continue
        if t_a not in team_db or t_b not in team_db:
            continue

        # P(A finishes above B in group) ≈ direct match prob + group-strength blend
        mp = blended_match_prob(team_db, t_a, t_b)
        p_direct_a = mp["p_win_a"] + 0.5 * mp["p_draw"]

        # Blend with relative qualification strength from MC if available
        if mc:
            qa = _qual.get(t_a, 0.5) if mc else 0.5
            qb = _qual.get(t_b, 0.5) if mc else 0.5
            p_strength_a = qa / (qa + qb) if (qa + qb) > 0 else 0.5
            p_a = 0.6 * p_direct_a + 0.4 * p_strength_a
        else:
            p_a = p_direct_a
        p_b = 1.0 - p_a

        sel = row.get("selection", "")
        t_sel = best_match(sel, model_teams)
        if t_sel == t_a:
            check(row["page"], row["match"], mkt, "",
                  f"{t_a} above {t_b} in group", row["odds"], p_a, "Group H2H")
        elif t_sel == t_b:
            check(row["page"], row["match"], mkt, "",
                  f"{t_b} above {t_a} in group", row["odds"], p_b, "Group H2H")

    # ── 18. EXACT POINTS IN GROUP STAGE ──────────────────────────────────────
    # For each team: simulate 3 group matches → P(exactly k points)
    import itertools as _itools2

    def _exact_points_probs(team: str, grp_letter: str) -> dict[int, float]:
        """
        Return {points: probability} for 0..9 points from 3 group matches.
        Uses direct match probabilities for the 3 opponents.
        """
        if grp_letter not in groups:
            return {}
        opponents = [t for t in groups[grp_letter] if t != team and t in team_db]
        if len(opponents) < 3:
            return {}
        # For each match, get (p_win, p_draw, p_lose)
        match_probs_list = []
        for opp in opponents:
            if team not in team_db or opp not in team_db:
                continue
            mp = blended_match_prob(team_db, team, opp)
            match_probs_list.append((mp["p_win_a"], mp["p_draw"], mp["p_win_b"]))
        if len(match_probs_list) < 3:
            return {}
        # Enumerate all 3^3 = 27 outcomes
        pts_prob: dict[int, float] = {}
        for outcomes in _itools2.product(["W","D","L"], repeat=3):
            prob = 1.0
            pts  = 0
            for i, outcome in enumerate(outcomes):
                pw, pd, pl = match_probs_list[i]
                if   outcome == "W": prob *= pw; pts += 3
                elif outcome == "D": prob *= pd; pts += 1
                else:                prob *= pl
            pts_prob[pts] = pts_prob.get(pts, 0.0) + prob
        return pts_prob

    # Build team → group letter map
    team_to_grp: dict[str, str] = {}
    for g_ltr, g_teams in groups.items():
        for t in g_teams:
            team_to_grp[t] = g_ltr

    for row in odds_data:
        mkt = row.get("market", "")
        if "Exact Points in Group Stage" not in mkt:
            continue
        team_raw = mkt.replace("Exact Points in Group Stage", "").strip()
        # Coolbet uses "Czechia" for Czech Republic
        if team_raw == "Czechia":
            team_raw = "Czech Republic"
        elif team_raw == "Turkiye":
            team_raw = "Turkey"
        elif team_raw == "Congo DR":
            team_raw = "DR Congo"
        team = best_match(team_raw, model_teams)
        if team is None:
            continue
        grp_letter = team_to_grp.get(team)
        if grp_letter is None:
            continue

        try:
            sel_str = str(row.get("selection", "")).strip()
            pts_val = int(sel_str)
        except (ValueError, TypeError):
            continue

        ep = _exact_points_probs(team, grp_letter)
        if not ep:
            continue
        model_p = ep.get(pts_val, 0.0)
        if model_p <= 0:
            continue
        check(row["page"], row["match"], mkt, sel_str,
              f"{team} exactly {pts_val} pts", row["odds"], model_p, "Exact Points")

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

    from src.models.tournament import load_team_db
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
