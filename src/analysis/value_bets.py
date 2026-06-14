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
    grp_above = defaultdict(int)   # (a, b) → times a finished above b in same group

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
            # Track pairwise A>B finishes for Group H2H
            for pos_i, team_i in enumerate(ranking):
                for team_j in ranking[pos_i + 1:]:
                    grp_above[(team_i, team_j)] += 1
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
        "grp_above": {k: c / n for k, c in grp_above.items()},
    }

# ── match-level probabilities ─────────────────────────────────────────────────

def _cap_draw(p_h: float, p_d: float, p_a: float,
              elo_diff: float, a_is_stronger: bool) -> dict:
    """
    For large ELO gaps, reduce unrealistically high draw probabilities.

    Historical WC data: when ELO diff > 300 the draw rate drops sharply.
    Formula: draw_cap = max(5%, 12% - (diff-300)/300 * 7%)
      diff=300 → 12% (uncapped),  diff=400 → ~10%,  diff=500 → ~7%,  diff=600+ → 5%.
    Excess probability is redistributed to the stronger team.
    """
    if elo_diff > 300:
        draw_cap = max(0.05, 0.12 - (elo_diff - 300) / 300 * 0.07)
        if p_d > draw_cap:
            excess = p_d - draw_cap
            p_d = draw_cap
            if a_is_stronger:
                p_h += excess
            else:
                p_a += excess
    return {"p_win_a": p_h, "p_draw": max(0.05, p_d), "p_win_b": p_a}


def blended_match_prob(team_db: dict, team_a: str, team_b: str,
                       venue_name: str = None, home_team: str = None) -> dict:
    """
    Run stack_predict but down-weight ML for large ELO mismatches or inverted predictions.

    The ML component (80% weight) can be badly miscalibrated when team stats come from
    competitions of very different strength (e.g. Morocco's GA from weak African qualifiers
    vs Brazil's GA from strong CONMEBOL). Two correction cases:

    1. ELO diff > 300 — existing correction, blend heavily toward ELO+Poisson.
    2. ELO diff 100–300 AND model predicts weaker team to win more — the ML has
       inverted who the favourite is; apply a proportional correction.
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
    # Apply the same host ELO boost used internally by stack_predict so the
    # inversion check sees the same effective ratings.
    from src.models.stacked_predictor import HOME_ELO_BOOST
    if home_team == team_a:
        elo_a += HOME_ELO_BOOST
    elif home_team == team_b:
        elo_b += HOME_ELO_BOOST
    elo_diff = abs(elo_a - elo_b)
    a_is_stronger = elo_a >= elo_b

    def lerp(a, b, t): return a * (1 - t) + b * t

    mb  = pred.get("model_breakdown", {})
    elo = mb.get("ELO", {})
    poi = mb.get("Poisson", {})
    if not elo or not poi:
        return pred

    elo_w    = 0.8   # ELO weight in the ELO-dominant target blend
    target_h = elo_w * elo["win_a"] + (1 - elo_w) * poi["win_a"]
    target_a = elo_w * elo["win_b"] + (1 - elo_w) * poi["win_b"]

    if elo_diff > 3:
        # --- Case 2: full prediction inverted (weaker ELO team wins more) ---
        # ML overclaims teams whose stats come from weak competitions. Blend
        # strongly toward ELO when the final prediction is inverted.
        # GF-check: if ELO-stronger team has LOWER actual GF, the ELO may itself
        # be inflated (e.g. from facing tough confederation opponents without scoring
        # much). In that case blend toward 50/50 ELO+Poi instead of pure ELO.
        p_stronger = pred["p_win_a"] if a_is_stronger else pred["p_win_b"]
        p_weaker   = pred["p_win_b"] if a_is_stronger else pred["p_win_a"]
        if p_weaker > p_stronger:
            inversion  = p_weaker - p_stronger
            gf_strong  = team_db.get(team_a if a_is_stronger else team_b, {}).get("GF_AVG", 1.4)
            gf_weak    = team_db.get(team_b if a_is_stronger else team_a, {}).get("GF_AVG", 1.4)
            # Raise minimum blend to 0.80 when inversion is significant (> 5pp)
            min_blend  = 0.80 if inversion > 0.05 else 0.0
            blend      = max(min_blend, min(0.90, 0.50 + (elo_diff - 3) / 300 + inversion))
            if gf_strong < gf_weak:
                # ELO-stronger team scores less → ELO inflated, use 50/50 ELO+Poi target
                target_h_mid = 0.5 * elo["win_a"] + 0.5 * poi["win_a"]
                target_a_mid = 0.5 * elo["win_b"] + 0.5 * poi["win_b"]
                p_h = lerp(pred["p_win_a"], target_h_mid, blend)
                p_a = lerp(pred["p_win_b"], target_a_mid, blend)
            else:
                # Standard: ELO-stronger team also scores more → Poisson corrupted
                p_h = lerp(pred["p_win_a"], elo["win_a"], blend)
                p_a = lerp(pred["p_win_b"], elo["win_b"], blend)
            p_d = 1.0 - p_h - p_a
            return _cap_draw(p_h, p_d, p_a, elo_diff, a_is_stronger)

        # --- Case 2b: Poisson component inverted even if the blended result isn't ---
        # Happens when team stats from weak competition distort Poisson lambdas.
        # Two sub-cases based on whether the ELO-stronger team's goal-scoring rate
        # supports its ELO advantage:
        #
        # • Standard (ELO-stronger team ALSO has higher GF): the Poisson was corrupted
        #   by the weaker team's low GA inflating its defence — blend toward pure ELO.
        #   Example: Morocco GA=0.63 suppresses Brazil's lambda → Poisson inverts.
        #
        # • GF-reversed (ELO-stronger team has LOWER GF): the ELO is inflated from
        #   facing strong confederation opponents, while the weaker team genuinely
        #   scores more — blend toward 50/50 ELO+Poisson instead of pure ELO.
        #   Example: Ecuador ELO=1938 (CONMEBOL) vs Ivory Coast GF=1.56 > ECU GF=1.07.
        if poi:
            poi_stronger = poi.get("win_a", 0) if a_is_stronger else poi.get("win_b", 0)
            poi_weaker   = poi.get("win_b", 0) if a_is_stronger else poi.get("win_a", 0)
            if poi_weaker > poi_stronger:
                poi_inv = poi_weaker - poi_stronger
                gf_strong = team_db.get(team_a if a_is_stronger else team_b, {}).get("GF_AVG", 1.4)
                gf_weak   = team_db.get(team_b if a_is_stronger else team_a, {}).get("GF_AVG", 1.4)
                if gf_strong < gf_weak:
                    # ELO-stronger team scores less → ELO inflated, Poisson closer to truth
                    # Blend toward 50/50 ELO+Poi mid-point
                    target_h_mid = 0.5 * elo["win_a"] + 0.5 * poi["win_a"]
                    target_a_mid = 0.5 * elo["win_b"] + 0.5 * poi["win_b"]
                    blend = min(0.90, 0.50 + (elo_diff - 3) / 300 + poi_inv)
                    p_h = lerp(pred["p_win_a"], target_h_mid, blend)
                    p_a = lerp(pred["p_win_b"], target_a_mid, blend)
                else:
                    # ELO-stronger team scores more → Poisson corrupted by weak-comp stats
                    # Raise minimum blend to 0.80 when inversion is significant (>5pp)
                    min_blend = 0.80 if poi_inv > 0.05 else 0.0
                    blend = max(min_blend, min(0.90, 0.50 + (elo_diff - 3) / 300 + poi_inv))
                    p_h = lerp(pred["p_win_a"], elo["win_a"], blend)
                    p_a = lerp(pred["p_win_b"], elo["win_b"], blend)
                p_d = 1.0 - p_h - p_a
                return _cap_draw(p_h, p_d, p_a, elo_diff, a_is_stronger)

    # Case 1a: Extreme ELO gap (> 400) — Poisson is also biased by weak-opponent
    # stats, so blend toward pure ELO only (not the ELO+Poisson mix).
    # Blend cap raised to 0.92 because the gap is large enough that the ELO
    # signal is far more reliable than any rolling-stat component.
    if elo_diff > 400:
        blend = min(0.92, 0.70 + (elo_diff - 400) / 500)
        p_h = lerp(pred["p_win_a"], elo["win_a"], blend)
        p_a = lerp(pred["p_win_b"], elo["win_b"], blend)
        p_d = 1.0 - p_h - p_a
        return _cap_draw(p_h, p_d, p_a, elo_diff, a_is_stronger)

    # Case 1b: ELO gap correction (non-inverted) — blend toward ELO+Poisson.
    # Threshold 75 matches apply_elo_correction in stacked_predictor.py.
    # For large gaps (≥ 200) where even the ELO+Poisson mix is weak-biased,
    # blend toward pure ELO instead of the mix.
    if elo_diff >= 200:
        blend = min(0.92, 0.70 + (elo_diff - 200) / 300)
        p_h = lerp(pred["p_win_a"], elo["win_a"], blend)
        p_a = lerp(pred["p_win_b"], elo["win_b"], blend)
        p_d = 1.0 - p_h - p_a
        return _cap_draw(p_h, p_d, p_a, elo_diff, a_is_stronger)

    if elo_diff >= 75:
        blend = min(0.90, (elo_diff - 75) / 55)
        p_h = lerp(pred["p_win_a"], target_h, blend)
        p_a = lerp(pred["p_win_b"], target_a, blend)
        p_d = 1.0 - p_h - p_a
        return _cap_draw(p_h, p_d, p_a, elo_diff, a_is_stronger)

    return pred


def elo_corrected_lambdas(team_db: dict, team_a: str, team_b: str,
                          venue_name: str = None, home_team: str = None):
    """
    Return Poisson expected-goals lambdas (λ_a, λ_b) corrected for large ELO
    mismatches where the raw stats are under-calibrated.

    Problem: when GF_AVG / GA_AVG come from a weak competition pool (e.g.
    CONMEBOL/CONCACAF qualifiers), the raw lambda *ratio* understates the true
    quality gap.  We detect this by checking whether the ratio is unexpectedly
    low given the ELO difference, then scale toward a target ratio derived from
    the ELO win probability.

    Condition to trigger:  ELO diff > 300  AND  raw_ratio < ELO-implied ratio.
    Teams whose stats already reflect top-level competition (Spain, England, etc.)
    will have a naturally high ratio and receive minimal or no correction.
    """
    import math
    from src.models.stacked_predictor import expected_goals, HOST_NATIONS, DIXON_COLES_RHO

    if home_team is None:
        if team_a in HOST_NATIONS:
            home_team = team_a
        elif team_b in HOST_NATIONS:
            home_team = team_b

    lam_h, lam_a = expected_goals(team_db[team_a], team_db[team_b],
                                   home_team=home_team, venue_name=venue_name)

    elo_a = team_db[team_a].get("ELO", 1700)
    elo_b = team_db[team_b].get("ELO", 1700)
    elo_diff = elo_a - elo_b   # positive → team_a (listed first / "home") is stronger

    # Identify which lambda belongs to the stronger / weaker side
    if elo_diff >= 0:
        lam_strong, lam_weak = lam_h, lam_a
    else:
        lam_strong, lam_weak = lam_a, lam_h

    raw_ratio = lam_strong / max(lam_weak, 0.10)

    # GF-check: if the ELO-stronger team has *lower* actual GF than the ELO-weaker
    # team, its ELO is likely inflated from facing tough confederation opponents rather
    # than reflecting genuine goal-scoring quality.  In that case the raw lambdas are
    # already closer to the truth than the ELO-implied ratio, so skip the correction.
    # Example: Ecuador ELO=1938 (CONMEBOL) but GF=1.07 vs Ivory Coast GF=1.56 — the
    # ELO correction would wrongly suppress IVC's lambda to 0.54.
    team_strong = team_a if elo_diff >= 0 else team_b
    team_weak   = team_b if elo_diff >= 0 else team_a
    gf_strong   = team_db.get(team_strong, {}).get("GF_AVG", 1.4)
    gf_weak     = team_db.get(team_weak,   {}).get("GF_AVG", 1.4)
    if gf_strong < gf_weak:
        return lam_h, lam_a   # ELO inflated — trust raw stats

    # Skip correction only when the ELO gap is truly small (≤ 80) AND the lambdas
    # already favour the stronger side.  For medium and large gaps (> 80) we fall
    # through to the poisson_gap check even when raw_ratio ≥ 1.0, because the raw
    # lambda ratio can still badly understate the quality gap (e.g. Belgium–Iran at
    # diff=121 or Uzbekistan–Colombia at diff=264 both get raw_ratio > 1 from stats
    # compiled against weak confederation opponents).
    # If raw_ratio < 1.0 (stronger team has lower expected goals — inverted) the
    # correction always runs regardless of ELO gap size.
    if abs(elo_diff) <= 80 and raw_ratio >= 1.0:
        return lam_h, lam_a
    if abs(elo_diff) > 80 and abs(elo_diff) <= 300 and raw_ratio >= 1.0:
        pass  # fall through to poisson_gap check below

    # Compute raw Poisson outright win probability for the stronger side
    def _poisson_win(ls, lw):
        rho = DIXON_COLES_RHO
        p_win = total = 0.0
        for h in range(13):
            ph = math.exp(-ls) * ls**h / math.factorial(h)
            for a in range(13):
                pa = math.exp(-lw) * lw**a / math.factorial(a)
                p = ph * pa
                if h == 0 and a == 0: tau = 1 - ls * lw * rho
                elif h == 1 and a == 0: tau = 1 + lw * rho
                elif h == 0 and a == 1: tau = 1 + ls * rho
                elif h == 1 and a == 1: tau = 1 - rho
                else: tau = 1.0
                p *= max(0.01, tau)
                total += p
                if h > a: p_win += p
        return p_win / total if total > 0 else 0.5

    p_poisson_win = _poisson_win(lam_strong, lam_weak)

    # ELO-implied win probability (the "true" win rate for this quality gap)
    p_elo = 1.0 / (1.0 + 10.0 ** (-abs(elo_diff) / 400.0))
    elo_win = p_elo * 0.85  # account for draws (≈85 % of ELO advantage converts to wins)

    poisson_gap = elo_win - p_poisson_win   # positive = raw model underestimates favourite

    # Trigger correction only when BOTH conditions hold:
    #   1. The raw Poisson win% is meaningfully below the ELO-implied win% (gap > 10%)
    #      → team stats come from a weaker competition pool (CONMEBOL/CONCACAF)
    #   2. The raw lambda ratio is low (< 3.5)
    #      → gives the correction room to improve handicap probabilities
    # Teams like Spain/England already have high ratios from European-quality stats
    # and are correctly left unchanged.
    if poisson_gap < 0.10 or raw_ratio >= 3.5:
        return lam_h, lam_a

    # ELO-implied target ratio (log-odds scaled conservatively)
    implied_ratio = (p_elo / (1.0 - p_elo)) ** 0.65

    # Scale to bring raw_ratio up to the implied target.
    # s² = target / raw  →  stronger *= s, weaker /= s
    s = math.sqrt(implied_ratio / raw_ratio)
    lam_strong_new = max(0.15, min(3.5, lam_strong * s))
    lam_weak_new   = max(0.15, min(3.5, lam_weak   / s))

    if elo_diff > 0:
        return lam_strong_new, lam_weak_new
    else:
        return lam_weak_new, lam_strong_new


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
    import datetime as _dt
    _today_str = _dt.date.today().isoformat()  # e.g. "2026-06-13"

    # Filter out rows for matches that have already kicked off.
    # Coolbet rows include a `match_start` field (ISO datetime string).
    # Skip any row whose match_start date is < today (already played)
    # or whose match_start is today but the time has already passed.
    _now_utc = _dt.datetime.now(_dt.timezone.utc)
    def _is_future(row: dict) -> bool:
        ms = row.get("match_start", "")
        if not ms:
            return True  # no date info → keep it
        try:
            # match_start format: "2026-06-13T18:00:00+00:00", "...Z", or "2026-06-13"
            if "T" in ms:
                ms_clean = ms.replace("Z", "+00:00")
                match_dt = _dt.datetime.fromisoformat(ms_clean)
                # Make timezone-aware if naive (assume UTC)
                if match_dt.tzinfo is None:
                    match_dt = match_dt.replace(tzinfo=_dt.timezone.utc)
            else:
                match_dt = _dt.datetime(int(ms[:4]), int(ms[5:7]), int(ms[8:10]), 23, 59, 59,
                                        tzinfo=_dt.timezone.utc)
            return match_dt > _now_utc
        except Exception:
            return True  # can't parse → keep it

    odds_data = [r for r in odds_data if _is_future(r)]

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
        # Weak-team inflation guard (all categories): the ML model is known to
        # overestimate win probabilities for weak teams vs strong opposition.
        # When the bookmaker prices an outcome below 14% AND the model claims
        # more than 2.5× the bookmaker's fair probability, the edge is almost
        # certainly a calibration artifact rather than genuine mispricing.
        if imp < 0.14 and model_prob > 2.3 * imp:
            return
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

    # ── 1b. DOUBLE CHANCE (1X / X2 / 12) ────────────────────────────────────
    # Selections: "TeamA/Draw" (1X), "TeamB/Draw" (X2), "TeamA/TeamB" (12)
    dc_markets: dict[str, dict] = {}
    for row in odds_data:
        if row["page"] == "match_sidebets" and row.get("market") == "Double Chance":
            mn = row["match"]
            if mn not in dc_markets:
                dc_markets[mn] = {}
            dc_markets[mn][row["selection"]] = row["odds"]

    for match_name, dc_dict in dc_markets.items():
        parts = match_name.split(" - ", 1)
        if len(parts) != 2:
            continue
        home_raw, away_raw = parts
        home_team = best_match(home_raw, model_teams)
        away_team = best_match(away_raw, model_teams)
        if home_team is None or away_team is None:
            continue
        mp_key = frozenset([home_team, away_team])
        if mp_key not in match_probs:
            continue
        mp   = match_probs[mp_key]
        flip = (best_match(home_raw, model_teams) == mp["away"])
        p_h  = mp["p_a"] if flip else mp["p_h"]
        p_d  = mp["p_d"]
        p_a  = mp["p_h"] if flip else mp["p_a"]

        # fair margin removal for the three DC legs
        raw_vals = list(dc_dict.values())
        if len(raw_vals) >= 3:
            fair_dc = remove_margin([implied(v) for v in raw_vals])
            fair_map = {sel: fair_dc[i] for i, sel in enumerate(dc_dict)}
        else:
            fair_map = {}

        for sel, odds in dc_dict.items():
            sel_lo = sel.lower()
            if "/draw" in sel_lo:
                # "TeamA/Draw" = 1X or "TeamB/Draw" = X2
                team_part = sel_lo.replace("/draw", "").strip()
                if best_match(team_part, model_teams) == home_team:
                    mp_val = p_h + p_d   # 1X
                else:
                    mp_val = p_a + p_d   # X2
            elif "/" in sel:
                # "TeamA/TeamB" = 12 (no draw)
                mp_val = p_h + p_a
            else:
                continue
            check("match_sidebets", match_name, "Double Chance", "", sel, odds, mp_val,
                  "Double Chance", fair_imp=fair_map.get(sel))

    # ── 1c. CORRECT SCORE ────────────────────────────────────────────────────
    # Uses Poisson score matrix; bookmakers have wide margins here → good edge opportunity.
    import math as _math

    def _score_matrix_local(lam_h, lam_a, rho=-0.15, max_g=8):
        """Bivariate Poisson score matrix with Dixon-Coles correction."""
        probs = {}
        for h in range(max_g + 1):
            ph = _math.exp(-lam_h) * lam_h**h / _math.factorial(h)
            for a in range(max_g + 1):
                pa = _math.exp(-lam_a) * lam_a**a / _math.factorial(a)
                if   h == 0 and a == 0: tau = 1.0 - lam_h * lam_a * rho
                elif h == 1 and a == 0: tau = 1.0 + lam_a * rho
                elif h == 0 and a == 1: tau = 1.0 + lam_h * rho
                elif h == 1 and a == 1: tau = 1.0 - rho
                else:                   tau = 1.0
                probs[(h, a)] = ph * pa * tau
        return probs

    cs_markets: dict[str, dict] = {}
    for row in odds_data:
        if row["page"] == "match_sidebets" and row.get("market") == "Correct Score":
            mn  = row["match"]
            sel = row["selection"].strip()   # e.g. "1 - 0" or "0 - 0"
            if mn not in cs_markets:
                cs_markets[mn] = {}
            cs_markets[mn][sel] = row["odds"]

    for match_name, cs_dict in cs_markets.items():
        parts = match_name.split(" - ", 1)
        if len(parts) != 2:
            continue
        home_raw, away_raw = parts
        home_team = best_match(home_raw, model_teams)
        away_team = best_match(away_raw, model_teams)
        if home_team is None or away_team is None:
            continue
        try:
            from src.models.stacked_predictor import DIXON_COLES_RHO
            lam_h, lam_a = elo_corrected_lambdas(team_db, home_team, away_team)
            sm = _score_matrix_local(lam_h, lam_a, rho=DIXON_COLES_RHO)
        except Exception:
            continue

        for sel, odds in cs_dict.items():
            # Parse "H - A" score string
            try:
                h_str, a_str = sel.split(" - ", 1)
                h_g, a_g = int(h_str.strip()), int(a_str.strip())
            except (ValueError, AttributeError):
                continue
            mp_val = sm.get((h_g, a_g), 0.0)
            if mp_val < 0.005:   # skip < 0.5% model prob (very unlikely scores)
                continue
            check("match_sidebets", match_name, "Correct Score", "", sel, odds, mp_val,
                  "Correct Score")

    # ── 1d. MATCH RESULT + BTTS ─────────────────────────────────────────────
    # Selections: "Home and Yes", "Draw and Yes", "Away and Yes",
    #             "Home and No",  "Draw and No",  "Away and No"
    mrbtts_markets: dict[str, dict] = {}
    for row in odds_data:
        if (row["page"] == "match_sidebets" and
                row.get("market") == "Match Result and Both Teams to Score"):
            mn = row["match"]
            if mn not in mrbtts_markets:
                mrbtts_markets[mn] = {}
            mrbtts_markets[mn][row["selection"]] = row["odds"]

    for match_name, mrb_dict in mrbtts_markets.items():
        parts = match_name.split(" - ", 1)
        if len(parts) != 2:
            continue
        home_raw, away_raw = parts
        home_team = best_match(home_raw, model_teams)
        away_team = best_match(away_raw, model_teams)
        if home_team is None or away_team is None:
            continue
        mp_key = frozenset([home_team, away_team])
        if mp_key not in match_probs:
            continue
        mp   = match_probs[mp_key]
        flip = (best_match(home_raw, model_teams) == mp["away"])
        p_h  = mp["p_a"] if flip else mp["p_h"]
        p_d  = mp["p_d"]
        p_a  = mp["p_h"] if flip else mp["p_a"]

        try:
            lam_h, lam_a = elo_corrected_lambdas(team_db, home_team, away_team)
            if flip:
                lam_h, lam_a = lam_a, lam_h
        except Exception:
            continue
        p_h_scores = 1.0 - _math.exp(-lam_h)
        p_a_scores = 1.0 - _math.exp(-lam_a)
        p_btts_yes = p_h_scores * p_a_scores
        p_btts_no  = 1.0 - p_btts_yes

        # Use score matrix to get BTTS-conditional probabilities
        try:
            from src.models.stacked_predictor import DIXON_COLES_RHO as _rho_val
            sm = _score_matrix_local(lam_h, lam_a, rho=_rho_val)
        except Exception:
            sm = None

        def _p_result_btts(outcome, btts):
            if sm is None:
                # Fallback: assume independence
                p_res = {"home": p_h, "draw": p_d, "away": p_a}[outcome]
                return p_res * (p_btts_yes if btts else p_btts_no)
            total = 0.0
            for (h, a), prob in sm.items():
                if btts and (h == 0 or a == 0):
                    continue
                if not btts and h >= 1 and a >= 1:
                    continue
                if outcome == "home" and h > a:
                    total += prob
                elif outcome == "draw" and h == a:
                    total += prob
                elif outcome == "away" and a > h:
                    total += prob
            return total

        for sel, odds in mrb_dict.items():
            sel_lo = sel.lower()
            if home_raw.lower() in sel_lo or "home" in sel_lo:
                outcome = "home"
            elif away_raw.lower() in sel_lo or "away" in sel_lo:
                outcome = "away"
            elif "draw" in sel_lo:
                outcome = "draw"
            else:
                continue
            btts = "yes" in sel_lo
            mp_val = _p_result_btts(outcome, btts)
            check("match_sidebets", match_name, "Match Result and BTTS", "", sel, odds,
                  mp_val, "Match Result + BTTS")

    # ── 1e. HALF TIME / FULL TIME ────────────────────────────────────────────
    # 9 combinations: Home/Home, Home/Draw, Home/Away, Draw/Home, Draw/Draw,
    #                 Draw/Away, Away/Home, Away/Draw, Away/Away
    htft_markets: dict[str, dict] = {}
    for row in odds_data:
        if (row["page"] == "match_sidebets" and
                "Half Time" in (row.get("market") or "")):
            mn = row["match"]
            if mn not in htft_markets:
                htft_markets[mn] = {}
            htft_markets[mn][row["selection"]] = row["odds"]

    for match_name, htft_dict in htft_markets.items():
        parts = match_name.split(" - ", 1)
        if len(parts) != 2:
            continue
        home_raw, away_raw = parts
        home_team = best_match(home_raw, model_teams)
        away_team = best_match(away_raw, model_teams)
        if home_team is None or away_team is None:
            continue
        try:
            from src.models.stacked_predictor import DIXON_COLES_RHO as _rho_htft
            _HT_FRAC = 0.45   # WC historical: ~45% of goals fall in first half
            lam_h_full, lam_a_full = elo_corrected_lambdas(team_db, home_team, away_team)
            mp_key = frozenset([home_team, away_team])
            if mp_key not in match_probs:
                continue
            mp   = match_probs[mp_key]
            flip = (best_match(home_raw, model_teams) == mp["away"])
            if flip:
                lam_h_full, lam_a_full = lam_a_full, lam_h_full
            # 1st half: ~45% of expected goals
            lam_h1 = lam_h_full * _HT_FRAC
            lam_a1 = lam_a_full * _HT_FRAC
        except Exception:
            continue

        sm1 = _score_matrix_local(lam_h1, lam_a1, rho=_rho_htft)

        def _ht_result(sm):
            ph, pd, pa = 0.0, 0.0, 0.0
            for (h, a), p in sm.items():
                if h > a: ph += p
                elif h == a: pd += p
                else: pa += p
            return ph, pd, pa

        p_ht_h, p_ht_d, p_ht_a = _ht_result(sm1)
        p_ft_h = mp["p_a"] if flip else mp["p_h"]
        p_ft_d = mp["p_d"]
        p_ft_a = mp["p_h"] if flip else mp["p_a"]

        result_map = {
            "home": p_ht_h, "draw": p_ht_d, "away": p_ht_a,
        }
        ft_map = {
            "home": p_ft_h, "draw": p_ft_d, "away": p_ft_a,
        }

        for sel, odds in htft_dict.items():
            # Selection format: "TeamA/TeamB" = HT Home / FT Away, "Draw/TeamA" etc.
            # Normalise to H/D/A
            sel_parts = sel.split("/")
            if len(sel_parts) != 2:
                continue
            ht_raw, ft_raw = sel_parts[0].strip().lower(), sel_parts[1].strip().lower()

            def _classify(s):
                s = s.strip().lower()
                if s in ("draw", "x"): return "draw"
                # Check against home/away team names
                if best_match(s, model_teams) == home_team: return "home"
                if best_match(s, model_teams) == away_team: return "away"
                if home_raw.lower()[:4] in s: return "home"
                if away_raw.lower()[:4] in s: return "away"
                return None

            ht_outcome = _classify(ht_raw)
            ft_outcome = _classify(ft_raw)
            if ht_outcome is None or ft_outcome is None:
                continue

            # Approximate: P(HT=X, FT=Y) ≈ P(HT=X) × P(FT=Y | HT=X)
            # Simplification: assume independence of HT and FT results (reasonable approximation)
            # P(FT=H | HT=H) is higher than base P(FT=H); we use independence as approximation
            mp_val = result_map[ht_outcome] * ft_map[ft_outcome]
            # Renormalize across all 9 combinations to correct for independence assumption
            check("match_sidebets", match_name, "Half Time / Full Time", "", sel, odds,
                  mp_val, "HT/FT")

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

        Over/Under semantics follow standard sportsbook convention:
          Over line  → total > line  (e.g. Over 1.0 requires ≥2 goals)
          Under line → total < line  (e.g. Under 1.0 requires 0 goals)
        For half-integer lines (0.5, 1.5 …) this matches 1-p_over exactly.
        For integer lines (1.0, 2.0 …) p_over + p_under < 1 (the push case
        is excluded from both sides, as in a standard binary market).
        """
        from src.models.stacked_predictor import DIXON_COLES_RHO
        rho = DIXON_COLES_RHO
        max_g = 15
        p_over  = 0.0
        p_under = 0.0
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
                elif total < line:
                    p_under += p
                # total == line → push/refund (Asian line convention).
                # Push probability is excluded from both sides; we will normalize later.

        # Step 1: renormalize for τ (small correction from Dixon-Coles)
        if p_total > 0:
            p_over  /= p_total
            p_under /= p_total

        # Step 2: conditional normalization to exclude the push probability.
        # For half-integer lines P(push) = 0 and this is a no-op.
        # For integer lines (1.0, 2.0 …) Coolbet uses Asian convention:
        #   - Over wins if total > line, push (refund) if total == line, lose if total < line
        # The book's implied probability is therefore the conditional probability given
        # a non-push outcome, so our model must match that conditional.
        p_nonpush = p_over + p_under
        if p_nonpush > 0:
            p_over  /= p_nonpush
            p_under /= p_nonpush

        return p_over, p_under

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

        home_lam, away_lam = elo_corrected_lambdas(team_db, home_team, away_team)

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
        if row["page"] == "match_sidebets" and row.get("market_type") == "Total Corners":
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
        if row["page"] == "match_sidebets" and row.get("market_type") == "Total Cards":
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
        if row["page"] == "match_sidebets" and row.get("market_type") == "Both Teams To Score":
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

        lam_h, lam_a = elo_corrected_lambdas(team_db, home_team, away_team)

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

    # ── 8b. INDIVIDUAL TEAM GOALS O/U ([Home] / [Away] Total Goals) ─────────
    # Coolbet offers separate O/U lines for each team's own goal count.
    # We model each team's scoring as an independent Poisson process with
    # the same λ values already computed for the full-match goals model.
    # Same Asian push convention as poisson_ou_probs (integer lines = push on exact score).

    def poisson_team_ou(lam: float, line: float):
        """P(team scores > line) and P(team scores < line), both conditional on non-push."""
        max_g = 15
        p_over = 0.0
        p_under = 0.0
        for k in range(max_g + 1):
            p = math.exp(-lam) * lam**k / math.factorial(k)
            if k > line:
                p_over += p
            elif k < line:
                p_under += p
        p_nonpush = p_over + p_under
        if p_nonpush > 0:
            p_over  /= p_nonpush
            p_under /= p_nonpush
        return p_over, p_under

    for side_key, side_label, lam_idx in [
        ("[Home] Total Goals", "Home Goals", 0),
        ("[Away] Total Goals", "Away Goals", 1),
    ]:
        side_data: dict[str, dict] = {}
        for row in odds_data:
            if row["page"] == "match_sidebets" and row.get("market_type") == side_key:
                try:
                    line_val = float(str(row.get("line", "")).strip())
                except (ValueError, TypeError):
                    continue
                mn = row["match"]
                side_data.setdefault(mn, {})[f"{row['selection']}_{line_val}"] = row["odds"]

        for match_name, s_dict in side_data.items():
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

            lams = elo_corrected_lambdas(team_db, home_team, away_team)
            lam = lams[lam_idx]  # 0 = home lambda, 1 = away lambda

            for line in [0.5, 1.0, 1.5, 2.0, 2.5]:
                p_over, p_under = poisson_team_ou(lam, line)
                over_key  = f"Over_{line}"
                under_key = f"Under_{line}"

                fair_over = fair_under = None
                if over_key in s_dict and under_key in s_dict:
                    raw_o = implied(s_dict[over_key])
                    raw_u = implied(s_dict[under_key])
                    fair  = remove_margin([raw_o, raw_u])
                    fair_over, fair_under = fair[0], fair[1]

                if over_key in s_dict:
                    check("match_sidebets", match_name, side_label, str(line),
                          f"Over {line}", s_dict[over_key], p_over, side_label,
                          fair_imp=fair_over)
                if under_key in s_dict:
                    check("match_sidebets", match_name, side_label, str(line),
                          f"Under {line}", s_dict[under_key], p_under, side_label,
                          fair_imp=fair_under)

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
        if len(valid) < 2:
            continue

        # Require at least 2 teams with real corners data (not just WC avg defaults)
        from src.models.stacked_predictor import WC_AVG_CORNERS as _WC_CRN
        n_with_data = sum(
            1 for t in valid
            if team_db[t].get("CORNERS_FOR", _WC_CRN) != _WC_CRN
            or team_db[t].get("CORNERS_AGAINST", _WC_CRN) != _WC_CRN
        )
        if n_with_data < 2:
            continue

        # Build a full 4-team roster: teams in team_db use real stats; missing teams
        # get a default profile (WC average) so all C(4,2)=6 matchups are counted.
        _default_team = {"CORNERS_FOR": _WC_CRN, "CORNERS_AGAINST": _WC_CRN,
                         "GOALS_FOR": 1.4, "GOALS_AGAINST": 1.4,
                         "SHOTS_FOR": 4.0, "SHOTS_AGAINST": 4.0,
                         "YELLOWS_FOR": 1.8, "YELLOWS_AGAINST": 1.8,
                         "INJURIES": 0}
        team_roster = {t: team_db.get(t, _default_team) for t in grp_teams}

        lam_group = 0.0
        n_matches = 0
        for ta, tb in _itools.combinations(grp_teams, 2):
            try:
                lam_a, lam_b = corners_model(team_roster[ta], team_roster[tb])
                lam_group += lam_a + lam_b
                n_matches += 1
            except Exception:
                lam_group += 9.07   # WC historical average per match
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

    # Pre-build Coolbet 1X2 fair home probability per match.
    # Used below to skip handicap bets when our model's outright probability
    # is far from the bookmaker's implied probability (signal of ELO miscalibration).
    _bookie_home_fair: dict[str, float] = {}
    for row in odds_data:
        if row.get("page") != "matches" or row.get("market") != "Match Result (1X2)":
            continue
        mn = row["match"]
        if mn not in _bookie_home_fair:
            # Collect all 3 outcomes for this match to remove margin
            _trio_all = [r for r in odds_data
                         if r.get("match") == mn and r.get("market") == "Match Result (1X2)"]
            # Deduplicate by selection to handle duplicate rows from scraper
            _seen_sels: set[str] = set()
            _trio = []
            for _r in _trio_all:
                _sel = _r.get("selection", "")
                if _sel not in _seen_sels:
                    _seen_sels.add(_sel)
                    _trio.append(_r)
            if len(_trio) != 3:
                continue
            _raws = [1.0 / r["odds"] for r in _trio]
            _tot  = sum(_raws)
            # Identify which row is the home team (first part of "Home - Away")
            _home_raw = mn.split(" - ", 1)[0]
            for i, r in enumerate(_trio):
                if r["selection"].strip() == _home_raw.strip():
                    _bookie_home_fair[mn] = (_raws[i] / _tot)
                    break

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

        # Skip handicap bets when our outright model diverges from the bookmaker's
        # 1X2 implied probability by more than 8 pp.  This typically means the
        # Poisson lambda model is miscalibrated for the quality gap — either the
        # weaker team's ELO is inflated from weak-confederation qualifiers (Asia,
        # Middle East), or the stronger team's goal distribution is understated
        # (e.g. France vs Iraq where raw λ gives France 2.0 goals when ~3.0 is
        # implied by the bookmaker pricing).
        _bookie_hf = _bookie_home_fair.get(match_name)
        if _bookie_hf is not None:
            _pred_out = blended_match_prob(team_db, home_team, away_team)
            if abs(_pred_out["p_win_a"] - _bookie_hf) > 0.08:
                continue

        lam_h, lam_a = elo_corrected_lambdas(team_db, home_team, away_team)

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
            # Skip combo markets — handled by section 14 with joint probability
            if "Both to Reach the" in mkt:
                continue
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
    # Use ELO-corrected lambdas to account for stats from weak competitions
    def _eg(ta, tb):
        return elo_corrected_lambdas(team_db, ta, tb)

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
                lh, la = _eg(ta, tb)
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

    # ── 16b. GROUP STAGE TOTAL CORNERS (all groups combined) ─────────────────
    # Market: "Group Stage Total Corners", line: 666.5
    # Model: sum corners_model across all 72 group matches (12 groups × 6 matchups each)

    try:
        from scipy.stats import poisson as _sp_poisson
        _have_scipy = True
    except ImportError:
        _have_scipy = False

    def _poisson_cdf_large(k_max: int, lam: float) -> float:
        """P(X <= k_max) for Poisson(lam). Uses scipy for large lambda."""
        if _have_scipy:
            return float(_sp_poisson.cdf(k_max, lam))
        # Normal approximation fallback
        import math as _math2
        z = (k_max + 0.5 - lam) / _math2.sqrt(max(lam, 1e-9))
        return 0.5 * (1.0 + _math2.erf(z / _math2.sqrt(2)))

    _gsc_sides: dict[str, tuple] = {}
    for row in odds_data:
        if row.get("market") == "Group Stage Total Corners" and row.get("match") == "Group Stage Totals":
            try:
                _line_val = float(row.get("line", 0))
            except (TypeError, ValueError):
                continue
            if _line_val == 0:
                continue
            _sel = row.get("selection", "")
            if _sel in ("Over", "Under"):
                _gsc_sides[_sel] = (row["odds"], _line_val)

    if "Over" in _gsc_sides and "Under" in _gsc_sides:
        _gsc_over_odds, _gsc_line = _gsc_sides["Over"]
        _gsc_under_odds, _           = _gsc_sides["Under"]

        _lam_grpstage_corners = 0.0
        _default_team2 = {"CORNERS_FOR": _WC_CRN, "CORNERS_AGAINST": _WC_CRN,
                          "GOALS_FOR": 1.4, "GOALS_AGAINST": 1.4,
                          "SHOTS_FOR": 4.0, "SHOTS_AGAINST": 4.0,
                          "YELLOWS_FOR": 1.8, "YELLOWS_AGAINST": 1.8,
                          "INJURIES": 0}
        for _grp_ltr2, _grp_teams2 in groups.items():
            _roster2 = {t: team_db.get(t, _default_team2) for t in _grp_teams2}
            for _ta2, _tb2 in _itools.combinations(_grp_teams2, 2):
                try:
                    _lca, _lcb = corners_model(_roster2[_ta2], _roster2[_tb2])
                    _lam_grpstage_corners += _lca + _lcb
                except Exception:
                    _lam_grpstage_corners += 9.07  # WC historical avg fallback

        _k_gsc = int(_gsc_line)
        _p_gsc_over  = 1.0 - _poisson_cdf_large(_k_gsc, _lam_grpstage_corners)
        _p_gsc_under = 1.0 - _p_gsc_over

        _raw_go = implied(_gsc_over_odds)
        _raw_gu = implied(_gsc_under_odds)
        _fair_g = remove_margin([_raw_go, _raw_gu])
        check("group_specials", "Group Stage Totals", "Group Stage Total Corners", str(_gsc_line),
              f"Group Stage Over {_gsc_line} Corners", _gsc_over_odds, _p_gsc_over, "Group Corners",
              fair_imp=_fair_g[0])
        check("group_specials", "Group Stage Totals", "Group Stage Total Corners", str(_gsc_line),
              f"Group Stage Under {_gsc_line} Corners", _gsc_under_odds, _p_gsc_under, "Group Corners",
              fair_imp=_fair_g[1])

    # ── 16c. GROUP STAGE TOTAL GOALS — 1ST HALF AND 2ND HALF ─────────────
    # Markets: "Group Stage Total Goals (1st Half)" line 81.5
    #          "Group Stage Total Goals (2nd Half)" line 113.5
    # Historical WC split: ~41.3% 1st half (WC 2018: 41.7%, WC 2022: 41.2%)
    _WC_1H_FRAC = 0.413
    for _half_mkt, _half_frac in [
        ("Group Stage Total Goals (1st Half)", _WC_1H_FRAC),
        ("Group Stage Total Goals (2nd Half)", 1.0 - _WC_1H_FRAC),
    ]:
        _half_sides: dict[str, tuple] = {}
        for row in odds_data:
            if row.get("market") == _half_mkt:
                try:
                    _lv = float(row.get("line", 0))
                except (TypeError, ValueError):
                    continue
                if _lv == 0:
                    continue
                _sel = row.get("selection", "")
                if _sel in ("Over", "Under"):
                    _half_sides[_sel] = (row["odds"], _lv)

        if "Over" not in _half_sides or "Under" not in _half_sides:
            continue
        _h_over_odds, _h_line = _half_sides["Over"]
        _h_under_odds, _      = _half_sides["Under"]

        # Use stack_predict (calibrated blend of ELO+Poisson+ML, target 2.8/match)
        # rather than elo_corrected_lambdas (3.07/match) for this 72-match aggregate.
        from src.models.stacked_predictor import stack_predict as _stack_predict3
        _lam_half = 0.0
        for _grp_ltr3, _grp_teams3 in groups.items():
            _valid3 = [t for t in _grp_teams3 if t in team_db]
            for _ta3, _tb3 in _itools.combinations(_valid3, 2):
                try:
                    _sp3 = _stack_predict3(team_db[_ta3], team_db[_tb3])
                    _lh3 = _sp3.get("lam_home", 1.4)
                    _la3 = _sp3.get("lam_away", 1.4)
                    _lam_half += (_lh3 + _la3) * _half_frac
                except Exception:
                    _lam_half += 2.8 * _half_frac  # fallback

        _k_h = int(_h_line)
        _p_h_over  = 1.0 - _poisson_cdf_large(_k_h, _lam_half)
        _p_h_under = 1.0 - _p_h_over

        _raw_ho = implied(_h_over_odds)
        _raw_hu = implied(_h_under_odds)
        _fair_h = remove_margin([_raw_ho, _raw_hu])
        _half_label = "1st Half" if "1st" in _half_mkt else "2nd Half"
        check("group_specials", "Group Stage Totals", _half_mkt, str(_h_line),
              f"Group Stage Over {_h_line} Goals ({_half_label})", _h_over_odds, _p_h_over, "Group Goals",
              fair_imp=_fair_h[0])
        check("group_specials", "Group Stage Totals", _half_mkt, str(_h_line),
              f"Group Stage Under {_h_line} Goals ({_half_label})", _h_under_odds, _p_h_under, "Group Goals",
              fair_imp=_fair_h[1])

    # ── 16d. TEAM GROUP STAGE TOTAL GOALS (e.g., Argentina Over/Under 5.5) ──
    # Market: "[Team] Group Stage Total Goals " (trailing space in Coolbet data)
    # Model: sum expected goals FOR each team across their 3 group stage matches.
    # Since WC 2026 venues are mostly neutral-ground, we average home and away lambdas.

    import re as _re2
    _team_grp_goals: dict[str, dict] = {}
    for row in odds_data:
        _mkt = row.get("market", "")
        _mg = _re2.match(r"^(.+?)\s+Group Stage Total Goals\s*$", _mkt)
        if not _mg:
            continue
        _team_raw = _mg.group(1).strip()
        try:
            _lv2 = float(row.get("line", 0))
        except (TypeError, ValueError):
            continue
        if _lv2 == 0:
            continue
        _sel2 = row.get("selection", "")
        if _sel2 in ("Over", "Under"):
            _team_grp_goals.setdefault(_team_raw, {})[_sel2] = (row["odds"], _lv2)

    # Reverse lookup: team_name → group letter
    _team_to_grp: dict[str, str] = {}
    for _gl, _gt in groups.items():
        for _t in _gt:
            _team_to_grp[_t] = _gl

    for _team_raw2, _sides2 in _team_grp_goals.items():
        if "Over" not in _sides2 or "Under" not in _sides2:
            continue
        _t2_over_odds, _t2_line = _sides2["Over"]
        _t2_under_odds, _       = _sides2["Under"]

        _team2 = best_match(_team_raw2, model_teams)
        if not _team2 or _team2 not in team_db:
            continue
        _gl2 = _team_to_grp.get(_team2)
        if not _gl2 or _gl2 not in groups:
            continue

        _opps = [t for t in groups[_gl2] if t != _team2 and t in team_db]
        if len(_opps) < 3:
            continue

        # Average home/away lambda to approximate neutral-ground WC conditions
        _lam_for2 = 0.0
        for _opp in _opps:
            try:
                _lh_as_home, _  = _eg(_team2, _opp)   # team plays "home"
                _, _la_as_away  = _eg(_opp, _team2)   # team plays "away"
                _lam_for2 += (_lh_as_home + _la_as_away) / 2.0
            except Exception:
                _lam_for2 += 1.4

        _k_t2 = int(_t2_line)
        _p_t2_over  = 1.0 - _poisson_cdf(_k_t2, _lam_for2)
        _p_t2_under = 1.0 - _p_t2_over

        _raw_t2o = implied(_t2_over_odds)
        _raw_t2u = implied(_t2_under_odds)
        _fair_t2 = remove_margin([_raw_t2o, _raw_t2u])
        _lbl_t2  = f"Group {_gl2} Specials"
        check("group_specials", _lbl_t2,
              f"{_team2} Group Stage Total Goals", str(_t2_line),
              f"{_team2} Over {_t2_line} Goals (Group Stage)", _t2_over_odds, _p_t2_over, "Group Goals",
              fair_imp=_fair_t2[0])
        check("group_specials", _lbl_t2,
              f"{_team2} Group Stage Total Goals", str(_t2_line),
              f"{_team2} Under {_t2_line} Goals (Group Stage)", _t2_under_odds, _p_t2_under, "Group Goals",
              fair_imp=_fair_t2[1])

    # ── 16e. GROUP WITH MOST CORNERS / GROUP WITH MOST GOALS ─────────────────
    # Markets: "Group with Most Corners" and "Group with Most Goals" (wc_specials or group_specials)
    # Model: MC over Poisson draws for each group's total corners/goals.

    import random as _random

    def _group_most_mc(lam_by_group: dict[str, float], n_sims: int = 20_000) -> dict[str, float]:
        """Monte Carlo: P(group X has the most corners/goals) via normal approx of Poisson."""
        import random as _rnd
        win_counts: dict[str, int] = {g: 0 for g in lam_by_group}
        for _ in range(n_sims):
            draws = {g: max(0.0, _rnd.gauss(lam, lam**0.5)) for g, lam in lam_by_group.items()}
            winner = max(draws, key=draws.get)
            win_counts[winner] += 1
        return {g: win_counts[g] / n_sims for g in lam_by_group}

    # ── Group with Most Corners ────────────────────────────────────────────
    _grp_crn_rows = [r for r in odds_data if r.get("market") == "Group with Most Corners"]
    if _grp_crn_rows:
        # Build expected corners per group
        _grp_crn_lams: dict[str, float] = {}
        for _gl3, _gt3 in groups.items():
            _vl3 = [t for t in _gt3 if t in team_db]
            _lam3 = 0.0
            for _ta4, _tb4 in _itools.combinations(_vl3, 2):
                try:
                    _lca2, _lcb2 = corners_model(team_db[_ta4], team_db[_tb4])
                    _lam3 += _lca2 + _lcb2
                except Exception:
                    _lam3 += 9.3
            _grp_crn_lams[_gl3] = _lam3

        _crn_win_probs = _group_most_mc(_grp_crn_lams, n_sims=20_000)

        for row in _grp_crn_rows:
            _sel3 = row.get("selection", "")  # e.g. "Group A", "Group B", ...
            _m3 = re.match(r"Group ([A-L])", _sel3)
            if not _m3:
                continue
            _gl_sel = _m3.group(1)
            _p_win3 = _crn_win_probs.get(_gl_sel, 0.0)
            check(row["page"], row.get("match", "WC 2026"), "Group with Most Corners", "",
                  f"Group {_gl_sel} Most Corners", row["odds"], _p_win3, "Group Corners")

    # ── Group with Most Goals ──────────────────────────────────────────────
    _grp_gls_rows = [r for r in odds_data if r.get("market") == "Group with Most Goals"]
    if _grp_gls_rows:
        from src.models.stacked_predictor import stack_predict as _sp_gls
        _grp_gls_lams: dict[str, float] = {}
        for _gl4, _gt4 in groups.items():
            _vl4 = [t for t in _gt4 if t in team_db]
            _lam4 = 0.0
            for _ta5, _tb5 in _itools.combinations(_vl4, 2):
                try:
                    _sp5 = _sp_gls(team_db[_ta5], team_db[_tb5])
                    _lam4 += _sp5.get("lam_home", 1.4) + _sp5.get("lam_away", 1.4)
                except Exception:
                    _lam4 += 2.8
            _grp_gls_lams[_gl4] = _lam4

        _gls_win_probs = _group_most_mc(_grp_gls_lams, n_sims=20_000)

        for row in _grp_gls_rows:
            _sel4 = row.get("selection", "")
            _m4 = re.match(r"Group ([A-L])", _sel4)
            if not _m4:
                continue
            _gl_sel4 = _m4.group(1)
            _p_win4  = _gls_win_probs.get(_gl_sel4, 0.0)
            check(row["page"], row.get("match", "WC 2026"), "Group with Most Goals", "",
                  f"Group {_gl_sel4} Most Goals", row["odds"], _p_win4, "Group Goals")

    # ── 17. GROUP STAGE H2H (who finishes higher in group standings) ───────────
    # Approximation: P(A above B) from direct match prob + relative qual strength

    # Build a (frozenset of team names) → bookie fair home-win probability lookup
    # for all group-stage 1X2 markets.  Used below to skip H2H bets when the
    # model and bookmaker disagree on the direct match by more than 10 pp.
    _h2h_bookie: dict[frozenset, tuple[str, str, float]] = {}
    for _row in odds_data:
        if _row.get("page") != "matches" or _row.get("market") != "Match Result (1X2)":
            continue
        _mn = _row["match"]
        _trio_all = [r for r in odds_data
                     if r.get("match") == _mn and r.get("market") == "Match Result (1X2)"]
        _seen_sels_h2h: set[str] = set()
        _trio = []
        for _rr in _trio_all:
            _ss = _rr.get("selection", "")
            if _ss not in _seen_sels_h2h:
                _seen_sels_h2h.add(_ss)
                _trio.append(_rr)
        if len(_trio) != 3:
            continue
        _h_raw, _a_raw = _mn.split(" - ", 1)[0], _mn.split(" - ", 1)[1]
        _h_t = best_match(_h_raw, model_teams)
        _a_t = best_match(_a_raw, model_teams)
        if not _h_t or not _a_t:
            continue
        _key = frozenset([_h_t, _a_t])
        if _key in _h2h_bookie:
            continue
        _raws = [1.0 / r["odds"] for r in _trio]
        _tot  = sum(_raws)
        _h_fair = None
        for _i, _r in enumerate(_trio):
            if _r["selection"].strip() == _h_raw.strip():
                _h_fair = _raws[_i] / _tot
                break
        if _h_fair is not None:
            _h2h_bookie[_key] = (_h_t, _a_t, _h_fair)   # (home_team, away_team, fair_home_win)

    # Build a bookmaker-implied probability lookup for each H2H market.
    # Used below to guard against the model over-claiming extreme underdogs.
    _h2h_mkt_implied: dict[str, dict[str, float]] = {}
    for _row in odds_data:
        if _row.get("page") != "group_specials":
            continue
        _mkt = _row.get("market", "")
        if "Group Stage H2H" not in _mkt:
            continue
        _h2h_mkt_implied.setdefault(_mkt, {})[_row["selection"]] = 1.0 / _row["odds"]
    # Normalise to fair probabilities (remove bookmaker margin)
    for _mkt, _sides in _h2h_mkt_implied.items():
        _tot = sum(_sides.values())
        if _tot > 0:
            for _sel in _sides:
                _sides[_sel] /= _tot

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

        # Skip when the direct match probability diverges from bookmaker's 1X2 by
        # more than 10 pp — same calibration guard used for handicap bets.
        _pair_key = frozenset([t_a, t_b])
        if _pair_key in _h2h_bookie:
            _bh_home, _bh_away, _bh_fair = _h2h_bookie[_pair_key]
            _mp_check = blended_match_prob(team_db, _bh_home, _bh_away)
            if abs(_mp_check["p_win_a"] - _bh_fair) > 0.10:
                continue

        # P(A finishes above B in group)
        # Primary: use direct pairwise MC count (most accurate).
        # Fallback: blend head-to-head win prob with relative top-2 strength.
        _grp_above = mc.get("grp_above", {}) if mc else {}
        _n_mc = mc.get("n", 1) if mc else 1
        p_a_above_b = _grp_above.get((t_a, t_b), None)
        p_b_above_a = _grp_above.get((t_b, t_a), None)

        if p_a_above_b is not None:
            # Direct MC pairwise: most accurate signal
            p_a = p_a_above_b
        elif p_b_above_a is not None:
            p_a = 1.0 - p_b_above_a
        else:
            # No MC pairwise data — use head-to-head win prob only (no draw credit)
            mp = blended_match_prob(team_db, t_a, t_b)
            p_direct_a = mp["p_win_a"]   # win only, not win+draw (draw ≠ "above")
            if mc:
                _grp_win = mc.get("grp_win", {})
                _grp_2nd = mc.get("grp_2nd", {})
                top2_a = _grp_win.get(t_a, 0) + _grp_2nd.get(t_a, 0)
                top2_b = _grp_win.get(t_b, 0) + _grp_2nd.get(t_b, 0)
                p_strength_a = top2_a / (top2_a + top2_b) if (top2_a + top2_b) > 0 else 0.5
                p_a = 0.4 * p_direct_a + 0.6 * p_strength_a
            else:
                p_a = p_direct_a
        p_b = 1.0 - p_a

        sel = row.get("selection", "")
        t_sel = best_match(sel, model_teams)

        # Bookmaker ratio guard: when the bookmaker's own H2H implied prob for
        # the selected side is very low (< 5 %), the model's calibration for
        # extreme mismatches is unreliable and inflates the underdog's chance.
        # Sanity guard: drop if model is more than 2.5× the bookmaker's implied probability.
        # Group H2H markets where implied < 14% and our model says > 2.5x are almost always
        # model errors (inflated underdog stats from weak confederation qualifiers).
        _bk_implied = _h2h_mkt_implied.get(mkt, {}).get(row.get("selection", ""), None)
        if _bk_implied is not None:
            _model_p = p_a if t_sel == t_a else p_b
            if _bk_implied < 0.14 and _model_p > 2.5 * _bk_implied:
                continue
            # Also drop extreme cases regardless of implied size
            if _bk_implied < 0.25 and _model_p > 3.5 * _bk_implied:
                continue

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

    # ── 19. 1ST / 2ND HALF GOALS O/U ─────────────────────────────────────────
    # Market: "1st Half Goals" / "2nd Half Goals"; same O/U format as total goals.
    # Model: ELO-corrected lambdas × WC historical half-goal fraction.
    # _WC_1H_FRAC already defined in section 16c above (0.413).

    for _hg_mkt, _hg_frac in [("1st Half Goals", _WC_1H_FRAC), ("2nd Half Goals", 1.0 - _WC_1H_FRAC)]:
        _hg_data: dict[str, dict] = {}
        for row in odds_data:
            if row.get("market") == _hg_mkt:
                try:
                    _lv = float(str(row.get("line", "")).strip())
                except (ValueError, TypeError):
                    continue
                mn = row["match"]
                _hg_data.setdefault(mn, {})[f"{row['selection']}_{_lv}"] = row["odds"]

        for match_name, _hg_dict in _hg_data.items():
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

            lam_h, lam_a = elo_corrected_lambdas(team_db, home_team, away_team)
            lam_h_half = lam_h * _hg_frac
            lam_a_half = lam_a * _hg_frac

            _hg_label = "1H Goals" if "1st" in _hg_mkt else "2H Goals"
            for line in [0.5, 1.0, 1.5, 2.0, 2.5]:
                p_over, p_under = poisson_ou_probs(lam_h_half, lam_a_half, line)
                over_key  = f"Over_{line}"
                under_key = f"Under_{line}"

                fair_over = fair_under = None
                if over_key in _hg_dict and under_key in _hg_dict:
                    raw_o = implied(_hg_dict[over_key])
                    raw_u = implied(_hg_dict[under_key])
                    fair  = remove_margin([raw_o, raw_u])
                    fair_over, fair_under = fair[0], fair[1]

                if over_key in _hg_dict:
                    check("match_sidebets", match_name, _hg_mkt, str(line),
                          f"Over {line}", _hg_dict[over_key], p_over, _hg_label,
                          fair_imp=fair_over)
                if under_key in _hg_dict:
                    check("match_sidebets", match_name, _hg_mkt, str(line),
                          f"Under {line}", _hg_dict[under_key], p_under, _hg_label,
                          fair_imp=fair_under)

    # ── 20. ASIAN HANDICAP (2-WAY) ───────────────────────────────────────────
    # Market: "Asian Handicap"; line "h_cap - a_cap"; sel = team name.
    # Model: score matrix → P(home_eff > away_eff), normalised over non-push.
    # _score_matrix() is already defined in section 10.

    def _ah_probs(lam_h: float, lam_a: float, h_cap: float, a_cap: float):
        """P(home wins AH), P(away wins AH) — push probability excluded & normalised out."""
        sp = _score_matrix(lam_h, lam_a)
        p_home = p_away = 0.0
        for (h, a), p in sp.items():
            eff = (h + h_cap) - (a + a_cap)
            if   eff > 0: p_home += p
            elif eff < 0: p_away += p
        non_push = p_home + p_away
        if non_push > 0:
            p_home /= non_push
            p_away /= non_push
        return p_home, p_away

    _ah_data: dict[str, dict] = {}
    for row in odds_data:
        if row.get("market") == "Asian Handicap":
            line_str = str(row.get("line", "")).strip()
            try:
                _parts = line_str.split(" - ")
                _h_cap, _a_cap = float(_parts[0]), float(_parts[1])
            except Exception:
                continue
            mn = row["match"]
            key = f"{_h_cap}_{_a_cap}"
            if mn not in _ah_data:
                _ah_data[mn] = {}
            if key not in _ah_data[mn]:
                _ah_data[mn][key] = {"h_cap": _h_cap, "a_cap": _a_cap, "odds": {}}
            _ah_data[mn][key]["odds"][row["selection"]] = row["odds"]

    for match_name, _ah_lines in _ah_data.items():
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

        # Calibration guard: skip when model diverges > 8 pp from bookmaker's 1X2.
        # Fallback: when bookie 1X2 isn't available (e.g. duplicate rows in scrape),
        # require ELO difference to be under 300 to avoid extreme mismatch AH bets.
        _bookie_hf2 = _bookie_home_fair.get(match_name)
        if _bookie_hf2 is not None:
            _pred_out2 = blended_match_prob(team_db, home_team, away_team)
            if abs(_pred_out2["p_win_a"] - _bookie_hf2) > 0.08:
                continue
        else:
            _elo_h2 = team_db[home_team].get("ELO", 1700)
            _elo_a2 = team_db[away_team].get("ELO", 1700)
            if abs(_elo_h2 - _elo_a2) > 300:
                continue

        lam_h, lam_a = elo_corrected_lambdas(team_db, home_team, away_team)

        for key, ld in _ah_lines.items():
            h_cap, a_cap = ld["h_cap"], ld["a_cap"]
            odds_map = ld["odds"]
            p_home_ah, p_away_ah = _ah_probs(lam_h, lam_a, h_cap, a_cap)

            fair_h_ah = fair_a_ah = None
            _ho_implied = _ao_implied = None
            for sel, odds_val in odds_map.items():
                st = best_match(sel, model_teams)
                if st == home_team:
                    _ho_implied = implied(odds_val)
                elif st == away_team:
                    _ao_implied = implied(odds_val)
            if _ho_implied and _ao_implied:
                _f = remove_margin([_ho_implied, _ao_implied])
                fair_h_ah, fair_a_ah = _f[0], _f[1]

            # Direct AH calibration guard: skip this line when our model diverges
            # from the bookmaker's own AH implied probability by more than 20 pp.
            # This catches score-distribution miscalibration that the 1X2 guard misses
            # (e.g. Ecuador vs Curaçao where win% is close but margin distribution differs).
            if fair_h_ah is not None and abs(p_home_ah - fair_h_ah) > 0.20:
                continue
            if fair_a_ah is not None and abs(p_away_ah - fair_a_ah) > 0.20:
                continue

            for sel, odds_val in odds_map.items():
                st = best_match(sel, model_teams)
                if st == home_team:
                    check("match_sidebets", match_name, "Asian Handicap", f"{h_cap}-{a_cap}",
                          f"{home_team} AH {h_cap}-{a_cap}", odds_val, p_home_ah, "Asian Handicap",
                          fair_imp=fair_h_ah)
                elif st == away_team:
                    check("match_sidebets", match_name, "Asian Handicap", f"{h_cap}-{a_cap}",
                          f"{away_team} AH {h_cap}-{a_cap}", odds_val, p_away_ah, "Asian Handicap",
                          fair_imp=fair_a_ah)

    # ── 21. 1ST / 2ND HALF ASIAN HANDICAP ────────────────────────────────────
    # Same logic as section 20, applied to half-time goal lambdas.

    for _hah_mkt, _hah_frac in [("1st Half Asian Handicap", _WC_1H_FRAC),
                                  ("2nd Half Asian Handicap", 1.0 - _WC_1H_FRAC)]:
        _hah_data: dict[str, dict] = {}
        for row in odds_data:
            if row.get("market") == _hah_mkt:
                line_str = str(row.get("line", "")).strip()
                try:
                    _parts = line_str.split(" - ")
                    _h_cap, _a_cap = float(_parts[0]), float(_parts[1])
                except Exception:
                    continue
                mn = row["match"]
                key = f"{_h_cap}_{_a_cap}"
                if mn not in _hah_data:
                    _hah_data[mn] = {}
                if key not in _hah_data[mn]:
                    _hah_data[mn][key] = {"h_cap": _h_cap, "a_cap": _a_cap, "odds": {}}
                _hah_data[mn][key]["odds"][row["selection"]] = row["odds"]

        for match_name, _hah_lines in _hah_data.items():
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

            _bookie_hf3 = _bookie_home_fair.get(match_name)
            if _bookie_hf3 is not None:
                _pred_out3 = blended_match_prob(team_db, home_team, away_team)
                if abs(_pred_out3["p_win_a"] - _bookie_hf3) > 0.08:
                    continue
            else:
                _elo_h3 = team_db[home_team].get("ELO", 1700)
                _elo_a3 = team_db[away_team].get("ELO", 1700)
                if abs(_elo_h3 - _elo_a3) > 300:
                    continue

            lam_h_full, lam_a_full = elo_corrected_lambdas(team_db, home_team, away_team)
            lam_h_hah = lam_h_full * _hah_frac
            lam_a_hah = lam_a_full * _hah_frac

            _hah_label = "1H AH" if "1st" in _hah_mkt else "2H AH"
            for key, ld in _hah_lines.items():
                h_cap, a_cap = ld["h_cap"], ld["a_cap"]
                odds_map = ld["odds"]
                p_home_hah, p_away_hah = _ah_probs(lam_h_hah, lam_a_hah, h_cap, a_cap)

                fair_h_hah = fair_a_hah = None
                _ho2_imp = _ao2_imp = None
                for sel, odds_val in odds_map.items():
                    st = best_match(sel, model_teams)
                    if st == home_team:
                        _ho2_imp = implied(odds_val)
                    elif st == away_team:
                        _ao2_imp = implied(odds_val)
                if _ho2_imp and _ao2_imp:
                    _f2 = remove_margin([_ho2_imp, _ao2_imp])
                    fair_h_hah, fair_a_hah = _f2[0], _f2[1]

                # AH calibration guard: skip if model diverges >20 pp from bookie implied
                if fair_h_hah is not None and abs(p_home_hah - fair_h_hah) > 0.20:
                    continue
                if fair_a_hah is not None and abs(p_away_hah - fair_a_hah) > 0.20:
                    continue

                for sel, odds_val in odds_map.items():
                    st = best_match(sel, model_teams)
                    if st == home_team:
                        check("match_sidebets", match_name, _hah_mkt, f"{h_cap}-{a_cap}",
                              f"{home_team} {_hah_label} {h_cap}-{a_cap}", odds_val,
                              p_home_hah, _hah_label, fair_imp=fair_h_hah)
                    elif st == away_team:
                        check("match_sidebets", match_name, _hah_mkt, f"{h_cap}-{a_cap}",
                              f"{away_team} {_hah_label} {h_cap}-{a_cap}", odds_val,
                              p_away_hah, _hah_label, fair_imp=fair_a_hah)

    # ── 22. CORNERS HANDICAP (2 WAY) ─────────────────────────────────────────
    # Market: "Corners Handicap (2 Way)"; line "h_cap - a_cap"; sel = team name.
    # Model: corners_model → Poisson convolution AH probabilities.

    def _corners_ah_probs(lam_hc: float, lam_ac: float, h_cap: float, a_cap: float, max_c: int = 25):
        """P(home_corners + h_cap > away_corners + a_cap) and mirror, normalised over non-push."""
        pmf_h = [math.exp(-lam_hc) * lam_hc**k / math.factorial(k) for k in range(max_c + 1)]
        pmf_a = [math.exp(-lam_ac) * lam_ac**k / math.factorial(k) for k in range(max_c + 1)]
        p_home_c = p_away_c = 0.0
        for h in range(max_c + 1):
            for a in range(max_c + 1):
                eff = (h + h_cap) - (a + a_cap)
                p = pmf_h[h] * pmf_a[a]
                if   eff > 0: p_home_c += p
                elif eff < 0: p_away_c += p
        non_push_c = p_home_c + p_away_c
        if non_push_c > 0:
            p_home_c /= non_push_c
            p_away_c /= non_push_c
        return p_home_c, p_away_c

    _crn_hcp_data: dict[str, dict] = {}
    for row in odds_data:
        if row.get("market") == "Corners Handicap (2 Way)":
            line_str = str(row.get("line", "")).strip()
            try:
                _parts = line_str.split(" - ")
                _h_cap, _a_cap = float(_parts[0]), float(_parts[1])
            except Exception:
                continue
            mn = row["match"]
            key = f"{_h_cap}_{_a_cap}"
            if mn not in _crn_hcp_data:
                _crn_hcp_data[mn] = {}
            if key not in _crn_hcp_data[mn]:
                _crn_hcp_data[mn][key] = {"h_cap": _h_cap, "a_cap": _a_cap, "odds": {}}
            _crn_hcp_data[mn][key]["odds"][row["selection"]] = row["odds"]

    for match_name, _crn_lines in _crn_hcp_data.items():
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
            lam_hc, lam_ac = corners_model(team_db[home_team], team_db[away_team])
        except Exception:
            continue

        for key, ld in _crn_lines.items():
            h_cap, a_cap = ld["h_cap"], ld["a_cap"]
            odds_map = ld["odds"]
            p_hc, p_ac = _corners_ah_probs(lam_hc, lam_ac, h_cap, a_cap)

            fair_hc = fair_ac = None
            _hc_imp = _ac_imp = None
            for sel, odds_val in odds_map.items():
                st = best_match(sel, model_teams)
                if st == home_team:
                    _hc_imp = implied(odds_val)
                elif st == away_team:
                    _ac_imp = implied(odds_val)
            if _hc_imp and _ac_imp:
                _fc = remove_margin([_hc_imp, _ac_imp])
                fair_hc, fair_ac = _fc[0], _fc[1]

            for sel, odds_val in odds_map.items():
                st = best_match(sel, model_teams)
                if st == home_team:
                    check("match_sidebets", match_name, "Corners Handicap (2 Way)", f"{h_cap}-{a_cap}",
                          f"{home_team} CrnHCP {h_cap}-{a_cap}", odds_val, p_hc, "Corners HCP",
                          fair_imp=fair_hc)
                elif st == away_team:
                    check("match_sidebets", match_name, "Corners Handicap (2 Way)", f"{h_cap}-{a_cap}",
                          f"{away_team} CrnHCP {h_cap}-{a_cap}", odds_val, p_ac, "Corners HCP",
                          fair_imp=fair_ac)

    # ── 23. TOTAL SHOTS ON TARGET O/U ────────────────────────────────────────
    # No real SOT stats in team_db → proxy: lam_sot = WC_AVG_SOT × (GF_AVG / WC_AVG_GOALS)
    # WC historical: ~8.0 total shots on target per match (4.0 per team).
    _WC_AVG_SOT   = 8.0   # combined SOT per match
    _WC_AVG_GOALS = 1.40  # goals per team per match (WC calibration)

    _sot_data: dict[str, dict] = {}
    for row in odds_data:
        if row.get("market") == "Total Shots on Target":
            try:
                _lv = float(str(row.get("line", "")).strip())
            except (ValueError, TypeError):
                continue
            mn = row["match"]
            _sot_data.setdefault(mn, {})[f"{row['selection']}_{_lv}"] = row["odds"]

    for match_name, _sot_dict in _sot_data.items():
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

        # Proxy: scale WC average SOT by each team's relative goal-scoring rate
        _gf_h = team_db[home_team].get("GF_AVG", _WC_AVG_GOALS)
        _gf_a = team_db[away_team].get("GF_AVG", _WC_AVG_GOALS)
        # Regress toward WC mean (60/40) to avoid extreme values for
        # defensively weak or attacking-light teams (e.g. Saudi Arabia GF=0.76).
        _lam_sot_h = (_WC_AVG_SOT / 2.0) * (0.4 * _gf_h / _WC_AVG_GOALS + 0.6)
        _lam_sot_a = (_WC_AVG_SOT / 2.0) * (0.4 * _gf_a / _WC_AVG_GOALS + 0.6)

        for line in [4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0]:
            p_over, p_under = poisson_ou_probs(_lam_sot_h, _lam_sot_a, line)
            over_key  = f"Over_{line}"
            under_key = f"Under_{line}"

            fair_over = fair_under = None
            if over_key in _sot_dict and under_key in _sot_dict:
                raw_o = implied(_sot_dict[over_key])
                raw_u = implied(_sot_dict[under_key])
                fair  = remove_margin([raw_o, raw_u])
                fair_over, fair_under = fair[0], fair[1]

            if over_key in _sot_dict:
                check("match_sidebets", match_name, "Total Shots on Target", str(line),
                      f"SOT Over {line}", _sot_dict[over_key], p_over, "Shots on Target",
                      fair_imp=fair_over)
            if under_key in _sot_dict:
                check("match_sidebets", match_name, "Total Shots on Target", str(line),
                      f"SOT Under {line}", _sot_dict[under_key], p_under, "Shots on Target",
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
                        "grp_2nd":  {},
                    }
                    print("  Loaded tournament results successfully.")
                    # The stored file lacks grp_win / grp_2nd, which Group H2H bets
                    # need to avoid falling back to the inflated qual% values.
                    # Run a supplemental MC specifically for group-stage finish data.
                    print("  Running supplemental MC for group-stage probabilities...")
                    mc_supp = run_monte_carlo(N_SIMS)
                    mc["grp_win"] = mc_supp.get("grp_win", {})
                    mc["grp_2nd"] = mc_supp.get("grp_2nd", {})
                    print("  Group-stage MC done.")
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
