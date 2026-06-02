"""
Simulador completo del Mundial 2026.

Formato:
  - 12 grupos de 4 equipos (fase de grupos)
  - Los 2 primeros de cada grupo + 8 mejores terceros = 32 equipos
  - Ronda de 32, Octavos, Cuartos, Semifinales, Final + 3er puesto

Uso rápido:
    python tournament.py                 # 1 simulación completa
    python tournament.py --sims 10000    # Monte Carlo (10k simulaciones)
    python tournament.py --show-groups   # muestra los grupos
"""

import math
import random
import json
import os
import sys
import time
from collections import defaultdict

from src.models.stacked_predictor import (
    stack_predict, expected_goals,
    penalty_win_prob, HOST_NATIONS,
)
from src.models.predictor import VENUES

# ─────────────────────────────────────────────
#  Ensemble matchup cache
#  Populated by precompute_ensemble_matchups() before MC runs.
#  Maps (team_a_name, team_b_name) → (p_win_a, p_draw, p_win_b)
# ─────────────────────────────────────────────
_MATCHUP_CACHE: dict = {}


def precompute_ensemble_matchups(team_names: list) -> None:
    """
    Load the trained EnsemblePredictor and pre-compute win probabilities
    for every ordered pair from team_names. Populates _MATCHUP_CACHE so
    sim_match() can use ensemble probs without re-running inference each call.
    """
    global _MATCHUP_CACHE
    from src.models.ensemble_predictor import EnsemblePredictor
    ep = EnsemblePredictor()
    ep.load()
    _MATCHUP_CACHE = ep.precompute_matchups(team_names)

# ─────────────────────────────────────────────
#  Datos base de equipos (ELO, forma, stats)
# ─────────────────────────────────────────────
BASE_TEAM_DB = {
    "Argentina":    {"ELO": 2095, "FORMA": 2.2, "GF_AVG": 1.9, "GA_AVG": 0.7},
    "France":       {"ELO": 2095, "FORMA": 2.3, "GF_AVG": 2.1, "GA_AVG": 0.7},
    "Brazil":       {"ELO": 2067, "FORMA": 2.0, "GF_AVG": 1.8, "GA_AVG": 0.7},
    "England":      {"ELO": 2021, "FORMA": 1.9, "GF_AVG": 1.7, "GA_AVG": 0.9},
    "Spain":        {"ELO": 2165, "FORMA": 2.5, "GF_AVG": 2.2, "GA_AVG": 0.5},
    "Germany":      {"ELO": 1923, "FORMA": 1.8, "GF_AVG": 1.9, "GA_AVG": 1.1},
    "Portugal":     {"ELO": 1993, "FORMA": 1.9, "GF_AVG": 2.0, "GA_AVG": 1.0},
    "Netherlands":  {"ELO": 1976, "FORMA": 1.8, "GF_AVG": 1.7, "GA_AVG": 1.0},
    "Belgium":      {"ELO": 1934, "FORMA": 1.7, "GF_AVG": 1.6, "GA_AVG": 0.9},
    "Croatia":      {"ELO": 1901, "FORMA": 1.7, "GF_AVG": 1.4, "GA_AVG": 0.9},
    "Italy":        {"ELO": 1910, "FORMA": 1.7, "GF_AVG": 1.4, "GA_AVG": 0.8},
    "Morocco":      {"ELO": 1882, "FORMA": 1.7, "GF_AVG": 1.3, "GA_AVG": 0.7},
    "Japan":        {"ELO": 1883, "FORMA": 1.7, "GF_AVG": 1.5, "GA_AVG": 0.9},
    "USA":          {"ELO": 1721, "FORMA": 1.5, "GF_AVG": 1.4, "GA_AVG": 1.1},
    "Mexico":       {"ELO": 1860, "FORMA": 1.5, "GF_AVG": 1.5, "GA_AVG": 1.2},
    "Canada":       {"ELO": 1756, "FORMA": 1.4, "GF_AVG": 1.3, "GA_AVG": 1.1},
    "Colombia":     {"ELO": 1891, "FORMA": 1.8, "GF_AVG": 1.6, "GA_AVG": 0.9},
    "Uruguay":      {"ELO": 1894, "FORMA": 1.7, "GF_AVG": 1.5, "GA_AVG": 0.9},
    "Ecuador":      {"ELO": 1757, "FORMA": 1.5, "GF_AVG": 1.3, "GA_AVG": 1.1},
    "South Korea":  {"ELO": 1791, "FORMA": 1.5, "GF_AVG": 1.4, "GA_AVG": 1.0},
    "Senegal":      {"ELO": 1798, "FORMA": 1.6, "GF_AVG": 1.4, "GA_AVG": 1.0},
    "Switzerland":  {"ELO": 1720, "FORMA": 1.5, "GF_AVG": 1.3, "GA_AVG": 0.9},
    "Poland":       {"ELO": 1788, "FORMA": 1.4, "GF_AVG": 1.3, "GA_AVG": 1.1},
    "Serbia":       {"ELO": 1821, "FORMA": 1.5, "GF_AVG": 1.4, "GA_AVG": 1.1},
    "Australia":    {"ELO": 1765, "FORMA": 1.4, "GF_AVG": 1.2, "GA_AVG": 1.1},
    "Iran":         {"ELO": 1783, "FORMA": 1.5, "GF_AVG": 1.3, "GA_AVG": 1.0},
    "Austria":      {"ELO": 1748, "FORMA": 1.5, "GF_AVG": 1.4, "GA_AVG": 1.1},
    "Turkey":       {"ELO": 1762, "FORMA": 1.5, "GF_AVG": 1.4, "GA_AVG": 1.1},
    "Nigeria":      {"ELO": 1762, "FORMA": 1.5, "GF_AVG": 1.4, "GA_AVG": 1.1},
    "Cameroon":     {"ELO": 1716, "FORMA": 1.4, "GF_AVG": 1.2, "GA_AVG": 1.2},
    "Ghana":        {"ELO": 1698, "FORMA": 1.3, "GF_AVG": 1.1, "GA_AVG": 1.2},
    "Tunisia":      {"ELO": 1701, "FORMA": 1.3, "GF_AVG": 1.0, "GA_AVG": 1.1},
    "Algeria":      {"ELO": 1747, "FORMA": 1.4, "GF_AVG": 1.2, "GA_AVG": 1.0},
    "Ivory Coast":  {"ELO": 1757, "FORMA": 1.5, "GF_AVG": 1.3, "GA_AVG": 1.1},
    "Saudi Arabia": {"ELO": 1674, "FORMA": 1.3, "GF_AVG": 1.1, "GA_AVG": 1.2},
    "Iraq":         {"ELO": 1651, "FORMA": 1.2, "GF_AVG": 1.0, "GA_AVG": 1.2},
    "Indonesia":    {"ELO": 1583, "FORMA": 1.1, "GF_AVG": 0.9, "GA_AVG": 1.4},
    "Uzbekistan":   {"ELO": 1674, "FORMA": 1.3, "GF_AVG": 1.1, "GA_AVG": 1.2},
    "Ukraine":      {"ELO": 1804, "FORMA": 1.5, "GF_AVG": 1.3, "GA_AVG": 1.1},
    "Slovakia":     {"ELO": 1729, "FORMA": 1.4, "GF_AVG": 1.2, "GA_AVG": 1.1},
    "Romania":      {"ELO": 1713, "FORMA": 1.3, "GF_AVG": 1.1, "GA_AVG": 1.2},
    "Panama":       {"ELO": 1632, "FORMA": 1.2, "GF_AVG": 0.9, "GA_AVG": 1.3},
    "Honduras":     {"ELO": 1590, "FORMA": 1.1, "GF_AVG": 0.9, "GA_AVG": 1.3},
    "Jamaica":      {"ELO": 1598, "FORMA": 1.1, "GF_AVG": 0.9, "GA_AVG": 1.3},
    "Paraguay":     {"ELO": 1697, "FORMA": 1.3, "GF_AVG": 1.1, "GA_AVG": 1.2},
    "Costa Rica":   {"ELO": 1672, "FORMA": 1.2, "GF_AVG": 1.0, "GA_AVG": 1.2},
    "Bolivia":      {"ELO": 1573, "FORMA": 1.0, "GF_AVG": 0.8, "GA_AVG": 1.4},
    "Venezuela":    {"ELO": 1634, "FORMA": 1.1, "GF_AVG": 0.9, "GA_AVG": 1.3},
    "New Zealand":  {"ELO": 1570, "FORMA": 1.0, "GF_AVG": 0.8, "GA_AVG": 1.4},
    "El Salvador":  {"ELO": 1541, "FORMA": 0.9, "GF_AVG": 0.7, "GA_AVG": 1.5},
    "Guatemala":    {"ELO": 1559, "FORMA": 1.0, "GF_AVG": 0.8, "GA_AVG": 1.4},
    "Egypt":        {"ELO": 1694, "FORMA": 1.3, "GF_AVG": 1.1, "GA_AVG": 1.1},
    "South Africa": {"ELO": 1610, "FORMA": 1.3, "GF_AVG": 1.0, "GA_AVG": 1.2},
    "Czech Republic": {"ELO": 1750, "FORMA": 1.5, "GF_AVG": 1.3, "GA_AVG": 1.0},
    "Bosnia & Herzegovina": {"ELO": 1690, "FORMA": 1.4, "GF_AVG": 1.2, "GA_AVG": 1.1},
    "Qatar":        {"ELO": 1550, "FORMA": 1.1, "GF_AVG": 0.9, "GA_AVG": 1.3},
    "Scotland":     {"ELO": 1760, "FORMA": 1.5, "GF_AVG": 1.3, "GA_AVG": 1.0},
    "Haiti":        {"ELO": 1490, "FORMA": 1.0, "GF_AVG": 0.8, "GA_AVG": 1.4},
    "Curaçao":      {"ELO": 1430, "FORMA": 0.9, "GF_AVG": 0.7, "GA_AVG": 1.4},
    "Sweden":       {"ELO": 1820, "FORMA": 1.6, "GF_AVG": 1.5, "GA_AVG": 0.9},
    "Norway":       {"ELO": 1750, "FORMA": 1.4, "GF_AVG": 1.5, "GA_AVG": 1.1},
    "Cape Verde":   {"ELO": 1560, "FORMA": 1.2, "GF_AVG": 0.9, "GA_AVG": 1.1},
    "DR Congo":     {"ELO": 1630, "FORMA": 1.3, "GF_AVG": 1.0, "GA_AVG": 1.2},
    "Jordan":       {"ELO": 1565, "FORMA": 1.2, "GF_AVG": 0.9, "GA_AVG": 1.2},
}

GROUPS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "wc2026_groups.json")
CACHE_STATS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "team_stats.json")
INJURIES_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "injuries.json")


# ─────────────────────────────────────────────
#  Carga de datos
# ─────────────────────────────────────────────
def load_team_db() -> dict:
    """Carga la base de datos de equipos combinando defaults + caché + lesiones."""
    db = {name: dict(data) | {"name": name} for name, data in BASE_TEAM_DB.items()}

    # Load SOS-adjusted live stats and blend with calibrated BASE values.
    # stats_client now returns SOS-adjusted FORMA/GF/GA (opponent-quality corrected),
    # so we can trust the live signal more — use 50/50 blend vs the old 40/60.
    # SOS adjustment already deflates Argentina's easy-qualifier stats and
    # upgrades Spain/France's elite UEFA opposition stats.
    _LIVE_WEIGHT = 0.50
    _BASE_WEIGHT = 0.50
    try:
        with open(CACHE_STATS) as f:
            cache = json.load(f)
        for team, entry in cache.items():
            if team in db and "data" in entry:
                d = entry["data"]
                base = BASE_TEAM_DB.get(team, {})

                # Prefer SOS-adjusted values if available; fall back to raw
                live_forma = d.get("FORMA",  db[team]["FORMA"])
                live_gf    = d.get("GF_AVG", db[team]["GF_AVG"])
                live_ga    = d.get("GA_AVG", db[team]["GA_AVG"])
                base_forma = base.get("FORMA",  db[team]["FORMA"])
                base_gf    = base.get("GF_AVG", db[team]["GF_AVG"])
                base_ga    = base.get("GA_AVG", db[team]["GA_AVG"])

                # Blend SOS-adjusted live data with calibrated BASE, then clamp
                db[team]["FORMA"]  = round(max(1.0, min(2.6,
                    _LIVE_WEIGHT * live_forma + _BASE_WEIGHT * base_forma)), 2)
                db[team]["GF_AVG"] = round(max(0.4, min(3.5,
                    _LIVE_WEIGHT * live_gf   + _BASE_WEIGHT * base_gf)),   2)
                db[team]["GA_AVG"] = round(max(0.3, min(2.5,
                    _LIVE_WEIGHT * live_ga   + _BASE_WEIGHT * base_ga)),   2)

                # Extended stats — use live value directly if available (no BASE baseline)
                _EXT_DEFAULTS = {
                    "CORNERS_FOR": 5.0, "CORNERS_AGAINST": 5.0,
                    "SOT_FOR": 4.0,     "SOT_AGAINST": 4.0,
                    "POSSESSION": 50.0, "YELLOW_CARDS": 1.8,
                    "XG_FOR": None,     "XG_AGAINST": None,
                }
                for stat, default in _EXT_DEFAULTS.items():
                    val = d.get(stat)
                    if val is not None:
                        db[team][stat] = val
                    elif stat not in db[team] and default is not None:
                        db[team][stat] = default

                # Store SOS diagnostics for transparency
                if "SOS_RATIO" in d:
                    db[team]["SOS_RATIO"]   = d["SOS_RATIO"]
                    db[team]["AVG_OPP_ELO"] = d["AVG_OPP_ELO"]
    except (FileNotFoundError, KeyError):
        pass

    # Cargar ELOs de caché (más actualizados)
    elos_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "elos.json")
    try:
        with open(elos_file) as f:
            elo_cache = json.load(f)
        for team, elo in elo_cache.get("data", {}).items():
            if team in db:
                db[team]["ELO"] = elo
    except (FileNotFoundError, KeyError):
        pass

    # Lesiones
    try:
        with open(INJURIES_FILE) as f:
            injuries = json.load(f)
        for team, entry in injuries.items():
            if team in db and not team.startswith("_"):
                db[team]["INJURIES"] = entry.get("count", 0)
    except (FileNotFoundError, KeyError):
        pass

    # Asegurar INJURIES en todos
    for team in db:
        db[team].setdefault("INJURIES", 0)

    return db


def load_groups() -> dict:
    try:
        with open(GROUPS_FILE) as f:
            raw = json.load(f)
        return {k: v for k, v in raw.items() if not k.startswith("_")}
    except FileNotFoundError:
        print("⚠ data/wc2026_groups.json no encontrado. Usando grupos de ejemplo.")
        return {}


# ─────────────────────────────────────────────
#  Simulación de un partido
# ─────────────────────────────────────────────
def _sample_poisson(lam: float) -> int:
    L = math.exp(-lam)
    k, p = 0, 1.0
    while p > L:
        k += 1
        p *= random.random()
    return k - 1


def sim_match(
    team_a: dict,
    team_b: dict,
    knockout: bool = False,
    home_team: str = None,
    round_number: int = 1,
    fatigue_a: bool = False,
    fatigue_b: bool = False,
    pens_a: bool = False,
    pens_b: bool = False,
    elo_ratings: dict = None,
    venue_name: str = None,
) -> dict:
    """
    Simula un partido y devuelve {gf_a, gf_b, winner_name, went_to_et, went_to_pens}.

    En eliminatorias, si hay empate va a penales usando penalty_win_prob.

    Params:
        home_team:    nombre del equipo local (para ventaja de sede, solo grupos)
        round_number: ronda actual (1=grupos, 2=R32, ...) para fatiga por edad
        fatigue_a/b:  si el equipo viene de tiempo extra en la ronda anterior
        pens_a/b:     si el equipo viene de penales en la ronda anterior
        elo_ratings:  dict mutable de ELOs en vivo; si se pasa, se usa en vez de team["ELO"]
    """
    # Apply live ELO overrides if provided
    if elo_ratings is not None:
        name_a = team_a.get("name", "")
        name_b = team_b.get("name", "")
        if name_a in elo_ratings:
            team_a = dict(team_a)
            team_a["ELO"] = elo_ratings[name_a]
        if name_b in elo_ratings:
            team_b = dict(team_b)
            team_b["ELO"] = elo_ratings[name_b]

    # Win probabilities: use pre-computed ensemble if available, else stack_predict
    _key = (team_a.get("name", ""), team_b.get("name", ""))
    if _key in _MATCHUP_CACHE:
        _pa, _pd, _pb = _MATCHUP_CACHE[_key]
        pred = {"p_win_a": _pa, "p_draw": _pd, "p_win_b": _pb}
    else:
        pred = stack_predict(team_a, team_b, home_team=home_team, round_number=round_number, venue_name=venue_name)

    lam_a, lam_b = expected_goals(team_a, team_b, home_team=home_team, round_number=round_number, venue_name=venue_name)

    # Apply fatigue penalties from prior extra time / penalties
    lam_a *= (1 - 0.06 * fatigue_a - 0.03 * pens_a)
    lam_b *= (1 - 0.06 * fatigue_b - 0.03 * pens_b)
    lam_a = max(0.15, lam_a)
    lam_b = max(0.15, lam_b)

    # Sample goals from Poisson and let the scoreline determine the outcome.
    # This keeps scores consistent with the Poisson lambdas (no truncation bias).
    # The stacked model probabilities are used as a light correction via
    # rejection sampling: if outcome disagrees with stacked model, accept with
    # probability proportional to stacked/poisson ratio (capped for stability).
    from src.models.stacked_predictor import _poisson_3way
    poi_pa, poi_pd, poi_pb = _poisson_3way(lam_a, lam_b)

    # Sample until we get a valid scoreline (max 5 attempts, then accept as-is)
    for _attempt in range(5):
        gf_a = _sample_poisson(lam_a)
        gf_b = _sample_poisson(lam_b)

        if gf_a > gf_b:
            poi_p = poi_pa
            stack_p = pred["p_win_a"]
        elif gf_a == gf_b:
            poi_p = poi_pd
            stack_p = pred["p_draw"]
        else:
            poi_p = poi_pb
            stack_p = pred["p_win_b"]

        # Accept with probability proportional to stacked/poisson ratio
        # (capped at 2.0 to prevent extreme rejection rates)
        if poi_p > 0:
            accept_ratio = min(2.0, stack_p / poi_p)
        else:
            accept_ratio = 1.0
        if random.random() < accept_ratio:
            break

    if gf_a > gf_b:
        winner = team_a["name"]
    elif gf_b > gf_a:
        winner = team_b["name"]
    else:
        winner = None  # Empate

    went_to_et = False
    went_to_pens = False

    # En eliminatoria: desempatar con tiempo extra, luego penales si sigue empate
    if knockout and winner is None:
        went_to_et = True
        name_a = team_a["name"]
        name_b = team_b["name"]

        # Extra time: ~30 min at reduced intensity (~0.35× of 90-min λ).
        # Historically ~30% of ET periods produce a goal.
        et_lam_a = lam_a * 0.35
        et_lam_b = lam_b * 0.35
        et_gf_a = _sample_poisson(et_lam_a)
        et_gf_b = _sample_poisson(et_lam_b)
        gf_a += et_gf_a
        gf_b += et_gf_b

        if gf_a > gf_b:
            winner = name_a
        elif gf_b > gf_a:
            winner = name_b
        else:
            # Still tied after ET — go to penalties
            p_pen_a = penalty_win_prob(name_a, name_b)
            if random.random() < p_pen_a:
                winner = name_a
            else:
                winner = name_b
            went_to_pens = True

    # Bayesian ELO update
    if elo_ratings is not None:
        na = team_a["name"]
        nb = team_b["name"]
        elo_a_live = elo_ratings.get(na, team_a["ELO"])
        elo_b_live = elo_ratings.get(nb, team_b["ELO"])
        K = 25 if knockout else 20
        expected_a = 1 / (1 + 10 ** ((elo_b_live - elo_a_live) / 400))
        if knockout and went_to_pens:
            # In penalty-decided knockouts, award the win to the actual winner
            # but with reduced K (penalty outcomes are somewhat random)
            result_a = 1.0 if winner == na else 0.0
            K = K * 0.6  # Reduced weight — pens are partially luck
        else:
            result_a = 1 if gf_a > gf_b else (0.5 if gf_a == gf_b else 0)
        delta = K * (result_a - expected_a)
        elo_ratings[na] = elo_a_live + delta
        elo_ratings[nb] = elo_b_live - delta

    return {
        "gf_a": gf_a,
        "gf_b": gf_b,
        "winner": winner,
        "team_a": team_a["name"],
        "team_b": team_b["name"],
        "went_to_et": went_to_et,
        "went_to_pens": went_to_pens,
    }


# ─────────────────────────────────────────────
#  Fase de grupos
# ─────────────────────────────────────────────
def sim_group(
    group_teams: list,
    team_db: dict,
    group_letter: str = "A",
    elo_ratings: dict = None,
    home_teams: set = None,
) -> tuple[list, dict]:
    """
    Simula todos los partidos de un grupo (round-robin).
    Retorna (clasificación ordenada, estadísticas por equipo, total_goals, over_2_5).

    home_teams: set of team names that are host nations (for ELO/goals boost)
    """
    if home_teams is None:
        home_teams = HOST_NATIONS

    # Mapeo aproximado de sedes por grupo para el Mundial 2026
    GROUP_VENUES = {
        "A": ["Ciudad de México", "Guadalajara", "Monterrey"],
        "B": ["Vancouver", "Toronto", "Seattle"],
        "C": ["Los Angeles", "San Francisco", "Seattle"],
        "D": ["San Francisco", "Los Angeles"],
        "E": ["Houston", "Dallas"],
        "F": ["Kansas City", "Dallas"],
        "G": ["Atlanta", "Miami"],
        "H": ["Miami", "Atlanta"],
        "I": ["Boston", "Philadelphia"],
        "J": ["Philadelphia", "New York / NJ"],
        "K": ["New York / NJ", "Boston"],
        "L": ["Toronto", "New York / NJ"]
    }

    stats = {t: {"pts": 0, "gd": 0, "gf": 0, "ga": 0, "w": 0, "d": 0, "l": 0}
             for t in group_teams}

    total_goals = 0
    over_2_5 = 0

    for i in range(len(group_teams)):
        for j in range(i + 1, len(group_teams)):
            ta = team_db.get(group_teams[i], {"name": group_teams[i], "ELO": 1600, "FORMA": 1.2, "GF_AVG": 1.0, "GA_AVG": 1.2, "INJURIES": 0})
            tb = team_db.get(group_teams[j], {"name": group_teams[j], "ELO": 1600, "FORMA": 1.2, "GF_AVG": 1.0, "GA_AVG": 1.2, "INJURIES": 0})
            ta["name"] = group_teams[i]
            tb["name"] = group_teams[j]

            # Determine home team for this match (only host nations get boost)
            match_home = None
            if group_teams[i] in home_teams:
                match_home = group_teams[i]
            elif group_teams[j] in home_teams:
                match_home = group_teams[j]

            venues_for_group = GROUP_VENUES.get(group_letter, ["New York / NJ"])
            # Use deterministic venue from the group's allowed venues
            match_venue = venues_for_group[(i + j) % len(venues_for_group)]

            m = sim_match(
                ta, tb,
                knockout=False,
                home_team=match_home,
                round_number=1,
                elo_ratings=elo_ratings,
                venue_name=match_venue,
            )

            for team, gf, ga in [
                (group_teams[i], m["gf_a"], m["gf_b"]),
                (group_teams[j], m["gf_b"], m["gf_a"]),
            ]:
                stats[team]["gf"] += gf
                stats[team]["ga"] += ga
                stats[team]["gd"] += gf - ga

            if m["winner"] == group_teams[i]:
                stats[group_teams[i]]["pts"] += 3
                stats[group_teams[i]]["w"] += 1
                stats[group_teams[j]]["l"] += 1
            elif m["winner"] == group_teams[j]:
                stats[group_teams[j]]["pts"] += 3
                stats[group_teams[j]]["w"] += 1
                stats[group_teams[i]]["l"] += 1
            else:
                stats[group_teams[i]]["pts"] += 1
                stats[group_teams[j]]["pts"] += 1
                stats[group_teams[i]]["d"] += 1
                stats[group_teams[j]]["d"] += 1

            tg = m["gf_a"] + m["gf_b"]
            total_goals += tg
            if tg > 2.5:
                over_2_5 += 1

    ranking = sorted(
        group_teams,
        key=lambda t: (-stats[t]["pts"], -stats[t]["gd"], -stats[t]["gf"], random.random()),
    )
    return ranking, stats, total_goals, over_2_5


def sim_group_stage(groups: dict, team_db: dict, elo_ratings: dict = None) -> dict:
    """
    Simula toda la fase de grupos.
    Retorna (results_dict, best_thirds_list)
    """
    results = {}
    all_thirds = []
    stage_goals = 0
    stage_over = 0

    for letter, teams in groups.items():
        ranking, stats, t_goals, o_goals = sim_group(teams, team_db, group_letter=letter, elo_ratings=elo_ratings)
        stage_goals += t_goals
        stage_over += o_goals
        results[letter] = {
            "ranking": ranking,
            "stats": stats,
            "first":  ranking[0],
            "second": ranking[1],
            "third":  ranking[2],
        }
        # Guardamos el 3ro con sus stats para seleccionar los mejores
        all_thirds.append({
            "group": letter,
            "team":  ranking[2],
            "pts":   stats[ranking[2]]["pts"],
            "gd":    stats[ranking[2]]["gd"],
            "gf":    stats[ranking[2]]["gf"],
        })

    # 8 mejores terceros
    best_thirds = sorted(
        all_thirds,
        key=lambda x: (-x["pts"], -x["gd"], -x["gf"], random.random()),
    )[:8]

    return results, [t["team"] for t in best_thirds], stage_goals, stage_over


# ─────────────────────────────────────────────
#  Fase eliminatoria
# ─────────────────────────────────────────────
def sim_knockout_round(
    matches: list[tuple],
    team_db: dict,
    round_name: str,
    silent: bool = False,
    round_number: int = 2,
    fatigue: dict = None,
    elo_ratings: dict = None,
) -> tuple[list, dict]:
    """
    Simula una ronda eliminatoria.
    matches: [(team_a_name, team_b_name), ...]
    Retorna (lista de ganadores, nuevo fatigue dict para la siguiente ronda, ko_goals, ko_over_2_5).

    fatigue: dict mapping team_name -> {"et": bool, "pens": bool}
             Fatigue resets at the start of each new round (between rounds = ~4 days rest).
    """
    if fatigue is None:
        fatigue = {}

    winners = []
    ko_goals = 0
    ko_over = 0
    # Fatigue for the NEXT round (reset from prior round — rest between rounds)
    next_fatigue = {}

    for idx, (ta_name, tb_name) in enumerate(matches):
        ta = team_db.get(ta_name, {"name": ta_name, "ELO": 1600, "FORMA": 1.2, "GF_AVG": 1.0, "GA_AVG": 1.2, "INJURIES": 0})
        tb = team_db.get(tb_name, {"name": tb_name, "ELO": 1600, "FORMA": 1.2, "GF_AVG": 1.0, "GA_AVG": 1.2, "INJURIES": 0})
        ta["name"], tb["name"] = ta_name, tb_name

        # Get fatigue state for this match (from PREVIOUS round — already rested, so reset)
        # Within a round there's no back-to-back; fatigue carries from prior round only
        f_a = fatigue.get(ta_name, {"et": False, "pens": False})
        f_b = fatigue.get(tb_name, {"et": False, "pens": False})

        # Sedes deterministas y realistas para fase eliminatoria
        if round_name == "Final":
            match_venue = "New York / NJ"
        elif round_name == "3er Puesto":
            match_venue = "Miami"
        elif round_name == "Semifinales":
            match_venue = ["Dallas", "Atlanta"][idx % 2]
        elif round_name == "Cuartos de Final":
            match_venue = ["Los Angeles", "Kansas City", "Miami", "Boston"][idx % 4]
        else:
            venue_names = list(VENUES.keys())
            match_venue = venue_names[idx % len(venue_names)]

        m = sim_match(
            ta, tb,
            knockout=True,
            home_team=None,  # No home advantage in knockouts
            round_number=round_number,
            fatigue_a=f_a["et"],
            fatigue_b=f_b["et"],
            pens_a=f_a["pens"],
            pens_b=f_b["pens"],
            elo_ratings=elo_ratings,
            venue_name=match_venue,
        )
        winners.append(m["winner"])

        tg = m["gf_a"] + m["gf_b"]
        ko_goals += tg
        if tg > 2.5:
            ko_over += 1

        # NOTE: We intentionally do NOT write elo_ratings back into team_db here.
        # team_db is shared across all Monte Carlo iterations — writing back would
        # accumulate ELO changes across thousands of simulations, inflating values to ~15,000.
        # The elo_ratings local dict handles within-simulation live updates only.

        # Track fatigue for next round: both teams from ET, winner from pens too
        if m["went_to_et"]:
            next_fatigue[ta_name] = {"et": True, "pens": m["went_to_pens"]}
            next_fatigue[tb_name] = {"et": True, "pens": m["went_to_pens"]}
        else:
            next_fatigue[ta_name] = {"et": False, "pens": False}
            next_fatigue[tb_name] = {"et": False, "pens": False}

        if not silent:
            score = f"{m['gf_a']}-{m['gf_b']}"
            pen_marker = " (pens)" if m["went_to_pens"] else (" (ET)" if m["went_to_et"] else "")
            print(f"    {ta_name:22s} {score} {tb_name}{pen_marker}")

    return winners, next_fatigue, ko_goals, ko_over


def sim_full_tournament(
    groups: dict,
    team_db: dict,
    silent: bool = False,
    fatigue: dict = None,
) -> dict:
    """
    Simula el torneo completo: grupos → eliminatorias.
    Retorna {champion, finalist, semifinalists, ...}

    fatigue: optional initial fatigue dict (usually empty {} or None)
    """
    if fatigue is None:
        fatigue = {}

    # Initialize live ELO ratings from team_db
    elo_ratings = {}
    for name, data in team_db.items():
        elo_ratings[name] = data.get("ELO", 1600)

    if not silent:
        print("  Fase de Grupos...")
    group_results, best_thirds, t_goals, t_over = sim_group_stage(groups, team_db, elo_ratings=elo_ratings)

    # After group stage, reset fatigue (teams had rest before knockouts)
    current_fatigue = {}

    if len(groups) != 12:
        raise ValueError(f"Official bracket expects 12 groups, got {len(groups)}.")

    # ── Build official FIFA WC 2026 R32 bracket ──────────────────────────
    winners = {g: group_results[g]["first"]  for g in groups}
    runners = {g: group_results[g]["second"] for g in groups}

    # Map each 3rd-place team to its source group letter
    third_to_group = {
        group_results[g]["ranking"][2]: g
        for g in groups if len(group_results[g]["ranking"]) >= 3
    }
    best_thirds_with_groups = [(t, third_to_group.get(t, "?")) for t in best_thirds]

    # Assign the 8 best-3rd teams to the 8 official slots using backtracking.
    # Slot constraints = eligible source groups per match (per official draw rules).
    _THIRDS_SLOTS = [
        ("M74", frozenset("ABCDF")),   # 1E vs best-3rd(A|B|C|D|F)
        ("M77", frozenset("CDFGH")),   # 1I vs best-3rd(C|D|F|G|H)
        ("M79", frozenset("CEFHI")),   # 1A vs best-3rd(C|E|F|H|I)
        ("M80", frozenset("EHIJK")),   # 1L vs best-3rd(E|H|I|J|K)
        ("M81", frozenset("BEFIJ")),   # 1D vs best-3rd(B|E|F|I|J)
        ("M82", frozenset("AEHIJ")),   # 1G vs best-3rd(A|E|H|I|J)
        ("M85", frozenset("EFGIJ")),   # 1B vs best-3rd(E|F|G|I|J)
        ("M87", frozenset("DEIJL")),   # 1K vs best-3rd(D|E|I|J|L)
    ]

    def _backtrack_thirds(slot_idx, remaining):
        if slot_idx == len(_THIRDS_SLOTS):
            return {}
        slot_id, eligible = _THIRDS_SLOTS[slot_idx]
        for i, (team, grp) in enumerate(remaining):
            if grp in eligible:
                rest = remaining[:i] + remaining[i + 1:]
                sub = _backtrack_thirds(slot_idx + 1, rest)
                if sub is not None:
                    return {slot_id: team, **sub}
        return None

    thirds = _backtrack_thirds(0, best_thirds_with_groups)
    if thirds is None:
        # Fallback: ignore group constraints, assign best-to-worst in slot order
        thirds = {}
        remaining = list(best_thirds_with_groups)
        for slot_id, _ in _THIRDS_SLOTS:
            thirds[slot_id] = remaining.pop(0)[0] if remaining else "TBD"

    # Official R32 pairings (M73–M88 in bracket order)
    r32_pairs = [
        (runners["A"],  runners["B"]),     # M73: 2A vs 2B
        (winners["E"],  thirds["M74"]),    # M74: 1E vs best-3rd(A|B|C|D|F)
        (winners["F"],  runners["C"]),     # M75: 1F vs 2C
        (winners["C"],  runners["F"]),     # M76: 1C vs 2F
        (winners["I"],  thirds["M77"]),    # M77: 1I vs best-3rd(C|D|F|G|H)
        (runners["E"],  runners["I"]),     # M78: 2E vs 2I
        (winners["A"],  thirds["M79"]),    # M79: 1A vs best-3rd(C|E|F|H|I)
        (winners["L"],  thirds["M80"]),    # M80: 1L vs best-3rd(E|H|I|J|K)
        (winners["D"],  thirds["M81"]),    # M81: 1D vs best-3rd(B|E|F|I|J)
        (winners["G"],  thirds["M82"]),    # M82: 1G vs best-3rd(A|E|H|I|J)
        (runners["K"],  runners["L"]),     # M83: 2K vs 2L
        (winners["H"],  runners["J"]),     # M84: 1H vs 2J
        (winners["B"],  thirds["M85"]),    # M85: 1B vs best-3rd(E|F|G|I|J)
        (winners["J"],  runners["H"]),     # M86: 1J vs 2H
        (winners["K"],  thirds["M87"]),    # M87: 1K vs best-3rd(D|E|I|J|L)
        (runners["D"],  runners["G"]),     # M88: 2D vs 2G
    ]
    # R16: W(M73,M74) → M89, W(M75,M76) → M90, ... sequential pairs
    # QF:  W(M89,M90) → QF1, W(M91,M92) → QF2, W(M93,M94) → QF3, W(M95,M96) → QF4
    # SF:  W(QF1,QF2) → SF1, W(QF3,QF4) → SF2

    # 16 partidos = 32 equipos únicos ✓
    rounds = [
        ("Ronda de 32",      r32_pairs, 32, 2),
        ("Octavos de Final", None,      16, 3),
        ("Cuartos de Final", None,       8, 4),
        ("Semifinales",      None,       4, 5),
    ]

    current_teams = None
    results_by_round = {}
    sf_participants  = []
    qf_participants  = []
    r32_participants = [t for pair in r32_pairs for t in pair]

    for rname, pairs, _, rnum in rounds:
        if pairs is None:
            pairs = [(current_teams[i], current_teams[i + 1])
                     for i in range(0, len(current_teams), 2)]

        if rname == "Cuartos de Final":
            qf_participants = list(current_teams)
        if rname == "Semifinales":
            sf_participants = list(current_teams)

        if not silent:
            print(f"\n  {rname}:")

        round_winners, current_fatigue, k_g, k_o = sim_knockout_round(
            pairs, team_db, rname, silent,
            round_number=rnum,
            fatigue=current_fatigue,
            elo_ratings=elo_ratings,
        )
        t_goals += k_g
        t_over += k_o
        results_by_round[rname] = round_winners
        current_teams = round_winners

    sf_winners = results_by_round["Semifinales"]                       # 2 ganadores
    sf_winners_set = set(sf_winners)
    sf_losers = [t for t in sf_participants if t not in sf_winners_set]  # 2 perdedores

    # 3er puesto (round_number=6)
    if not silent:
        print("\n  Tercer Puesto:")
    third_winners, _, k_g, k_o = sim_knockout_round(
        [(sf_losers[0], sf_losers[1])], team_db, "3er Puesto", silent,
        round_number=6,
        fatigue=current_fatigue,
        elo_ratings=elo_ratings,
    )
    t_goals += k_g
    t_over += k_o
    third_winner = third_winners[0]

    # Final (round_number=6)
    if not silent:
        print("\n  Final:")
    final_pair = [(sf_winners[0], sf_winners[1])]
    final_winners, _, k_g, k_o = sim_knockout_round(
        final_pair, team_db, "Final", silent,
        round_number=6,
        fatigue=current_fatigue,
        elo_ratings=elo_ratings,
    )
    t_goals += k_g
    t_over += k_o
    champion = final_winners[0]
    finalist = sf_winners[1] if champion == sf_winners[0] else sf_winners[0]

    return {
        "champion":         champion,
        "finalist":         finalist,
        "third":            third_winner,
        "quarterfinalists": qf_participants,
        "semifinalists":    sf_participants,
        "r32_participants": r32_participants,
        "group_results":    {g: r["ranking"] for g, r in group_results.items()},
        "best_thirds":      best_thirds,
        "total_goals":      t_goals,
        "over_2_5":         t_over,
        "results_by_round": results_by_round,   # winners per round, in bracket slot order
    }


# ─────────────────────────────────────────────
#  Monte Carlo
# ─────────────────────────────────────────────
def monte_carlo(groups: dict, team_db: dict, n: int = 10_000) -> dict:
    """
    Ejecuta n simulaciones y acumula estadísticas de probabilidad.
    Retorna probabilidades de campeonar, llegar a final, semifinales, etc.
    """
    wins = defaultdict(int)
    finals = defaultdict(int)
    semis = defaultdict(int)
    quarters = defaultdict(int)
    qualifications = defaultdict(int)
    thirds_place = defaultdict(int)
    
    total_goals_sum = 0
    total_over_2_5 = 0

    # Pre-compute ensemble matchup cache once for all MC iterations
    all_team_names = list({t for g in groups.values() if isinstance(g, list) for t in g})
    precompute_ensemble_matchups(all_team_names)

    t0 = time.time()
    for i in range(n):
        r = sim_full_tournament(groups, team_db, silent=True)
        wins[r["champion"]] += 1
        finals[r["finalist"]] += 1
        finals[r["champion"]] += 1
        for team in r["semifinalists"]:
            semis[team] += 1
        for team in r["quarterfinalists"]:
            quarters[team] += 1
        for team in r["r32_participants"]:
            qualifications[team] += 1
        thirds_place[r["third"]] += 1
        
        total_goals_sum += r["total_goals"]
        total_over_2_5 += r["over_2_5"]

        if (i + 1) % 1000 == 0:
            elapsed = time.time() - t0
            print(f"  Simulación {i+1}/{n} ({elapsed:.1f}s)...")

    print(f"\nTotal: {n} simulaciones en {time.time()-t0:.1f}s")

    all_teams = set(wins) | set(finals) | set(semis) | set(quarters)

    table = []
    for team in sorted(all_teams, key=lambda t: -wins[t]):
        table.append({
            "team":        team,
            "champion_%":  round(wins[team] / n * 100, 2),
            "final_%":     round(finals[team] / n * 100, 2),
            "semi_%":      round(semis[team] / n * 100, 2),
            "quarter_%":   round(quarters[team] / n * 100, 2),
            "qual_%":      round(qualifications[team] / n * 100, 2),
            "third_%":     round(thirds_place.get(team, 0) / n * 100, 2),
        })
        
    avg_goals = total_goals_sum / n
    avg_over_2_5 = total_over_2_5 / n
    # 104 total matches in the 48-team format
    pct_over_2_5 = (avg_over_2_5 / 104) * 100

    return {
        "table": table,
        "metrics": {
            "avg_goals": round(avg_goals, 1),
            "avg_over_2_5": round(avg_over_2_5, 1),
            "pct_over_2_5": round(pct_over_2_5, 1)
        }
    }


# ─────────────────────────────────────────────
#  CLI principal
# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
#  Narrative insights generator
# ─────────────────────────────────────────────

# Bracket half assignment (based on balanced_order in sim_full_tournament).
# Half A = matches 0-7 in the R32 bracket (left side of draw).
# Half B = matches 8-15 (right side of draw).
# Groups that seed into each half as GROUP WINNERS ("1" = group winner slot):
_BRACKET_HALF = {
    # Half A — potential QF/SF cluster
    "A": "A", "H": "A", "I": "A", "D": "A",
    "E": "A", "L": "A",
    # Half B — potential QF/SF cluster
    "B": "B", "G": "B", "J": "B", "C": "B",
    "F": "B", "K": "B",
}

# Known strongest rivals per bracket half (for narrative danger warnings)
_HALF_A_POWERS = ["Spain", "Brazil", "Germany", "Argentina", "Belgium"]
_HALF_B_POWERS = ["France", "England", "Netherlands", "Portugal", "Colombia"]


def _group_difficulty(group_teams: list, team_db: dict, own_team: str) -> tuple[str, float]:
    """Return (difficulty_label, avg_opponent_elo) for a team's group opponents."""
    elos = []
    for t in group_teams:
        if t != own_team:
            elo = team_db.get(t, {}).get("ELO", 1600)
            elos.append(elo)
    if not elos:
        return "Unknown", 1600.0
    avg = sum(elos) / len(elos)
    if avg >= 1900:
        label = "🔴 Very Hard"
    elif avg >= 1800:
        label = "🟠 Hard"
    elif avg >= 1720:
        label = "🟡 Medium"
    elif avg >= 1640:
        label = "🟢 Easy"
    else:
        label = "🟢 Very Easy"
    return label, avg


def _find_team_group(team: str, groups: dict) -> str | None:
    """Return the group letter a team belongs to."""
    for letter, members in groups.items():
        if team in members:
            return letter
    return None


def _elo_tier(elo: float) -> str:
    if elo >= 2100:  return "World-class (top 2)"
    if elo >= 2050:  return "Elite (top 5)"
    if elo >= 1980:  return "Top 10"
    if elo >= 1900:  return "Top 15"
    if elo >= 1820:  return "Top 20"
    return "Contender"


def _forma_label(forma: float) -> str:
    if forma >= 2.4:  return "exceptional"
    if forma >= 2.1:  return "very strong"
    if forma >= 1.8:  return "solid"
    if forma >= 1.5:  return "inconsistent"
    return "poor"


def generate_insights(table: list, team_db: dict, groups: dict, top_n: int = 8):
    """
    Print a plain-English narrative for each of the top_n predicted teams,
    covering: group difficulty, bracket half danger, ELO standing,
    form quality, SOS adjustment, and key risks.
    """
    print(f"\n{'='*65}")
    print(f"  📊  PREDICTION INSIGHTS — WHY EACH TEAM IS RANKED HERE")
    print(f"{'='*65}")

    # General Bracket Explanation
    print("\n  🔍 BRACKET STRUCTURE & ODDS EXPLANATION")
    print("  " + "-" * 50)
    print("  The World Cup 2026 bracket structure strongly dictates these odds.")
    print("  Based on projected group winners, the knockout stage splits into:")
    print("    ▶ HALF A (The 'Heavyweight' Side): Spain, Argentina, Brazil, Germany, Belgium.")
    print("      This half is highly congested with top-tier ELO teams. They will cannibalize")
    print("      each other in the Quarter-Finals and Semi-Finals, making it harder for any")
    print("      single team (except Spain, who is dominant) to reach the Final.")
    print("    ▶ HALF B (The 'Open' Side): France, England, Portugal, Netherlands.")
    print("      This half is slightly weaker at the absolute top end, creating a vacuum")
    print("      where France and England have a statistically easier path to the Final,")
    print("      inflating their overall tournament win percentages compared to teams in Half A.")
    print("\n  TEAM BREAKDOWN:")
    print("  " + "-" * 50)

    # Build a quick rank lookup
    rank_map = {row["team"]: i + 1 for i, row in enumerate(table)}

    for i, row in enumerate(table[:top_n]):
        team      = row["team"]
        champ_pct = row["champion_%"]
        final_pct = row["final_%"]
        semi_pct  = row["semi_%"]
        qtr_pct   = row["quarter_%"]

        td = team_db.get(team, {})
        elo   = td.get("ELO",    1700)
        forma = td.get("FORMA",  1.5)
        sos   = td.get("SOS_RATIO", None)
        opp_elo = td.get("AVG_OPP_ELO", None)

        group_letter = _find_team_group(team, groups)
        group_teams  = groups.get(group_letter, [])
        diff_label, avg_opp_elo = _group_difficulty(group_teams, team_db, team)
        bracket_half = _BRACKET_HALF.get(group_letter, "?")

        # Find dangerous rivals in same bracket half
        same_half_powers = _HALF_A_POWERS if bracket_half == "A" else _HALF_B_POWERS
        dangers = [t for t in same_half_powers if t != team and rank_map.get(t, 99) <= 10]

        # ELO rank among all teams
        sorted_by_elo = sorted(team_db.items(), key=lambda x: -x[1].get("ELO", 0))
        elo_rank = next((i + 1 for i, (n, _) in enumerate(sorted_by_elo) if n == team), "?")

        # ── Build the narrative ──────────────────────────────────────────────
        lines = []

        # 1. Opening: rank + win probability
        medal = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣", "6️⃣", "7️⃣", "8️⃣"][i]
        lines.append(f"{medal} #{i+1} {team.upper()} — {champ_pct:.1f}% to win the tournament")

        # 2. ELO standing
        tier = _elo_tier(elo)
        lines.append(f"   ELO {elo} (#{elo_rank} globally) — {tier}.")

        # 3. Recent form + SOS context
        forma_desc = _forma_label(forma)
        if sos is not None:
            if sos < 0.92:
                sos_note = (f"⚠ SOS-adjusted downward (avg opponent ELO {opp_elo:.0f} — "
                            f"qualified against weaker competition).")
            elif sos > 1.05:
                sos_note = (f"✅ SOS-boosted (avg opponent ELO {opp_elo:.0f} — "
                            f"form built against elite opposition).")
            else:
                sos_note = (f"Neutral SOS (avg opponent ELO {opp_elo:.0f} — "
                            f"balanced schedule).")
            lines.append(f"   Recent form: {forma_desc} (FORMA {forma:.2f}). {sos_note}")
        else:
            lines.append(f"   Recent form: {forma_desc} (FORMA {forma:.2f}, from calibrated model base).")

        # 4. Group draw analysis
        group_rivals = [t for t in group_teams if t != team]
        rival_str = ", ".join(
            f"{t} ({team_db.get(t,{}).get('ELO',1600):.0f})"
            for t in sorted(group_rivals, key=lambda x: -team_db.get(x, {}).get("ELO", 0))
        )
        lines.append(f"   Group {group_letter}: {diff_label} — faces {rival_str}.")

        # 5. Bracket path
        if dangers:
            danger_str = " and ".join(
                f"{d} (#{rank_map.get(d,'?')})"
                for d in dangers[:2]
            )
            lines.append(
                f"   Bracket half {bracket_half}: shares the draw with {danger_str} — "
                f"could clash as early as the Quarter-Finals."
            )
        else:
            other_half = "B" if bracket_half == "A" else "A"
            other_powers = _HALF_B_POWERS if bracket_half == "A" else _HALF_A_POWERS
            safe_from = [t for t in other_powers if rank_map.get(t, 99) <= 5]
            if safe_from:
                safe_str = " and ".join(safe_from[:2])
                lines.append(
                    f"   Bracket half {bracket_half}: relatively open path — "
                    f"{safe_str} are on the other side of the draw (potential Final opponents only)."
                )
            else:
                lines.append(f"   Bracket half {bracket_half}: favourable draw with no top-5 rivals until the Final.")

        # 6. Key probability milestone
        if semi_pct >= 40:
            path_note = f"Reaches the Semi-Finals in {semi_pct:.0f}% of simulations — consistent contender."
        elif semi_pct >= 25:
            path_note = f"Semi-Final probability {semi_pct:.0f}% — capable but route dependent."
        else:
            path_note = f"Semi-Final probability {semi_pct:.0f}% — upset potential needed for deep run."
        lines.append(f"   {path_note}")

        # 7. One-line verdict
        if i == 0:
            verdict = "▶ MODEL'S PICK: Clear favourite based on ELO dominance and group draw."
        elif champ_pct >= 10:
            verdict = "▶ Genuine title contender — tournament could go their way."
        elif champ_pct >= 5:
            verdict = "▶ Dark horse — can win it all if bracket breaks right."
        else:
            verdict = "▶ Overachiever risk — model sees limited ceiling without upsets."
        lines.append(f"   {verdict}")

        # Print the block
        print()
        for line in lines:
            print(line)

    print(f"\n{'='*65}\n")


def main():
    args = sys.argv[1:]
    n_sims = 10_000
    show_groups = "--show-groups" in args

    for i, a in enumerate(args):
        if a == "--sims" and i + 1 < len(args):
            n_sims = int(args[i + 1])

    team_db = load_team_db()
    groups  = load_groups()

    if not groups:
        print("Error: no hay grupos definidos.")
        return

    if show_groups:
        print("\n=== Grupos Mundial 2026 ===")
        for letter, teams in groups.items():
            print(f"  Grupo {letter}: {', '.join(teams)}")
        print()

    if "--single" in args or n_sims == 1:
        print("\n" + "=" * 50)
        print("  SIMULACIÓN DEL MUNDIAL 2026")
        print("=" * 50)
        result = sim_full_tournament(groups, team_db)
        print(f"\n{'='*50}")
        print(f"  CAMPEÓN DEL MUNDO: {result['champion'].upper()}")
        print(f"  Finalista:         {result['finalist']}")
        print(f"  3er Puesto:        {result['third']}")
        sf_other = [t for t in result["semifinalists"]
                    if t not in [result["champion"], result["finalist"], result["third"]]]
        if sf_other:
            print(f"  4to Puesto:        {sf_other[0]}")
        print(f"{'='*50}\n")
        return

    print(f"\n{'='*55}")
    print(f"  MONTE CARLO — {n_sims:,} SIMULACIONES — MUNDIAL 2026")
    print(f"{'='*55}\n")

    mc_result = monte_carlo(groups, team_db, n=n_sims)
    table = mc_result["table"]
    metrics = mc_result["metrics"]

    print(f"\n{'Equipo':<22} {'Campeón':>9} {'Final':>8} {'Semi':>7} {'Cuartos':>8} {'3ro':>7}")
    print("-" * 68)
    for row in table[:24]:   # Mostrar los 24 más probables
        print(f"  {row['team']:<20} {row['champion_%']:>7.2f}%  {row['final_%']:>6.2f}%  "
              f"{row['semi_%']:>5.2f}%  {row['quarter_%']:>6.2f}%  {row['third_%']:>5.2f}%")

    print("\n  GOAL METRICS:")
    print("  " + "-" * 20)
    print(f"  Avg Goals / Tournament : {metrics['avg_goals']}")
    print(f"  Avg Over 2.5 Matches   : {metrics['avg_over_2_5']}")
    print(f"  Expected Over 2.5 %    : {metrics['pct_over_2_5']}%")

    # Narrative insights — reload a fresh team_db so MC-mutated ELOs don't corrupt display
    fresh_team_db = load_team_db()
    generate_insights(table, fresh_team_db, groups)

    # Guardar resultados
    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "mc_results.json")
    with open(out_path, "w") as f:
        json.dump(mc_result, f, indent=2, ensure_ascii=False)
    print(f"\nResultados guardados en data/mc_results.json")


if __name__ == "__main__":
    main()
