"""
FIFA World Cup 2026 — Web Predictor
Run: python app.py  →  http://localhost:5007
"""
import json
import math
import os
import random
from flask import Flask, jsonify, request, send_from_directory
from src.models.stacked_predictor import (
    stack_predict, expected_goals, _poisson_pmf, penalty_win_prob, DRAW_BOOST
)
from src.models.points_optimizer import find_optimal_pick
from src.models.predictor import VENUES
from src.models.tournament import load_team_db, match_probs as _match_probs
from src.utils.injuries_client import get_injury_detail
from src.utils.odds_client import get_match_odds

app = Flask(__name__, static_folder="static", static_url_path="")

# ── Key attacking threats per team (WC 2026 squads) ──────────────────────────
KEY_PLAYERS = {
    "Argentina":    [{"name": "Lautaro Martínez", "role": "ST",  "share": 0.32},
                     {"name": "Julián Álvarez",   "role": "ST",  "share": 0.24},
                     {"name": "Lionel Messi",      "role": "RW",  "share": 0.26}],
    "France":       [{"name": "Kylian Mbappé",    "role": "ST",  "share": 0.38},
                     {"name": "Marcus Thuram",     "role": "ST",  "share": 0.22},
                     {"name": "Ousmane Dembélé",   "role": "RW",  "share": 0.18}],
    "Brazil":       [{"name": "Vinicius Jr",       "role": "LW",  "share": 0.34},
                     {"name": "Endrick",           "role": "ST",  "share": 0.26},
                     {"name": "Rodrygo",           "role": "RW",  "share": 0.20}],
    "England":      [{"name": "Harry Kane",        "role": "ST",  "share": 0.36},
                     {"name": "Bukayo Saka",       "role": "RW",  "share": 0.22},
                     {"name": "Phil Foden",        "role": "AM",  "share": 0.18}],
    "Spain":        [{"name": "Lamine Yamal",      "role": "RW",  "share": 0.28},
                     {"name": "Álvaro Morata",     "role": "ST",  "share": 0.26},
                     {"name": "Pedri",             "role": "CM",  "share": 0.16}],
    "Germany":      [{"name": "Florian Wirtz",     "role": "AM",  "share": 0.28},
                     {"name": "Jamal Musiala",     "role": "AM",  "share": 0.26},
                     {"name": "Kai Havertz",       "role": "ST",  "share": 0.24}],
    "Portugal":     [{"name": "Cristiano Ronaldo", "role": "ST",  "share": 0.34},
                     {"name": "Bruno Fernandes",   "role": "AM",  "share": 0.24},
                     {"name": "Rafael Leão",       "role": "LW",  "share": 0.20}],
    "Netherlands":  [{"name": "Cody Gakpo",        "role": "LW",  "share": 0.30},
                     {"name": "Memphis Depay",     "role": "ST",  "share": 0.26},
                     {"name": "Joshua Zirkzee",    "role": "ST",  "share": 0.20}],
    "Belgium":      [{"name": "Romelu Lukaku",     "role": "ST",  "share": 0.34},
                     {"name": "Kevin De Bruyne",   "role": "AM",  "share": 0.22},
                     {"name": "Leandro Trossard",  "role": "LW",  "share": 0.18}],
    "Croatia":      [{"name": "Luka Modrić",       "role": "CM",  "share": 0.18},
                     {"name": "Andrej Kramarić",   "role": "ST",  "share": 0.30},
                     {"name": "Ivan Perišić",      "role": "LW",  "share": 0.22}],
    "Colombia":     [{"name": "Luis Díaz",         "role": "LW",  "share": 0.28},
                     {"name": "Jhon Córdoba",      "role": "ST",  "share": 0.26},
                     {"name": "James Rodríguez",   "role": "AM",  "share": 0.20}],
    "Uruguay":      [{"name": "Darwin Núñez",      "role": "ST",  "share": 0.36},
                     {"name": "Federico Valverde", "role": "CM",  "share": 0.20},
                     {"name": "Facundo Pellistri",  "role": "RW",  "share": 0.16}],
    "Japan":        [{"name": "Takumi Minamino",   "role": "AM",  "share": 0.24},
                     {"name": "Ayase Ueda",        "role": "ST",  "share": 0.28},
                     {"name": "Kaoru Mitoma",      "role": "LW",  "share": 0.22}],
    "Morocco":      [{"name": "Youssef En-Nesyri", "role": "ST",  "share": 0.34},
                     {"name": "Hakim Ziyech",      "role": "RW",  "share": 0.24},
                     {"name": "Sofiane Boufal",    "role": "LW",  "share": 0.18}],
    "USA":          [{"name": "Christian Pulisic", "role": "AM",  "share": 0.30},
                     {"name": "Ricardo Pepi",      "role": "ST",  "share": 0.28},
                     {"name": "Gio Reyna",         "role": "AM",  "share": 0.18}],
    "Mexico":       [{"name": "Hirving Lozano",    "role": "RW",  "share": 0.26},
                     {"name": "Santiago Giménez",  "role": "ST",  "share": 0.34},
                     {"name": "Alexis Vega",       "role": "LW",  "share": 0.16}],
    "Canada":       [{"name": "Alphonso Davies",   "role": "LW",  "share": 0.26},
                     {"name": "Jonathan David",    "role": "ST",  "share": 0.38},
                     {"name": "Tajon Buchanan",    "role": "RW",  "share": 0.18}],
    "Senegal":      [{"name": "Sadio Mané",        "role": "LW",  "share": 0.34},
                     {"name": "Ismaïla Sarr",      "role": "RW",  "share": 0.24},
                     {"name": "Nicolas Jackson",   "role": "ST",  "share": 0.22}],
    "Switzerland":  [{"name": "Breel Embolo",      "role": "ST",  "share": 0.30},
                     {"name": "Xherdan Shaqiri",   "role": "AM",  "share": 0.24},
                     {"name": "Ruben Vargas",      "role": "LW",  "share": 0.18}],
    "Austria":      [{"name": "Marcel Sabitzer",   "role": "AM",  "share": 0.24},
                     {"name": "Christoph Baumgartner", "role": "AM", "share": 0.22},
                     {"name": "Michael Gregoritsch", "role": "ST", "share": 0.28}],
    "Turkey":       [{"name": "Hakan Çalhanoğlu",  "role": "CM",  "share": 0.22},
                     {"name": "Kerem Aktürkoğlu",  "role": "LW",  "share": 0.26},
                     {"name": "Arda Güler",        "role": "AM",  "share": 0.24}],
    "South Korea":  [{"name": "Son Heung-min",     "role": "LW",  "share": 0.38},
                     {"name": "Lee Kang-in",       "role": "AM",  "share": 0.24},
                     {"name": "Hwang Hee-chan",    "role": "RW",  "share": 0.18}],
    "Serbia":       [{"name": "Dušan Vlahović",    "role": "ST",  "share": 0.36},
                     {"name": "Dušan Tadić",       "role": "AM",  "share": 0.22},
                     {"name": "Aleksandar Mitrović","role": "ST",  "share": 0.24}],
    "Ecuador":      [{"name": "Enner Valencia",    "role": "ST",  "share": 0.34},
                     {"name": "Moisés Caicedo",    "role": "CM",  "share": 0.16},
                     {"name": "Ángel Mena",        "role": "RW",  "share": 0.20}],
    "Iran":         [{"name": "Mehdi Taremi",      "role": "ST",  "share": 0.38},
                     {"name": "Sardar Azmoun",     "role": "ST",  "share": 0.28},
                     {"name": "Alireza Jahanbakhsh","role": "RW",  "share": 0.18}],
    "Nigeria":      [{"name": "Victor Osimhen",    "role": "ST",  "share": 0.40},
                     {"name": "Samuel Chukwueze",  "role": "RW",  "share": 0.20},
                     {"name": "Wilfred Ndidi",     "role": "CM",  "share": 0.12}],
    "Norway":       [{"name": "Erling Haaland",     "role": "ST",  "share": 0.45},
                     {"name": "Martin Ødegaard",     "role": "AM",  "share": 0.28},
                     {"name": "Alexander Sørloth",   "role": "ST",  "share": 0.15}],
    "Sweden":       [{"name": "Alexander Isak",     "role": "ST",  "share": 0.38},
                     {"name": "Viktor Gyökeres",     "role": "ST",  "share": 0.32},
                     {"name": "Dejan Kulusevski",    "role": "RW",  "share": 0.20}],
    "Scotland":     [{"name": "Scott McTominay",    "role": "CM",  "share": 0.22},
                     {"name": "Che Adams",           "role": "ST",  "share": 0.26},
                     {"name": "Lyndon Dykes",        "role": "ST",  "share": 0.18}],
    "Czech Republic":[{"name": "Patrik Schick",     "role": "ST",  "share": 0.36},
                     {"name": "Tomáš Souček",        "role": "CM",  "share": 0.20},
                     {"name": "Lukáš Provod",        "role": "AM",  "share": 0.16}],
    "Bosnia & Herzegovina": [{"name": "Edin Džeko", "role": "ST",  "share": 0.30},
                     {"name": "Miralem Pjanić",      "role": "CM",  "share": 0.18},
                     {"name": "Sead Kolašinac",      "role": "LB",  "share": 0.10}],
}


# ── Helpers ────────────────────────────────────────────────────────────────────
def scoreline_matrix(lam_a: float, lam_b: float, max_goals: int = 6) -> list:
    """Full probability matrix (0..max_goals × 0..max_goals), normalised."""
    cells = []
    total = 0.0
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            p = _poisson_pmf(lam_a, i) * _poisson_pmf(lam_b, j)
            cells.append({"gf_a": i, "gf_b": j, "prob": p})
            total += p
    for c in cells:
        c["prob"] = round(c["prob"] / total, 5)
        c["score"] = f"{c['gf_a']}-{c['gf_b']}"
    cells.sort(key=lambda x: -x["prob"])
    return cells


def _team_or_default(team_db: dict, name: str) -> dict:
    t = dict(team_db.get(name, {
        "ELO": 1650, "FORMA": 1.2, "GF_AVG": 1.0, "GA_AVG": 1.2, "INJURIES": 0
    }))
    t["name"] = name
    return t


# ── Routes ──────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/scores")
def scores():
    return send_from_directory("static", "scores.html")


@app.route("/optimizer")
def optimizer():
    return send_from_directory("static", "index.html")


@app.route("/api/teams")
def api_teams():
    from src.models.tournament import BASE_TEAM_DB
    return jsonify(sorted(BASE_TEAM_DB.keys()))


@app.route("/api/venues")
def api_venues():
    return jsonify(list(VENUES.keys()))


@app.route("/api/groups")
def api_groups():
    try:
        with open("data/wc2026_groups.json") as f:
            raw = json.load(f)
    except FileNotFoundError:
        return jsonify({"error": "groups file not found"}), 404

    team_db = load_team_db()
    # Round-robin pairing order: standard WC schedule within a 4-team group
    # R1: 0v1, 2v3  |  R2: 0v2, 1v3  |  R3: 0v3, 1v2
    ROUND_PAIRS = [(0,1),(2,3),(0,2),(1,3),(0,3),(1,2)]

    result = {}
    for gid, teams in raw.items():
        if gid.startswith("_") or not isinstance(teams, list):
            continue
        matches = []
        for (i, j) in ROUND_PAIRS:
            if i >= len(teams) or j >= len(teams):
                continue
            an, bn = teams[i], teams[j]
            ta = _team_or_default(team_db, an)
            tb = _team_or_default(team_db, bn)
            home = None
            if an in {"USA","Mexico","Canada"}: home = an
            elif bn in {"USA","Mexico","Canada"}: home = bn
            pred = _match_probs(ta, tb, home_team=home)
            lam_a, lam_b = expected_goals(ta, tb, home_team=home)
            matrix = scoreline_matrix(lam_a, lam_b)
            top3 = [{"score": m["score"], "prob": m["prob"]} for m in matrix[:3]]
            matches.append({
                "round":      ROUND_PAIRS.index((i,j))//2 + 1,
                "team_a":     an,
                "team_b":     bn,
                "p_win_a":    round(pred["p_win_a"], 3),
                "p_draw":     round(pred["p_draw"],  3),
                "p_win_b":    round(pred["p_win_b"], 3),
                "xg_a":       round(lam_a, 2),
                "xg_b":       round(lam_b, 2),
                "xg_total":   round(lam_a + lam_b, 2),
                "top_score":  matrix[0]["score"],
                "top_scores": top3,
                "is_host_match": home is not None,
            })
        team_elos = {t: team_db.get(t, {}).get("ELO", 0) for t in teams}
        result[gid] = {"teams": teams, "team_elos": team_elos, "matches": matches}

    return jsonify(result)


def _poisson_sample(lam: float) -> int:
    """Knuth algorithm for Poisson random variate."""
    if lam <= 0:
        return 0
    L = math.exp(-min(lam, 20))
    k, p = 0, 1.0
    while p > L:
        p *= random.random()
        k += 1
    return k - 1


@app.route("/api/group_standings")
def api_group_standings():
    try:
        with open("data/wc2026_groups.json") as f:
            raw = json.load(f)
    except FileNotFoundError:
        return jsonify({"error": "groups file not found"}), 404

    team_db = load_team_db()
    ROUND_PAIRS = [(0,1),(2,3),(0,2),(1,3),(0,3),(1,2)]
    N_SIMS = 8000

    result = {}
    for gid, teams in raw.items():
        if gid.startswith("_") or not isinstance(teams, list):
            continue

        # Pre-compute probabilities and xG for all 6 matches
        match_data = []
        for (i, j) in ROUND_PAIRS:
            an, bn = teams[i], teams[j]
            ta = _team_or_default(team_db, an)
            tb = _team_or_default(team_db, bn)
            home = None
            if an in {"USA","Mexico","Canada"}: home = an
            elif bn in {"USA","Mexico","Canada"}: home = bn
            pred = _match_probs(ta, tb, home_team=home)
            lam_a, lam_b = expected_goals(ta, tb, home_team=home)
            match_data.append({
                "i": i, "j": j,
                "p_a": pred["p_win_a"],
                "p_d": pred["p_draw"],
                "p_b": pred["p_win_b"],
                "lam_a": lam_a,
                "lam_b": lam_b,
            })

        # Monte Carlo group simulation
        pos_counts = {t: [0, 0, 0, 0] for t in teams}
        pts_total  = {t: 0 for t in teams}
        gd_total   = {t: 0 for t in teams}

        for _ in range(N_SIMS):
            pts = {t: 0 for t in teams}
            gd  = {t: 0 for t in teams}
            gf  = {t: 0 for t in teams}

            for m in match_data:
                an, bn = teams[m["i"]], teams[m["j"]]
                r = random.random()
                if r < m["p_a"]:
                    ga = _poisson_sample(m["lam_a"])
                    gb = _poisson_sample(m["lam_b"])
                    if ga <= gb: ga = gb + 1
                    pts[an] += 3
                elif r < m["p_a"] + m["p_d"]:
                    g  = _poisson_sample((m["lam_a"] + m["lam_b"]) / 2)
                    ga = gb = g
                    pts[an] += 1
                    pts[bn] += 1
                else:
                    ga = _poisson_sample(m["lam_a"])
                    gb = _poisson_sample(m["lam_b"])
                    if gb <= ga: gb = ga + 1
                    pts[bn] += 3

                gd[an] += ga - gb;  gd[bn] += gb - ga
                gf[an] += ga;       gf[bn] += gb

            ranked = sorted(teams,
                key=lambda t: (pts[t], gd[t], gf[t], random.random()),
                reverse=True)
            for pos, t in enumerate(ranked):
                pos_counts[t][pos] += 1
                pts_total[t] += pts[t]
                gd_total[t]  += gd[t]

        standings = []
        for t in teams:
            c = pos_counts[t]
            standings.append({
                "team":      t,
                "p1":        round(c[0] / N_SIMS, 3),
                "p2":        round(c[1] / N_SIMS, 3),
                "p3":        round(c[2] / N_SIMS, 3),
                "p4":        round(c[3] / N_SIMS, 3),
                "p_advance": round((c[0] + c[1]) / N_SIMS, 3),
                "avg_pts":   round(pts_total[t] / N_SIMS, 2),
                "avg_gd":    round(gd_total[t]  / N_SIMS, 2),
            })
        standings.sort(key=lambda x: -x["p_advance"])
        result[gid] = standings

    return jsonify(result)


@app.route("/api/group_odds")
def api_group_odds():
    from src.utils.odds_client import get_all_odds
    try:
        with open("data/wc2026_groups.json") as f:
            raw = json.load(f)
    except FileNotFoundError:
        return jsonify({"error": "groups file not found"}), 404

    team_db   = load_team_db()
    live_odds = get_all_odds()   # dict keyed by frozenset({teamA, teamB}) → odds
    ROUND_PAIRS = [(0,1),(2,3),(0,2),(1,3),(0,3),(1,2)]
    ROUND_NAMES = {1:"R1", 2:"R2", 3:"R3"}

    # Normalise team names so our group names match the API's spelling
    # (API might use "Côte d'Ivoire" vs our "Ivory Coast", etc.)
    _NAME_NORM = {
        "Ivory Coast":          "Ivory Coast",
        "Côte d'Ivoire":        "Ivory Coast",
        "Curaçao":              "Curaçao",
        "Curacao":              "Curaçao",
        "Bosnia & Herzegovina": "Bosnia & Herzegovina",
        "Bosnia and Herzegovina": "Bosnia & Herzegovina",
        "DR Congo":             "DR Congo",
        "Congo DR":             "DR Congo",
        "Democratic Republic of Congo": "DR Congo",
        "Cape Verde":           "Cape Verde",
        "Cabo Verde":           "Cape Verde",
        "Czech Republic":       "Czech Republic",
        "Czechia":              "Czech Republic",
        "USA":                  "USA",
        "United States":        "USA",
    }

    def _norm(name):
        return _NAME_NORM.get(name, name)

    # Re-key live_odds with normalised names
    normalised_odds = {}
    for fs, val in live_odds.items():
        names = list(fs)
        nkey = frozenset([_norm(names[0]), _norm(names[1])])
        normalised_odds[nkey] = {**val,
            "home": _norm(val["home"]),
            "away": _norm(val["away"])}

    rows = []
    for gid, teams in raw.items():
        if gid.startswith("_") or not isinstance(teams, list):
            continue
        for ri, (i, j) in enumerate(ROUND_PAIRS):
            an, bn = teams[i], teams[j]
            ta = _team_or_default(team_db, an)
            tb = _team_or_default(team_db, bn)
            home = None
            if an in {"USA","Mexico","Canada"}: home = an
            elif bn in {"USA","Mexico","Canada"}: home = bn
            pred = _match_probs(ta, tb, home_team=home)
            lam_a, lam_b = expected_goals(ta, tb, home_team=home)
            matrix = scoreline_matrix(lam_a, lam_b)

            p_a = pred["p_win_a"]
            p_d = pred["p_draw"]
            p_b = pred["p_win_b"]

            # Fair (no-vig) odds
            fair = {"a": round(1/p_a,2), "draw": round(1/p_d,2), "b": round(1/p_b,2)}

            # Look up live odds using normalised key
            mkt = None
            key = frozenset([an, bn])
            if key in normalised_odds:
                raw_mkt = normalised_odds[key]
                if raw_mkt["home"] == an:
                    mkt = {"a": raw_mkt["a"], "draw": raw_mkt["draw"], "b": raw_mkt["b"]}
                else:
                    mkt = {"a": raw_mkt["b"], "draw": raw_mkt["draw"], "b": raw_mkt["a"]}

            # Edge = model_prob - vig-adjusted implied prob
            # Remove bookmaker margin first so we're comparing fairly
            ev = None
            if mkt:
                raw_a    = 1 / mkt["a"]
                raw_d    = 1 / mkt["draw"]
                raw_b    = 1 / mkt["b"]
                overround = raw_a + raw_d + raw_b          # typically 1.05-1.10
                impl_a   = raw_a / overround
                impl_d   = raw_d / overround
                impl_b   = raw_b / overround
                ev = {
                    "a":    round(p_a - impl_a, 3),
                    "draw": round(p_d - impl_d, 3),
                    "b":    round(p_b - impl_b, 3),
                    "impl_a":   round(impl_a, 3),
                    "impl_d":   round(impl_d, 3),
                    "impl_b":   round(impl_b, 3),
                    "overround": round(overround, 4),
                }
                ev["max"]  = round(max(ev["a"], ev["draw"], ev["b"]), 3)
                ev["best"] = max(("a","draw","b"), key=lambda k: ev[k])

            rows.append({
                "group":     gid,
                "round":     ROUND_NAMES[ri//2 + 1],
                "team_a":    an,
                "team_b":    bn,
                "xg_a":      round(lam_a, 2),
                "xg_b":      round(lam_b, 2),
                "xg_total":  round(lam_a + lam_b, 2),
                "p_win_a":   round(p_a, 3),
                "p_draw":    round(p_d, 3),
                "p_win_b":   round(p_b, 3),
                "fair":      fair,
                "market":    mkt,
                "ev":        ev,
                "top_scores": [{"score": m["score"], "prob": m["prob"]} for m in matrix[:3]],
                "is_host":   home is not None,
            })

    # has_odds = True only if at least some WC matches matched
    matched = sum(1 for r in rows if r["market"])
    rows.sort(key=lambda r: -(r["ev"]["max"] if r["ev"] else 0))
    return jsonify({"has_odds": matched > 0, "matched_count": matched, "matches": rows})


@app.route("/api/validate")
def api_validate():
    from src.analysis.calibrate import (
        WC2022_MATCHES, evaluate_individual_models,
        grid_search_weights, blend_probs, compute_log_loss, accuracy,
        _make_teams, _predict_outcome
    )
    from src.models.stacked_predictor import DEFAULT_WEIGHTS, stack_predict as sp

    model_evals = evaluate_individual_models()
    best_w, best_ll, best_acc = grid_search_weights(model_evals, step=0.05)

    dw3 = {k: DEFAULT_WEIGHTS[k] for k in ["elo","poisson","ml"]}
    preds_dict = {k: model_evals[k]["preds"] for k in ["elo","poisson","ml"]}
    blended    = blend_probs(preds_dict, dw3)
    ll_def     = compute_log_loss(blended, model_evals["results"])
    acc_def    = accuracy(blended, model_evals["results"])

    stacked_preds = []
    for m in WC2022_MATCHES:
        ta, tb = _make_teams(m)
        pred = sp(ta, tb)
        stacked_preds.append((pred["p_win_a"], pred["p_draw"], pred["p_win_b"]))
    ll_stacked  = compute_log_loss(stacked_preds, model_evals["results"])
    acc_stacked = accuracy(stacked_preds, model_evals["results"])

    results_list = model_evals["results"]
    matches_detail = []
    for i, m in enumerate(WC2022_MATCHES):
        pa, pd, pb = stacked_preds[i]
        predicted = _predict_outcome(stacked_preds[i])
        actual    = results_list[i]
        matches_detail.append({
            "team_a":    m["name_a"],
            "team_b":    m["name_b"],
            "elo_a":     m["elo_a"],
            "elo_b":     m["elo_b"],
            "p_win_a":   round(pa, 3),
            "p_draw":    round(pd, 3),
            "p_win_b":   round(pb, 3),
            "predicted": predicted,
            "actual":    actual,
            "correct":   predicted == actual,
        })

    return jsonify({
        "n_matches": len(WC2022_MATCHES),
        "models": {
            "elo":     {"log_loss": round(model_evals["elo"]["log_loss"], 4),
                        "accuracy": round(model_evals["elo"]["accuracy"], 4)},
            "poisson": {"log_loss": round(model_evals["poisson"]["log_loss"], 4),
                        "accuracy": round(model_evals["poisson"]["accuracy"], 4)},
            "ml":      {"log_loss": round(model_evals["ml"]["log_loss"], 4),
                        "accuracy": round(model_evals["ml"]["accuracy"], 4)},
            "stacked": {"log_loss": round(ll_stacked, 4),
                        "accuracy": round(acc_stacked, 4)},
        },
        "optimal_weights": best_w,
        "optimal_log_loss": round(best_ll, 4),
        "optimal_accuracy": round(best_acc, 4),
        "current_log_loss": round(ll_def, 4),
        "current_accuracy": round(acc_def, 4),
        "matches": matches_detail,
    })


@app.route("/api/tournament")
def api_tournament():
    try:
        with open("data/mc_results.json") as f:
            data = json.load(f)
        
        # If it's the old list format, handle it gracefully
        if isinstance(data, list):
            data.sort(key=lambda r: -r.get("champion_%", 0))
            return jsonify({"table": data, "metrics": None})
            
        # New dict format
        data["table"].sort(key=lambda r: -r.get("champion_%", 0))
        return jsonify(data)
    except FileNotFoundError:
        return jsonify({"table": [], "metrics": None})


@app.route("/api/predict", methods=["POST"])
def api_predict():
    body         = request.get_json(force=True)
    team_a_name  = body.get("team_a", "")
    team_b_name  = body.get("team_b", "")
    manual_odds  = body.get("odds")          # {a, draw, b} or null
    venue_name   = body.get("venue") or None # optional venue name

    if not team_a_name or not team_b_name:
        return jsonify({"error": "team_a and team_b required"}), 400

    team_db = load_team_db()
    ta = _team_or_default(team_db, team_a_name)
    tb = _team_or_default(team_db, team_b_name)

    # Host advantage (group stage logic — always apply if host nation)
    home_team = None
    if team_a_name in {"USA", "Mexico", "Canada"}:
        home_team = team_a_name
    elif team_b_name in {"USA", "Mexico", "Canada"}:
        home_team = team_b_name

    pred  = stack_predict(ta, tb, home_team=home_team, venue_name=venue_name)
    lam_a, lam_b = expected_goals(ta, tb, home_team=home_team, venue_name=venue_name)

    # Scoreline matrix
    matrix = scoreline_matrix(lam_a, lam_b)

    # Points optimizer
    opt = find_optimal_pick(lam_a, lam_b)

    # Odds
    bookie = manual_odds
    if not bookie:
        fetched = get_match_odds(team_a_name, team_b_name)
        if fetched:
            bookie = {
                "a":    fetched["ODDS_BOOKIE_A"],
                "draw": fetched["ODDS_BOOKIE_DRAW"],
                "b":    fetched["ODDS_BOOKIE_B"],
            }

    fair_odds = {
        "a":    round(1 / pred["p_win_a"], 2),
        "draw": round(1 / pred["p_draw"],  2),
        "b":    round(1 / pred["p_win_b"], 2),
    }

    ev = None
    if bookie and all(bookie.get(k, 0) > 0 for k in ("a", "draw", "b")):
        ev = {
            "a":    round(pred["p_win_a"] * bookie["a"]    - 1, 4),
            "draw": round(pred["p_draw"]  * bookie["draw"] - 1, 4),
            "b":    round(pred["p_win_b"] * bookie["b"]    - 1, 4),
        }

    # Key players + xG contribution
    def enrich_players(name, lam):
        raw = KEY_PLAYERS.get(name, [])
        return [{"name": p["name"], "role": p["role"],
                 "xg": round(lam * p["share"], 2)} for p in raw]

    return jsonify({
        "team_a":        team_a_name,
        "team_b":        team_b_name,
        "probs":         {"win_a": round(pred["p_win_a"], 4),
                          "draw":  round(pred["p_draw"],  4),
                          "win_b": round(pred["p_win_b"], 4)},
        "xg":            {"a": round(lam_a, 2), "b": round(lam_b, 2)},
        "fair_odds":     fair_odds,
        "bookie_odds":   bookie,
        "ev":            ev,
        "scoreline_matrix": matrix,
        "top_scorelines":   matrix[:12],
        "optimal_pick":     opt["max_ev_pick"],
        "top_candidates":   opt["top_candidates"],
        "pick_delta_ev":    opt["pick_delta_ev"],
        "draw_probability": opt["draw_probability"],
        "model_breakdown":  pred["model_breakdown"],
        "penalty_prob_a":   round(penalty_win_prob(team_a_name, team_b_name), 3),
        "injuries_a":    get_injury_detail(team_a_name),
        "injuries_b":    get_injury_detail(team_b_name),
        "players_a":     enrich_players(team_a_name, lam_a),
        "players_b":     enrich_players(team_b_name, lam_b),
        "elo_a":         ta.get("ELO", 0),
        "elo_b":         tb.get("ELO", 0),
    })


@app.route("/api/optimal_pick", methods=["POST"])
def api_optimal_pick():
    """
    Returns the scoreline prediction that maximises expected points in the
    guessing game (5 pts exact / 3 pts correct outcome + GD / 2 pts correct outcome).

    Request body: {"team_a": "Spain", "team_b": "Morocco", "venue": "optional"}
    """
    body        = request.get_json(force=True)
    team_a_name = body.get("team_a", "")
    team_b_name = body.get("team_b", "")
    venue_name  = body.get("venue") or None

    if not team_a_name or not team_b_name:
        return jsonify({"error": "team_a and team_b required"}), 400

    team_db = load_team_db()
    ta = _team_or_default(team_db, team_a_name)
    tb = _team_or_default(team_db, team_b_name)

    home_team = None
    if team_a_name in {"USA", "Mexico", "Canada"}:
        home_team = team_a_name
    elif team_b_name in {"USA", "Mexico", "Canada"}:
        home_team = team_b_name

    pred          = stack_predict(ta, tb, home_team=home_team, venue_name=venue_name)
    lam_a, lam_b  = expected_goals(ta, tb, home_team=home_team, venue_name=venue_name)
    opt           = find_optimal_pick(lam_a, lam_b)

    return jsonify({
        "team_a":           team_a_name,
        "team_b":           team_b_name,
        "probs":            {"win_a": round(pred["p_win_a"], 4),
                             "draw":  round(pred["p_draw"],  4),
                             "win_b": round(pred["p_win_b"], 4)},
        "xg":               {"a": round(lam_a, 2), "b": round(lam_b, 2)},
        "optimal_pick":     opt["max_ev_pick"],
        "max_prob_pick":    opt["max_prob_pick"],
        "top_candidates":   opt["top_candidates"],
        "pick_delta_ev":    opt["pick_delta_ev"],
        "draw_probability": opt["draw_probability"],
        "draw_boost_applied": DRAW_BOOST,
        "elo_a":            ta.get("ELO", 0),
        "elo_b":            tb.get("ELO", 0),
    })


@app.route("/api/porra")
def api_porra():
    """
    Returns optimal picks for all WC 2026 group-stage fixtures, in date order.
    For completed matches also returns actual result and points scored.
    """
    import csv as _csv
    from src.models.points_optimizer import find_optimal_pick

    # Team name normalisation: CSV names → team_db names
    NAME_MAP = {
        "United States":          "USA",
        "Bosnia and Herzegovina": "Bosnia & Herzegovina",
    }

    # Host nations for home advantage
    HOST_NATIONS = {"USA", "Mexico", "Canada"}

    team_db  = load_team_db()

    def _team(name):
        t = dict(team_db.get(name, {
            "ELO": 1650, "FORMA": 1.2, "GF_AVG": 1.0, "GA_AVG": 1.2, "INJURIES": 0
        }))
        t["name"] = name
        return t

    def _score_pick(pick_a, pick_b, actual_a, actual_b):
        """Points earned for a pick given the actual result (exclusive tiers)."""
        def outcome(x, y): return 1 if x > y else (0 if x == y else -1)
        if pick_a == actual_a and pick_b == actual_b:
            return 5
        if outcome(pick_a, pick_b) == outcome(actual_a, actual_b):
            if (pick_a - pick_b) == (actual_a - actual_b):
                return 3
            return 2
        return 0

    rows = []
    with open("data/intl_results.csv", newline="", encoding="utf-8") as f:
        for row in _csv.DictReader(f):
            if row.get("tournament") != "FIFA World Cup":
                continue
            if row.get("date", "") < "2026-06-01":
                continue

            date      = row["date"]
            home_csv  = row["home_team"]
            away_csv  = row["away_team"]
            home_name = NAME_MAP.get(home_csv, home_csv)
            away_name = NAME_MAP.get(away_csv, away_csv)

            # Determine home advantage
            home_team = None
            if home_name in HOST_NATIONS:
                home_team = home_name
            elif away_name in HOST_NATIONS:
                home_team = away_name

            ta = _team(home_name)
            tb = _team(away_name)

            try:
                lam_a, lam_b = expected_goals(ta, tb, home_team=home_team)
                pred = stack_predict(ta, tb, home_team=home_team)
                opt  = find_optimal_pick(lam_a, lam_b)
            except Exception:
                lam_a = lam_b = 1.25
                pred = {"p_win_a": 0.333, "p_draw": 0.334, "p_win_b": 0.333}
                opt  = {"max_ev_pick": {"score": "1-0", "gf_a": 1, "gf_b": 0,
                                        "expected_pts": 0, "prob": 0},
                        "draw_probability": 0.33}

            pick = opt["max_ev_pick"]

            # Actual result if available
            hs = row.get("home_score", "NA")
            as_ = row.get("away_score", "NA")
            played   = hs not in ("NA", "")
            actual   = None
            pts_earned = None
            if played:
                actual = {"home": int(hs), "away": int(as_)}
                pts_earned = _score_pick(
                    pick["gf_a"], pick["gf_b"],
                    actual["home"], actual["away"]
                )

            rows.append({
                "date":        date,
                "home":        home_name,
                "away":        away_name,
                "city":        row.get("city", ""),
                "probs":       {
                    "win_a": round(pred["p_win_a"], 3),
                    "draw":  round(pred["p_draw"],  3),
                    "win_b": round(pred["p_win_b"], 3),
                },
                "xg":          {"a": round(lam_a, 2), "b": round(lam_b, 2)},
                "optimal_pick": pick,
                "draw_probability": opt["draw_probability"],
                "played":      played,
                "actual":      actual,
                "pts_earned":  pts_earned,
            })

    return jsonify({"fixtures": rows, "total": len(rows)})


@app.route("/api/bets")
def api_bets():
    """
    Returns all group-stage value bets where model probability
    beats the fixed market line. Quarter-Kelly sizing on $1000 bankroll.
    """
    try:
        with open("data/wc2026_groups.json") as f:
            raw = json.load(f)
    except FileNotFoundError:
        return jsonify({"error": "groups file not found"}), 404

    from src.models.stacked_predictor import _compute_lambdas

    team_db = load_team_db()
    BANKROLL = 1000.0

    # Fallback fixed odds (used only when Coolbet data has no odds for that match/line)
    FIXED_FALLBACK = {
        "Over 1.5":   1.45,
        "Under 1.5":  2.80,
        "Over 2.5":   1.85,
        "Under 2.5":  1.85,
        "Over 3.5":   2.05,
        "Under 3.5":  1.78,
        "BTTS Yes":   1.82,
        "BTTS No":    2.02,
    }

    # Build per-match O/U odds lookup from live Coolbet data
    # Structure: coolbet_ou[(home, away, "Over/Under", line_float)] = odds
    _BETS_NORM = {
        "Czechia":              "Czech Republic",
        "Congo DR":             "DR Congo",
        "Democratic Republic of Congo": "DR Congo",
        "Côte d'Ivoire":        "Ivory Coast",
        "Cabo Verde":           "Cape Verde",
        "Bosnia and Herzegovina": "Bosnia & Herzegovina",
        "United States":        "USA",
        "Curacao":              "Curaçao",
    }
    coolbet_ou: dict = {}
    coolbet_btts: dict = {}
    try:
        import json as _json
        from pathlib import Path as _Path
        _raw = _json.loads(_Path("data/coolbet/latest.json").read_text())
        for _r in _raw:
            if _r.get("page") != "matches":
                continue
            _mkt = _r.get("market", "")
            # Normalize Coolbet team names to match model team names
            _home = _BETS_NORM.get(_r.get("home", "").strip(), _r.get("home", "").strip())
            _away = _BETS_NORM.get(_r.get("away", "").strip(), _r.get("away", "").strip())
            if _mkt == "Total Goals Over / Under":
                _sel  = _r.get("selection", "")   # "Over" or "Under"
                _line = _r.get("line", "")
                try:
                    _lf = float(_line)
                    coolbet_ou[(_home, _away, _sel, _lf)] = float(_r["odds"])
                except (ValueError, TypeError):
                    pass
            elif _mkt == "Both Teams to Score":
                _sel = _r.get("selection", "")
                coolbet_btts[(_home, _away, _sel)] = float(_r["odds"])
    except Exception:
        pass  # fall back to FIXED if file missing or malformed

    def _get_ou_odds(home: str, away: str, sel: str, line: float):
        """Return live Coolbet O/U odds, or None if the match isn't listed yet."""
        # Try direct match
        o = coolbet_ou.get((home, away, sel, line))
        if o:
            return o
        # Try reversed (Coolbet sometimes lists away team first)
        o = coolbet_ou.get((away, home, sel, line))
        if o:
            return o
        # Fuzzy: match any key where both team names appear (handles minor name diffs)
        hl, al = home.lower(), away.lower()
        for (ch, ca, cs, cl), co in coolbet_ou.items():
            if cs == sel and cl == line:
                if (hl in ch.lower() or ch.lower() in hl) and \
                   (al in ca.lower() or ca.lower() in al):
                    return co
        # Match not found — Coolbet doesn't have this game yet (e.g. Round 3)
        return None

    def _get_btts_odds(home: str, away: str, sel: str):
        """Return live Coolbet BTTS odds, or None if not available."""
        o = coolbet_btts.get((home, away, sel))
        if o: return o
        o = coolbet_btts.get((away, home, sel))
        if o: return o
        return None  # Coolbet does not offer an in-match BTTS market

    def kelly(p, odds, frac=0.25, cap=0.04):
        ev = p * odds - 1
        if ev <= 0:
            return 0.0, ev
        k = ev / (odds - 1)
        return min(k * frac, cap), ev

    ROUND_PAIRS = [(0,1),(2,3),(0,2),(1,3),(0,3),(1,2)]
    bets = []

    for gid, teams in raw.items():
        if gid.startswith("_") or not isinstance(teams, list):
            continue

        for ri, (i, j) in enumerate(ROUND_PAIRS):
            an, bn = teams[i], teams[j]
            ta = _team_or_default(team_db, an)
            tb = _team_or_default(team_db, bn)
            home = None
            if an in {"USA","Mexico","Canada"}: home = an
            elif bn in {"USA","Mexico","Canada"}: home = bn

            lam_a, lam_b = expected_goals(ta, tb, home_team=home)

            # Build Poisson grid
            grid = {}
            for a in range(12):
                pa = _poisson_pmf(lam_a, a)
                for b in range(12):
                    grid[(a, b)] = pa * _poisson_pmf(lam_b, b)

            probs = {
                "Over 1.5":  sum(v for (a,b),v in grid.items() if a+b > 1),
                "Under 1.5": sum(v for (a,b),v in grid.items() if a+b <= 1),
                "Over 2.5":  sum(v for (a,b),v in grid.items() if a+b > 2),
                "Under 2.5": sum(v for (a,b),v in grid.items() if a+b <= 2),
                "Over 3.5":  sum(v for (a,b),v in grid.items() if a+b > 3),
                "Under 3.5": sum(v for (a,b),v in grid.items() if a+b <= 3),
                "BTTS Yes":  sum(v for (a,b),v in grid.items() if a>=1 and b>=1),
                "BTTS No":   sum(v for (a,b),v in grid.items() if not(a>=1 and b>=1)),
            }

            rnd = ri // 2 + 1
            match_label = f"{an} vs {bn}"

            # Get live Coolbet odds for this specific match.
            # BTTS may be None when Coolbet has no in-match market for it.
            live_odds = {
                "Over 1.5":  _get_ou_odds(an, bn, "Over",  1.5),
                "Under 1.5": _get_ou_odds(an, bn, "Under", 1.5),
                "Over 2.5":  _get_ou_odds(an, bn, "Over",  2.5),
                "Under 2.5": _get_ou_odds(an, bn, "Under", 2.5),
                "Over 3.5":  _get_ou_odds(an, bn, "Over",  3.5),
                "Under 3.5": _get_ou_odds(an, bn, "Under", 3.5),
                "BTTS Yes":  _get_btts_odds(an, bn, "Yes"),
                "BTTS No":   _get_btts_odds(an, bn, "No"),
            }

            for mkt, mkt_odds in live_odds.items():
                if mkt_odds is None:
                    continue  # no real bookmaker odds available — skip
                p = probs[mkt]
                p_mkt = 1.0 / mkt_odds
                if p <= p_mkt:
                    continue  # no value

                stake_frac, ev = kelly(p, mkt_odds)
                edge = (p - p_mkt) * 100

                # Categorise market for grouping
                if "1.5" in mkt:
                    cat = "O/U 1.5"
                elif "2.5" in mkt:
                    cat = "O/U 2.5"
                elif "3.5" in mkt:
                    cat = "O/U 3.5"
                else:
                    cat = "BTTS"

                bets.append({
                    "group":      gid,
                    "round":      rnd,
                    "match":      match_label,
                    "team_a":     an,
                    "team_b":     bn,
                    "market":     mkt,
                    "cat":        cat,
                    "p_model":    round(p, 4),
                    "p_mkt":      round(p_mkt, 4),
                    "mkt_odds":   mkt_odds,
                    "edge":       round(edge, 1),
                    "ev_pct":     round(ev * 100, 1),
                    "stake_pct":  round(stake_frac * 100, 2),
                    "stake_usd":  round(min(stake_frac, 0.04) * BANKROLL, 1),
                    "xg_a":       round(lam_a, 2),
                    "xg_b":       round(lam_b, 2),
                    "xg_total":   round(lam_a + lam_b, 2),
                })

    bets.sort(key=lambda x: -x["ev_pct"])

    total_stake  = sum(b["stake_usd"] for b in bets)
    total_ev     = sum(b["stake_usd"] * b["ev_pct"] / 100 for b in bets)
    roi          = (total_ev / total_stake * 100) if total_stake else 0

    # Suggested parlays
    top_o25  = sorted([b for b in bets if b["market"] == "Over 2.5"],  key=lambda x: -x["p_model"])[:5]
    top_u35  = sorted([b for b in bets if b["market"] == "Under 3.5"], key=lambda x: -x["p_model"])[:5]
    top_btts = sorted([b for b in bets if b["market"] == "BTTS Yes"],  key=lambda x: -x["p_model"])[:5]

    def parlay_info(legs, stake=10):
        cp = 1.0
        co = 1.0
        for b in legs:
            cp *= b["p_model"]
            co *= b["mkt_odds"]
        return {
            "legs":   [{"match": b["match"], "market": b["market"],
                        "odds": b["mkt_odds"], "p": round(b["p_model"]*100,1)} for b in legs],
            "combined_p": round(cp * 100, 2),
            "combined_odds": round(co, 1),
            "suggested_stake": stake,
            "expected_return": round(stake * co * cp, 1),
        }

    parlays = {
        "over_25":  parlay_info(top_o25),
        "under_35": parlay_info(top_u35),
        "btts":     parlay_info(top_btts),
    }

    # Per-market summary
    cats = ["O/U 1.5", "O/U 2.5", "O/U 3.5", "BTTS"]
    mkt_summary = {}
    for cat in cats:
        cat_bets = [b for b in bets if b["cat"] == cat]
        s = sum(b["stake_usd"] for b in cat_bets)
        e = sum(b["stake_usd"] * b["ev_pct"] / 100 for b in cat_bets)
        mkt_summary[cat] = {
            "count": len(cat_bets),
            "stake": round(s, 1),
            "ev":    round(e, 1),
            "roi":   round(e / s * 100 if s else 0, 1),
        }

    return jsonify({
        "bets": bets,
        "summary": {
            "total_bets":   len(bets),
            "total_stake":  round(total_stake, 1),
            "expected_profit": round(total_ev, 1),
            "roi":          round(roi, 1),
        },
        "by_market": mkt_summary,
        "parlays":   parlays,
    })


@app.route("/api/bracket")
def api_bracket():
    """
    Runs N Monte Carlo simulations and returns a consensus bracket:
    for each slot in each round, the most likely team(s) with probabilities.
    Cached to data/bracket_consensus.json for 2 hours.
    """
    import time as _time
    from pathlib import Path
    from collections import defaultdict
    from src.models.tournament import sim_full_tournament

    cache_path = Path("data/bracket_consensus.json")
    if cache_path.exists() and _time.time() - cache_path.stat().st_mtime < 7200:
        with open(cache_path, encoding="utf-8") as f:
            return jsonify(json.load(f))

    try:
        with open("data/wc2026_groups.json") as f:
            groups_raw = json.load(f)
        groups = {k: v for k, v in groups_raw.items() if not k.startswith("_")}
    except FileNotFoundError:
        return jsonify({"error": "groups file not found"}), 404

    team_db = load_team_db()

    # Pre-compute ensemble predictions for all team pairs once before the MC loop
    from src.models.tournament import precompute_ensemble_matchups
    all_team_names = list({t for g in groups.values() if isinstance(g, list) for t in g})
    precompute_ensemble_matchups(all_team_names)

    N = 100000

    # Slot counters — each position in the bracket tracked independently
    def slot_list(n): return [defaultdict(int) for _ in range(n)]

    r32_slot  = slot_list(32)   # 16 matches × 2 sides
    r32w_slot = slot_list(16)   # 16 R32 winners (= R16 entrants)
    r16_slot  = slot_list(16)   # 8 matches × 2 sides
    r16w_slot = slot_list(8)    # 8 R16 winners (= QF entrants)
    qf_slot   = slot_list(8)    # 4 matches × 2 sides
    qf_w_slot = slot_list(4)    # 4 QF winners (= SF entrants)
    sf_slot   = slot_list(4)    # 2 matches × 2 sides
    sf_w_slot = slot_list(2)    # 2 SF winners → Final
    final_a   = defaultdict(int)
    final_b   = defaultdict(int)
    champ_cnt = defaultdict(int)
    grp_first  = {g: defaultdict(int) for g in groups}
    grp_second = {g: defaultdict(int) for g in groups}

    for _ in range(N):
        r = sim_full_tournament(groups, team_db, silent=True)

        # Group results
        for g, ranking in r["group_results"].items():
            if len(ranking) >= 2:
                grp_first[g][ranking[0]]  += 1
                grp_second[g][ranking[1]] += 1

        # R32 slots (32 participants in pair order)
        pts = r["r32_participants"]
        for i in range(32):
            r32_slot[i][pts[i]] += 1

        # R32 winners (Ronda de 32)
        r32w = r["results_by_round"].get("Ronda de 32", [])
        for i, t in enumerate(r32w):
            r32w_slot[i][t] += 1

        # R16 slots from R32 winners (pair i = r32w[2i] vs r32w[2i+1])
        for i in range(8):
            if 2*i+1 < len(r32w):
                r16_slot[2*i][r32w[2*i]]     += 1
                r16_slot[2*i+1][r32w[2*i+1]] += 1

        # R16 winners (Octavos de Final)
        r16w = r["results_by_round"].get("Octavos de Final", [])
        for i, t in enumerate(r16w):
            r16w_slot[i][t] += 1

        # QF slots from R16 winners
        for i in range(4):
            if 2*i+1 < len(r16w):
                qf_slot[2*i][r16w[2*i]]     += 1
                qf_slot[2*i+1][r16w[2*i+1]] += 1

        # QF winners (Cuartos de Final)
        qfw = r["results_by_round"].get("Cuartos de Final", [])
        for i, t in enumerate(qfw):
            qf_w_slot[i][t] += 1

        # SF slots from QF winners
        for i in range(2):
            if 2*i+1 < len(qfw):
                sf_slot[2*i][qfw[2*i]]     += 1
                sf_slot[2*i+1][qfw[2*i+1]] += 1

        # SF winners (Semifinales)
        sfw = r["results_by_round"].get("Semifinales", [])
        for i, t in enumerate(sfw):
            sf_w_slot[i][t] += 1

        # Final participants
        if len(sfw) >= 2:
            final_a[sfw[0]] += 1
            final_b[sfw[1]] += 1

        champ_cnt[r["champion"]] += 1

    def top(counter, n=3):
        total = sum(counter.values()) or 1
        return [{"team": t, "prob": round(c / total, 4)}
                for t, c in sorted(counter.items(), key=lambda x: -x[1])[:n]]

    def build_match(slot_a_counter, slot_b_counter):
        return {"a": top(slot_a_counter, 2), "b": top(slot_b_counter, 2)}

    # Build group stage data
    group_data = {}
    for g in sorted(groups.keys()):
        group_data[g] = {
            "teams":  groups[g],
            "first":  top(grp_first[g]),
            "second": top(grp_second[g]),
        }

    # Build knockout rounds as list of matches
    r32_matches  = [build_match(r32_slot[2*i],  r32_slot[2*i+1])  for i in range(16)]
    r16_matches  = [build_match(r16_slot[2*i],  r16_slot[2*i+1])  for i in range(8)]
    qf_matches   = [build_match(qf_slot[2*i],   qf_slot[2*i+1])   for i in range(4)]
    sf_matches   = [build_match(sf_slot[2*i],   sf_slot[2*i+1])   for i in range(2)]
    final_match  = build_match(final_a, final_b)
    champion     = top(champ_cnt, 12)

    result = {
        "groups":    group_data,
        "r32":       r32_matches,
        "r16":       r16_matches,
        "qf":        qf_matches,
        "sf":        sf_matches,
        "final":     final_match,
        "champion":  champion,
        "n_sims":    N,
    }

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)

    return jsonify(result)


@app.route("/api/valuebets")
def api_valuebets():
    """Returns scraped Coolbet value bets from value_bets.json."""
    try:
        with open("data/coolbet/value_bets.json", encoding="utf-8") as f:
            data = json.load(f)
        return jsonify(data)
    except FileNotFoundError:
        return jsonify([]), 200


@app.route("/api/team_stats")
def api_team_stats():
    """
    Per-team corners and shots averages from StatsBomb match data
    (WC 2018, Euro 2020, WC 2022, Euro/Copa 2024).
    """
    import csv as _csv
    from pathlib import Path as _Path

    stats_path = _Path("data/statsbomb_match_stats.csv")
    if not stats_path.exists():
        return jsonify([]), 200

    acc = {}
    with stats_path.open(encoding="utf-8") as f:
        reader = _csv.DictReader(f)
        for row in reader:
            try:
                hc = float(row["home_corners"])
                ac = float(row["away_corners"])
                hs = float(row["home_shots"])
                as_ = float(row["away_shots"])
                hp = float(row["home_possession"]) if row.get("home_possession") else None
            except (ValueError, KeyError):
                continue

            for team, c_for, c_ag, s_for, s_ag, poss in [
                (row["home_team"], hc, ac, hs, as_, hp),
                (row["away_team"], ac, hc, as_, hs, (100 - hp) if hp is not None else None),
            ]:
                if team not in acc:
                    acc[team] = {"c_for": [], "c_ag": [], "s_for": [], "s_ag": [], "poss": [], "n": 0}
                acc[team]["c_for"].append(c_for)
                acc[team]["c_ag"].append(c_ag)
                acc[team]["s_for"].append(s_for)
                acc[team]["s_ag"].append(s_ag)
                if poss is not None:
                    acc[team]["poss"].append(poss)
                acc[team]["n"] += 1

    def _avg(lst):
        return round(sum(lst) / len(lst), 1) if lst else None

    result = []
    for team, d in sorted(acc.items()):
        result.append({
            "team":             team,
            "matches":          d["n"],
            "corners_for":      _avg(d["c_for"]),
            "corners_against":  _avg(d["c_ag"]),
            "corners_diff":     round(_avg(d["c_for"]) - _avg(d["c_ag"]), 1) if d["c_for"] else None,
            "shots_for":        _avg(d["s_for"]),
            "shots_against":    _avg(d["s_ag"]),
            "shots_diff":       round(_avg(d["s_for"]) - _avg(d["s_ag"]), 1) if d["s_for"] else None,
            "possession":       _avg(d["poss"]) if d["poss"] else None,
        })

    result.sort(key=lambda x: -(x["corners_for"] or 0))
    return jsonify(result)


@app.route("/api/strategies")
def api_strategies():
    team_db = load_team_db()
    try:
        with open("data/wc2026_groups.json") as f:
            groups = json.load(f)
    except FileNotFoundError:
        return jsonify({"error": "groups file not found"}), 404
    
    # Run a quick simulation to get qualification probabilities
    # Using 1000 sims for speed in the web response
    n_sims = 1000
    qual_counts = {t: 0 for t in team_db}
    from src.models.tournament import sim_group_stage
    
    for _ in range(n_sims):
        results, best_thirds, _, _ = sim_group_stage(groups, team_db)
        qualified = set()
        for letter, res in results.items():
            if not letter.startswith("_"):
                qualified.add(res['first'])
                qualified.add(res['second'])
        for t in best_thirds:
            qualified.add(t)
        for t in qualified:
            if t in qual_counts:
                qual_counts[t] += 1
                
    qual_probs = {t: count / n_sims for t, count in qual_counts.items()}
    
    def get_prob(name):
        return qual_probs.get(name, 0.5)

    # 90% Strategy: Elite Trio
    p_spain = get_prob("Spain")
    p_england = get_prob("England")
    p_france = get_prob("France")
    prob_90 = p_spain * p_england * p_france
    
    # 80% Strategy: Big Five
    p_argentina = get_prob("Argentina")
    p_germany = get_prob("Germany")
    prob_80 = prob_90 * p_argentina * p_germany

    return jsonify({
        "strategy_90": {
            "name": "THE 90% ULTRA-SAFE",
            "prob": round(prob_90 * 100, 2),
            "legs": [
                {"team": "Spain", "prob": round(p_spain * 100, 1)},
                {"team": "England", "prob": round(p_england * 100, 1)},
                {"team": "France", "prob": round(p_france * 100, 1)}
            ]
        },
        "strategy_80": {
            "name": "THE 80% HIGH-VALUE",
            "prob": round(prob_80 * 100, 2),
            "legs": [
                {"team": "Spain", "prob": round(p_spain * 100, 1)},
                {"team": "England", "prob": round(p_england * 100, 1)},
                {"team": "France", "prob": round(p_france * 100, 1)},
                {"team": "Argentina", "prob": round(p_argentina * 100, 1)},
                {"team": "Germany", "prob": round(p_germany * 100, 1)}
            ]
        }
    })


if __name__ == "__main__":
    print("🌍 WC 2026 Predictor → http://localhost:5007")
    app.run(debug=True, port=5007, host="0.0.0.0")
