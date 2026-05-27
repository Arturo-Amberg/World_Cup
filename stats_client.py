"""
stats_client.py — API-Football stats with Strength of Schedule (SOS) adjustment.

Why SOS matters:
  Argentina's FORMA=3.0 was built playing CONMEBOL qualifiers where they dominated.
  Spain's FORMA=2.2 came from UEFA Nations League vs Germany, France, Netherlands.
  Raw stats without SOS heavily bias teams that played weaker opposition.

SOS formula:
  sos_ratio = avg_opponent_elo / BASELINE_ELO   (BASELINE ≈ 1750)
  adj_forma  = raw_forma  * sos_ratio ** 0.70   (strongest adjustment)
  adj_gf     = raw_gf     * sos_ratio ** 0.40   (milder — scoring varies with SOS)
  adj_ga     = raw_ga     / sos_ratio ** 0.30   (inverse — less credit for low GA vs weak opponents)

Exponents < 1 prevent over-correction while still capturing the real quality signal.
"""

import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()
API_FOOTBALL_KEY = os.getenv("API_FOOTBALL_KEY")

HEADERS = {
    "x-rapidapi-key": API_FOOTBALL_KEY,
    "x-rapidapi-host": "v3.football.api-sports.io"
}
BASE_URL = "https://v3.football.api-sports.io"

# ELO baseline: average ELO of a typical competitive international opponent.
# UEFA qualifiers avg ~1780, CONMEBOL ~1820, CAF ~1680, AFC ~1700 — overall ~1750
BASELINE_OPPONENT_ELO = 1750

# Name aliases to bridge API-Football names → our ELO cache names
_NAME_MAP = {
    "Korea Republic":        "South Korea",
    "Republic of Korea":     "South Korea",
    "Côte d'Ivoire":         "Ivory Coast",
    "Cote d'Ivoire":         "Ivory Coast",
    "DR Congo":              "DR Congo",
    "Congo DR":              "DR Congo",
    "United States":         "USA",
    "United States of America": "USA",
    "IR Iran":               "Iran",
    "Bosnia":                "Bosnia & Herzegovina",
    "Bosnia-Herzegovina":    "Bosnia & Herzegovina",
    "Curacao":               "Curaçao",
    "Cabo Verde":            "Cape Verde",
    "China PR":              "China",
    "Türkiye":               "Turkey",
}

_elo_cache: dict | None = None


def _load_elo_cache() -> dict:
    """Load ELO ratings from local cache (no API call needed)."""
    global _elo_cache
    if _elo_cache is None:
        elos_path = os.path.join(os.path.dirname(__file__), "data", "elos.json")
        try:
            with open(elos_path) as f:
                data = json.load(f)
            _elo_cache = data.get("data", {})
        except FileNotFoundError:
            _elo_cache = {}
    return _elo_cache


def _lookup_elo(opponent_name: str) -> float | None:
    """Look up an opponent's ELO by name, with alias resolution."""
    elos = _load_elo_cache()
    name = _NAME_MAP.get(opponent_name, opponent_name)
    if name in elos:
        return float(elos[name])
    # Fuzzy fallback: check if any ELO key is a substring match
    name_lower = name.lower()
    for elo_name, elo_val in elos.items():
        if name_lower in elo_name.lower() or elo_name.lower() in name_lower:
            return float(elo_val)
    return None


def _sos_multiplier(avg_opp_elo: float) -> dict:
    """
    Compute SOS adjustment factors given average opponent ELO.

    Returns dict with keys: forma_mult, gf_mult, ga_mult, avg_opp_elo, sos_ratio.
    """
    ratio = avg_opp_elo / BASELINE_OPPONENT_ELO
    return {
        "forma_mult":   round(ratio ** 0.70, 4),  # Strong: wins vs elite count more
        "gf_mult":      round(ratio ** 0.40, 4),  # Mild: goals vs strong opponents count more
        "ga_mult":      round(ratio ** 0.30, 4),  # Inverse applied in caller: ga / ga_mult
        "avg_opp_elo":  round(avg_opp_elo, 0),
        "sos_ratio":    round(ratio, 4),
    }


def get_team_stats(team_name: str) -> dict | None:
    """
    Fetch FORMA/GF_AVG/GA_AVG for the last 10 completed matches via API-Football,
    then apply Strength of Schedule (SOS) adjustment based on average opponent ELO.

    Returns dict with keys:
        FORMA, GF_AVG, GA_AVG          — SOS-adjusted values (ready for model use)
        RAW_FORMA, RAW_GF, RAW_GA      — unadjusted raw values
        SOS_RATIO, AVG_OPP_ELO         — diagnostic info
        OPPONENTS                       — list of opponent names + ELOs
    """
    if not API_FOOTBALL_KEY or API_FOOTBALL_KEY == "tu_clave_api_football_aqui":
        print("API_FOOTBALL_KEY not configured.")
        return None

    try:
        # ── Step 1: Find team ID ──────────────────────────────────────────────
        search_res = requests.get(
            f"{BASE_URL}/teams", headers=HEADERS,
            params={"search": team_name}, timeout=10
        )
        search_res.raise_for_status()
        data = search_res.json()

        if not data.get("response"):
            return None

        team_id = None
        for t in data["response"]:
            if t["team"]["national"]:
                team_id = t["team"]["id"]
                break
        if not team_id:
            team_id = data["response"][0]["team"]["id"]

        # ── Step 2: Fetch fixtures (last 4 seasons) ───────────────────────────
        all_fixtures = []
        for year in [2023, 2024, 2025, 2026]:
            fix_res = requests.get(
                f"{BASE_URL}/fixtures", headers=HEADERS,
                params={"team": team_id, "season": year}, timeout=12
            )
            if fix_res.status_code == 200:
                resp = fix_res.json()
                if "response" in resp:
                    all_fixtures.extend(resp["response"])

        # ── Step 3: Filter completed matches, sort, take last 10 ─────────────
        played = [
            f for f in all_fixtures
            if f["fixture"]["status"]["short"] in ("FT", "AET", "PEN")
        ]
        played.sort(key=lambda x: x["fixture"]["timestamp"])
        recent = played[-10:]

        if not recent:
            return None

        # ── Step 4: Compute raw stats + collect opponent ELOs ─────────────────
        total_points = 0
        total_gf     = 0
        total_ga     = 0
        games        = 0
        opp_elos     = []
        opponents    = []

        for fix in recent:
            goals = fix["goals"]
            teams = fix["teams"]

            is_home    = (teams["home"]["id"] == team_id)
            gf         = goals["home"] if is_home else goals["away"]
            ga         = goals["away"] if is_home else goals["home"]
            opp_info   = teams["away"] if is_home else teams["home"]
            opp_name   = opp_info.get("name", "Unknown")

            if gf is None or ga is None:
                continue

            total_gf += gf
            total_ga += ga
            games    += 1

            if gf > ga:
                total_points += 3
            elif gf == ga:
                total_points += 1

            # Opponent ELO lookup
            opp_elo = _lookup_elo(opp_name)
            if opp_elo:
                opp_elos.append(opp_elo)
            opponents.append({
                "name": opp_name,
                "elo":  opp_elo,
                "gf":   gf,
                "ga":   ga,
            })

        if games == 0:
            return None

        raw_forma = round(total_points / games, 2)
        raw_gf    = round(total_gf / games, 2)
        raw_ga    = round(total_ga / games, 2)

        # ── Step 5: SOS adjustment ────────────────────────────────────────────
        if opp_elos:
            avg_opp_elo = sum(opp_elos) / len(opp_elos)
        else:
            avg_opp_elo = BASELINE_OPPONENT_ELO  # No data → no adjustment

        sos = _sos_multiplier(avg_opp_elo)

        adj_forma = round(raw_forma * sos["forma_mult"], 2)
        adj_gf    = round(raw_gf   * sos["gf_mult"],    2)
        adj_ga    = round(raw_ga   / sos["ga_mult"],     2)  # Inverse: high SOS → higher adj_ga

        # Clamp to reasonable international ranges
        adj_forma = max(0.5, min(3.0, adj_forma))
        adj_gf    = max(0.3, min(4.0, adj_gf))
        adj_ga    = max(0.2, min(3.0, adj_ga))

        return {
            # SOS-adjusted values — USE THESE in the model
            "FORMA":       adj_forma,
            "GF_AVG":      adj_gf,
            "GA_AVG":      adj_ga,
            # Raw unadjusted values — for transparency
            "RAW_FORMA":   raw_forma,
            "RAW_GF":      raw_gf,
            "RAW_GA":      raw_ga,
            # SOS diagnostics
            "SOS_RATIO":   sos["sos_ratio"],
            "AVG_OPP_ELO": sos["avg_opp_elo"],
            "SOS_FORMA_MULT": sos["forma_mult"],
            "OPPONENTS":   opponents,
        }

    except Exception as e:
        print(f"Error fetching API-Football stats for {team_name}: {e}")
        return None


if __name__ == "__main__":
    print("=== SOS-Adjusted Stats Test ===\n")
    for team in ["Argentina", "Spain", "France", "Tunisia"]:
        stats = get_team_stats(team)
        if stats:
            print(f"{team}:")
            print(f"  Raw:      FORMA={stats['RAW_FORMA']}  GF={stats['RAW_GF']}  GA={stats['RAW_GA']}")
            print(f"  Adj SOS:  FORMA={stats['FORMA']}      GF={stats['GF_AVG']}  GA={stats['GA_AVG']}")
            print(f"  SOS:      ratio={stats['SOS_RATIO']}  avg_opp_elo={stats['AVG_OPP_ELO']}")
            print(f"  Opponents: {[o['name'] for o in stats['OPPONENTS']]}")
            print()
        else:
            print(f"{team}: no data\n")
