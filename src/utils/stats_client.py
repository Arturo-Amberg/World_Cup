"""
stats_client.py — API-Football stats with match-level SOS weighting.

Why SOS matters:
  Argentina's FORMA=3.0 was built playing CONMEBOL qualifiers where they dominated.
  Spain's FORMA=2.2 came from UEFA Nations League vs Germany, France, Netherlands.
  Raw stats without SOS heavily bias teams that played weaker opposition.

Match-level weighting (replaces single global multiplier):
  Each match is weighted individually by opponent ELO so goals/wins vs elite count more
  and blowouts vs minnows are discounted — rather than applying one multiplier to the
  whole average.

  w_i        = (opp_elo_i / BASELINE_ELO) ^ alpha
  adj_gf     = Σ(gf_i  × w_i)  / Σ(w_i)     alpha=0.40  (attack credit vs strength)
  adj_ga     = Σ(ga_i  / w_i)  / Σ(1/w_i)   alpha=0.30  (conceding vs weak hurts more)
  adj_forma  = Σ(pts_i × w_i)  / Σ(w_i)     alpha=0.70  (wins vs elite count most)

  Matches with unknown opponent ELO use BASELINE_ELO (weight=1, no adjustment).
  Uses last 15 matches for statistical stability of the per-match weighting.
"""

import os
import json
import time
import requests
from dotenv import load_dotenv

load_dotenv()

# Support two API keys — rotates to the backup automatically on 429
_API_KEYS = [
    k for k in [
        os.getenv("API_FOOTBALL_KEY"),
        os.getenv("API_FOOTBALL_KEY_2"),
    ] if k and k not in ("tu_clave_api_football_aqui", "")
]
_key_index = 0

BASE_URL = "https://v3.football.api-sports.io"


def _current_headers() -> dict:
    return {
        "x-rapidapi-key":  _API_KEYS[_key_index],
        "x-rapidapi-host": "v3.football.api-sports.io",
    }


# Keep HEADERS as an alias so any external code that imports it still works
API_FOOTBALL_KEY = _API_KEYS[0] if _API_KEYS else None
HEADERS = _current_headers()

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

# ── Rate limiter + key rotation ───────────────────────────────────────────────
# API-Football free tier: 10 req/min per key, ~100 req/day per key.
# Strategy: round-robin every request across all keys so both daily quotas
# are consumed evenly. Also rotate on 429 (per-minute limit) AND on quota
# exhaustion (API returns HTTP 200 but body has errors.requests field).
_last_request_at: float = 0.0
_MIN_INTERVAL: float = 7.0   # seconds between API calls — stays under 10 req/min


def _is_quota_error(resp) -> bool:
    """Return True if the response is a quota-exhausted error (HTTP 200 with error body)."""
    if resp.status_code != 200:
        return False
    try:
        body = resp.json()
        errors = body.get("errors", {})
        if isinstance(errors, dict) and errors:
            return True          # any error dict means quota/auth problem
        if isinstance(errors, list) and errors:
            return True
    except Exception:
        pass
    return False


def _api_get(url: str, headers: dict, params: dict, timeout: int = 12):
    """
    Rate-limited requests.get with round-robin key selection and rotation on
    both 429 (per-minute limit) and quota-exhausted body errors (HTTP 200 + errors).
    The `headers` parameter is accepted for API compatibility but ignored —
    we always build headers from the current active key.
    """
    global _last_request_at, _key_index

    # Enforce minimum interval between calls
    elapsed = time.time() - _last_request_at
    if elapsed < _MIN_INTERVAL:
        time.sleep(_MIN_INTERVAL - elapsed)
    _last_request_at = time.time()

    # Round-robin: advance key on every call so both daily quotas drain evenly
    _key_index = (_key_index + 1) % len(_API_KEYS)

    resp = requests.get(url, headers=_current_headers(), params=params, timeout=timeout)

    # Rotate on HTTP 429 (per-minute) or body quota error, try each key once
    if resp.status_code == 429 or _is_quota_error(resp):
        original_idx = _key_index
        for attempt in range(len(_API_KEYS) - 1):
            _key_index = (_key_index + 1) % len(_API_KEYS)
            time.sleep(2)
            _last_request_at = time.time()
            resp = requests.get(url, headers=_current_headers(), params=params, timeout=timeout)
            if resp.status_code != 429 and not _is_quota_error(resp):
                print(f"  [key rotation] key {original_idx+1} exhausted → switched to key {_key_index+1}")
                break   # good response, stop rotating

    resp.raise_for_status()
    return resp


_elo_cache: dict | None = None

# Persistent cache for per-fixture statistics (completed matches never change).
# Keyed by str(fixture_id) → {str(team_id): {stat_type: value}}.
_FIXTURE_STATS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "fixture_stats_cache.json")
_fixture_stats_mem: dict | None = None


def _load_fixture_stats_cache() -> dict:
    global _fixture_stats_mem
    if _fixture_stats_mem is None:
        try:
            with open(_FIXTURE_STATS_PATH) as f:
                _fixture_stats_mem = json.load(f)
        except FileNotFoundError:
            _fixture_stats_mem = {}
    return _fixture_stats_mem


def _save_fixture_stats_cache() -> None:
    os.makedirs(os.path.dirname(_FIXTURE_STATS_PATH), exist_ok=True)
    with open(_FIXTURE_STATS_PATH, "w") as f:
        json.dump(_fixture_stats_mem, f)


def _fetch_fixture_stats(fixture_id: int) -> dict | None:
    """
    Fetch statistics for one completed fixture, with permanent local cache.
    Returns {str(team_id): {stat_type: value}} or None on failure.
    """
    cache = _load_fixture_stats_cache()
    key = str(fixture_id)
    if key in cache:
        return cache[key]

    try:
        res = _api_get(
            f"{BASE_URL}/fixtures/statistics",
            headers=HEADERS,
            params={"fixture": fixture_id},
            timeout=10,
        )
        if res.status_code != 200:
            return None
        data = res.json()
        if not data.get("response"):
            return None

        stats_by_team: dict[str, dict] = {}
        for entry in data["response"]:
            tid = str(entry["team"]["id"])
            flat: dict = {}
            for s in entry["statistics"]:
                val = s["value"]
                if isinstance(val, str) and val.endswith("%"):
                    try:
                        val = float(val.rstrip("%"))
                    except ValueError:
                        val = None
                elif val is not None:
                    try:
                        val = float(val)
                    except (TypeError, ValueError):
                        val = None
                flat[s["type"]] = val
            stats_by_team[tid] = flat

        cache[key] = stats_by_team
        _save_fixture_stats_cache()
        return stats_by_team
    except Exception:
        return None


def _load_elo_cache() -> dict:
    """Load ELO ratings from local cache (no API call needed)."""
    global _elo_cache
    if _elo_cache is None:
        elos_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "elos.json")
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


# Alpha exponents — controls how aggressively opponent quality rescales each stat.
# Values < 1 prevent over-correction while preserving the real quality signal.
_ALPHA_FORMA = 0.70   # Wins vs elite count most
_ALPHA_GF    = 0.40   # Goals for — mild upweight vs strong opponents
_ALPHA_GA    = 0.30   # Goals against — inverse: conceding vs weak hurts more


def _match_weights(opp_elo: float) -> tuple[float, float, float]:
    """Return (w_forma, w_gf, w_ga_inv) for a single match given opponent ELO."""
    ratio = opp_elo / BASELINE_OPPONENT_ELO
    return (
        ratio ** _ALPHA_FORMA,
        ratio ** _ALPHA_GF,
        ratio ** (-_ALPHA_GA),   # inverse weight for GA
    )


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
        search_res = _api_get(
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
            fix_res = _api_get(
                f"{BASE_URL}/fixtures", headers=HEADERS,
                params={"team": team_id, "season": year}, timeout=12
            )
            if fix_res.status_code == 200:
                resp = fix_res.json()
                if "response" in resp:
                    all_fixtures.extend(resp["response"])

        # ── Step 3: Filter completed matches, sort, take last 15 ─────────────
        # 15 matches gives the per-match ELO weighting enough data to be stable.
        played = [
            f for f in all_fixtures
            if f["fixture"]["status"]["short"] in ("FT", "AET", "PEN")
        ]
        played.sort(key=lambda x: x["fixture"]["timestamp"])
        recent = played[-15:]

        if not recent:
            return None

        # ── Step 4: Parse matches, resolve opponent ELOs + fetch fixture stats ──
        match_data = []
        for fix in recent:
            goals    = fix["goals"]
            teams    = fix["teams"]
            fix_id   = fix["fixture"]["id"]

            is_home  = (teams["home"]["id"] == team_id)
            gf       = goals["home"] if is_home else goals["away"]
            ga       = goals["away"] if is_home else goals["home"]
            our_side = "home" if is_home else "away"
            opp_side = "away" if is_home else "home"
            opp_info = teams[opp_side]
            opp_name = opp_info.get("name", "Unknown")
            opp_id   = str(opp_info.get("id", ""))

            if gf is None or ga is None:
                continue

            pts     = 3 if gf > ga else (1 if gf == ga else 0)
            raw_elo = _lookup_elo(opp_name)
            eff_elo = raw_elo if raw_elo else BASELINE_OPPONENT_ELO

            # ── Per-fixture extended stats ────────────────────────────────────
            fx_stats = _fetch_fixture_stats(fix_id) or {}
            our_s = fx_stats.get(str(team_id), {})
            opp_s = fx_stats.get(opp_id, {})

            # Our corners = "Corner Kicks" in our stats
            # Opponent corners for us = "Corner Kicks" in opp stats (= our corners against)
            corners_for     = our_s.get("Corner Kicks")
            corners_against = opp_s.get("Corner Kicks")
            sot_for         = our_s.get("Shots on Goal")
            sot_against     = opp_s.get("Shots on Goal")
            possession      = our_s.get("Ball Possession")   # already parsed to float (55.0 etc.)
            yellow_cards    = our_s.get("Yellow Cards", 0) or 0
            xg_for          = our_s.get("expected_goals")
            xg_against      = opp_s.get("expected_goals")

            match_data.append({
                "name":            opp_name,
                "elo":             raw_elo,
                "eff_elo":         eff_elo,
                "gf":              gf,
                "ga":              ga,
                "pts":             pts,
                "corners_for":     corners_for,
                "corners_against": corners_against,
                "sot_for":         sot_for,
                "sot_against":     sot_against,
                "possession":      possession,
                "yellow_cards":    yellow_cards,
                "xg_for":          xg_for,
                "xg_against":      xg_against,
            })

        if not match_data:
            return None

        games = len(match_data)

        # ── Step 5: Match-level ELO-weighted averages ─────────────────────────
        # Each match contributes proportionally to opponent quality rather than
        # equally — a 3-0 win over France counts more than a 3-0 win over Andorra.
        # Same weighting logic applied to all stats (goals, corners, shots, cards).

        raw_forma   = round(sum(m["pts"] for m in match_data) / games, 2)
        raw_gf      = round(sum(m["gf"]  for m in match_data) / games, 2)
        raw_ga      = round(sum(m["ga"]  for m in match_data) / games, 2)
        avg_opp_elo = sum(m["eff_elo"] for m in match_data) / games

        wf_sum = wg_sum = wga_sum = 0.0
        wf_pts = wg_gf  = wga_ga  = 0.0

        # Extended stat accumulators — only count matches where stat was available
        wc_sum = wca_sum = wsot_sum = wsota_sum = wposs_sum = wyc_sum = wxgf_sum = wxga_sum = 0.0
        wc_cf  = wca_ca  = wsot_s   = wsota_s   = wposs_p  = wyc_y  = wxgf_x   = wxga_x   = 0.0

        for m in match_data:
            w_forma, w_gf, w_ga_inv = _match_weights(m["eff_elo"])
            wf_sum  += w_forma;  wf_pts += m["pts"] * w_forma
            wg_sum  += w_gf;     wg_gf  += m["gf"]  * w_gf
            wga_sum += w_ga_inv; wga_ga += m["ga"]  * w_ga_inv

            # Corners — same attack/defense weighting as goals
            if m["corners_for"] is not None:
                wc_sum += w_gf;     wc_cf  += m["corners_for"]  * w_gf
            if m["corners_against"] is not None:
                wca_sum += w_ga_inv; wca_ca += m["corners_against"] * w_ga_inv

            # Shots on target — same direction as GF
            if m["sot_for"] is not None:
                wsot_sum  += w_gf;     wsot_s  += m["sot_for"]     * w_gf
            if m["sot_against"] is not None:
                wsota_sum += w_ga_inv; wsota_s += m["sot_against"] * w_ga_inv

            # Possession — mildly weighted by strength (facing elites, poss is harder to keep)
            if m["possession"] is not None:
                wposs_sum += w_forma; wposs_p += m["possession"] * w_forma

            # Yellow cards — inverse weight: cards vs weak opponents are more meaningful
            if m["yellow_cards"] is not None:
                wyc_sum += w_ga_inv; wyc_y += m["yellow_cards"] * w_ga_inv

            # xG — same as goals
            if m["xg_for"] is not None:
                wxgf_sum += w_gf;     wxgf_x += m["xg_for"]     * w_gf
            if m["xg_against"] is not None:
                wxga_sum += w_ga_inv; wxga_x += m["xg_against"] * w_ga_inv

        adj_forma = round(max(0.5, min(3.0, wf_pts / wf_sum)), 2)
        adj_gf    = round(max(0.3, min(4.0, wg_gf  / wg_sum)), 2)
        adj_ga    = round(max(0.2, min(3.0, wga_ga / wga_sum)), 2)

        # Extended stats — fall back to None if no data was available in any match
        def _wavg(num, den, lo, hi):
            return round(max(lo, min(hi, num / den)), 2) if den > 0 else None

        adj_corners_for     = _wavg(wc_cf,  wc_sum,  1.0, 12.0)
        adj_corners_against = _wavg(wca_ca, wca_sum, 1.0, 12.0)
        adj_sot_for         = _wavg(wsot_s,  wsot_sum,  0.5, 12.0)
        adj_sot_against     = _wavg(wsota_s, wsota_sum, 0.5, 12.0)
        adj_possession      = _wavg(wposs_p, wposs_sum, 25.0, 75.0)
        adj_yellow_cards    = _wavg(wyc_y,  wyc_sum,  0.2, 6.0)
        adj_xg_for          = _wavg(wxgf_x, wxgf_sum, 0.1, 5.0)
        adj_xg_against      = _wavg(wxga_x, wxga_sum, 0.1, 5.0)

        sos_ratio = round(avg_opp_elo / BASELINE_OPPONENT_ELO, 4)

        return {
            # ── Core model inputs (SOS-weighted) ──────────────────────────────
            "FORMA":            adj_forma,
            "GF_AVG":           adj_gf,
            "GA_AVG":           adj_ga,
            # ── Extended stats (SOS-weighted, None if no fixture data) ─────────
            "CORNERS_FOR":      adj_corners_for,
            "CORNERS_AGAINST":  adj_corners_against,
            "SOT_FOR":          adj_sot_for,
            "SOT_AGAINST":      adj_sot_against,
            "POSSESSION":       adj_possession,
            "YELLOW_CARDS":     adj_yellow_cards,
            "XG_FOR":           adj_xg_for,
            "XG_AGAINST":       adj_xg_against,
            # ── Raw unweighted values ─────────────────────────────────────────
            "RAW_FORMA":        raw_forma,
            "RAW_GF":           raw_gf,
            "RAW_GA":           raw_ga,
            # ── SOS diagnostics ───────────────────────────────────────────────
            "SOS_RATIO":        sos_ratio,
            "AVG_OPP_ELO":      round(avg_opp_elo, 0),
            "SOS_FORMA_MULT":   round((avg_opp_elo / BASELINE_OPPONENT_ELO) ** _ALPHA_FORMA, 4),
            "OPPONENTS":        [{"name": m["name"], "elo": m["elo"], "gf": m["gf"], "ga": m["ga"]} for m in match_data],
        }

    except Exception as e:
        print(f"Error fetching API-Football stats for {team_name}: {e}")
        return None


if __name__ == "__main__":
    print("=== Match-Level SOS-Weighted Stats Test ===\n")
    for team in ["Argentina", "Spain", "France", "Tunisia"]:
        stats = get_team_stats(team)
        if stats:
            print(f"{team}:")
            print(f"  Raw:      FORMA={stats['RAW_FORMA']}  GF={stats['RAW_GF']}  GA={stats['RAW_GA']}")
            print(f"  Weighted: FORMA={stats['FORMA']}      GF={stats['GF_AVG']}  GA={stats['GA_AVG']}")
            print(f"  Corners:  FOR={stats['CORNERS_FOR']}  AGAINST={stats['CORNERS_AGAINST']}")
            print(f"  Shots OT: FOR={stats['SOT_FOR']}      AGAINST={stats['SOT_AGAINST']}")
            print(f"  Poss={stats['POSSESSION']}%  YellowCards={stats['YELLOW_CARDS']}  xG_for={stats['XG_FOR']}  xG_ag={stats['XG_AGAINST']}")
            print(f"  SOS:      ratio={stats['SOS_RATIO']}  avg_opp_elo={stats['AVG_OPP_ELO']}")
            print(f"  Opponents ({len(stats['OPPONENTS'])} matches):")
            for o in stats["OPPONENTS"]:
                elo_str = f"ELO {o['elo']:.0f}" if o["elo"] else "ELO ?"
                print(f"    {o['name']:<25} {elo_str}  {o['gf']}-{o['ga']}")
            print()
        else:
            print(f"{team}: no data\n")
