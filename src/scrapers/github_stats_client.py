"""
github_stats_client.py — National team stats from the martj42/international_results
GitHub dataset (https://github.com/martj42/international_results).

No API key required. Downloads the full CSV once per session (cached to disk),
then computes SOS-weighted FORMA / GF_AVG / GA_AVG using the same ELO weighting
logic as stats_client.py.

Extended stats (corners, shots, possession, xG) are NOT available from this
source — those fields will be None. The core model inputs (FORMA, GF_AVG,
GA_AVG) are fully accurate.
"""

import io
import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests

# ── Config ────────────────────────────────────────────────────────────────────
_CSV_URL   = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
_CSV_CACHE = Path(__file__).parent.parent.parent / "data" / "intl_results.csv"
_CSV_TTL   = 6 * 3600   # re-download after 6 hours

BASELINE_OPPONENT_ELO = 1750
_ALPHA_FORMA = 1.20   # increased from 0.70 — more aggressively discounts weak-opponent wins
_ALPHA_GF    = 0.70   # increased from 0.40
_ALPHA_GA    = 0.50   # increased from 0.30

# Map dataset team names → our internal names (ELO cache keys)
# Direction: dataset name → internal name
_DATASET_TO_INTERNAL = {
    "United States":            "USA",
    "South Korea":              "South Korea",
    "Ivory Coast":              "Ivory Coast",
    "DR Congo":                 "DR Congo",
    "Bosnia and Herzegovina":   "Bosnia & Herzegovina",
    "Curaçao":                  "Curaçao",
    "Cape Verde":               "Cape Verde",
    "Turkey":                   "Turkey",
    "Iran":                     "Iran",
    "North Macedonia":          "North Macedonia",
    "Trinidad and Tobago":      "Trinidad & Tobago",
    "St. Kitts and Nevis":      "Saint Kitts and Nevis",
    "Sao Tome and Principe":    "São Tomé and Príncipe",
}

# Reverse: our internal name → dataset name (for looking up a team's matches)
_INTERNAL_TO_DATASET = {v: k for k, v in _DATASET_TO_INTERNAL.items()}
# Add pass-throughs for teams whose names match in both systems
_PASS_THROUGH = [
    "Spain", "France", "England", "Germany", "Brazil", "Argentina",
    "Portugal", "Netherlands", "Belgium", "Croatia", "Morocco", "Japan",
    "Mexico", "Colombia", "Uruguay", "Switzerland", "Senegal", "Austria",
    "Norway", "Sweden", "Scotland", "Czech Republic", "Serbia", "Ukraine",
    "Ecuador", "Canada", "Poland", "Australia", "Iran", "Tunisia", "Algeria",
    "Saudi Arabia", "Egypt", "Nigeria", "Cameroon", "Paraguay",
    "Costa Rica", "Panama", "Honduras", "Jamaica", "Bolivia", "Venezuela",
    "South Africa", "Iraq", "Indonesia", "Uzbekistan", "Slovakia", "Romania",
    "New Zealand", "El Salvador", "Guatemala", "Qatar",
    "Jordan", "Haiti", "Ghana", "Kenya", "Zimbabwe",
]
for t in _PASS_THROUGH:
    if t not in _INTERNAL_TO_DATASET:
        _INTERNAL_TO_DATASET[t] = t

# ── CSV download / cache ──────────────────────────────────────────────────────

_df_cache = None   # in-memory pandas DataFrame


def _load_csv() -> "pd.DataFrame":
    """Return the results DataFrame, downloading/refreshing as needed."""
    global _df_cache
    if _df_cache is not None:
        return _df_cache

    import pandas as pd

    # Use disk cache if fresh
    if _CSV_CACHE.exists():
        age = time.time() - _CSV_CACHE.stat().st_mtime
        if age < _CSV_TTL:
            _df_cache = pd.read_csv(_CSV_CACHE, parse_dates=["date"])
            return _df_cache

    # Download fresh copy
    try:
        resp = requests.get(_CSV_URL, timeout=20)
        resp.raise_for_status()
        _CSV_CACHE.parent.mkdir(exist_ok=True)
        _CSV_CACHE.write_text(resp.text, encoding="utf-8")
        _df_cache = pd.read_csv(io.StringIO(resp.text), parse_dates=["date"])
        print(f"  [github_stats] Downloaded {len(_df_cache):,} rows from martj42/international_results")
    except Exception as e:
        # Fall back to stale disk cache if available
        if _CSV_CACHE.exists():
            print(f"  [github_stats] Download failed ({e}), using stale cache")
            _df_cache = pd.read_csv(_CSV_CACHE, parse_dates=["date"])
        else:
            raise RuntimeError(f"Could not load international results CSV: {e}") from e

    return _df_cache


# ── ELO helpers (mirrors stats_client.py) ─────────────────────────────────────

_elo_cache: dict | None = None


def _load_elo_cache() -> dict:
    global _elo_cache
    if _elo_cache is None:
        elos_path = Path(__file__).parent.parent.parent / "data" / "elos.json"
        try:
            data = json.loads(elos_path.read_text())
            _elo_cache = data.get("data", {})
        except FileNotFoundError:
            _elo_cache = {}
    return _elo_cache


def _lookup_elo(opponent_name: str) -> float | None:
    """Resolve ELO for an opponent by name, with alias + fuzzy fallback."""
    elos = _load_elo_cache()
    # Normalise via dataset→internal map first
    name = _DATASET_TO_INTERNAL.get(opponent_name, opponent_name)
    if name in elos:
        return float(elos[name])
    name_lower = name.lower()
    for elo_name, elo_val in elos.items():
        if name_lower in elo_name.lower() or elo_name.lower() in name_lower:
            return float(elo_val)
    return None


def _match_weights(opp_elo: float) -> tuple[float, float, float]:
    ratio = opp_elo / BASELINE_OPPONENT_ELO
    return (
        ratio ** _ALPHA_FORMA,
        ratio ** _ALPHA_GF,
        ratio ** (-_ALPHA_GA),
    )


# ── Main public function ───────────────────────────────────────────────────────

def get_team_stats(team_name: str) -> dict | None:
    """
    Compute SOS-weighted stats for *team_name* from the GitHub CSV dataset.

    Returns the same dict schema as stats_client.get_team_stats(), with
    CORNERS_FOR / CORNERS_AGAINST / SOT_FOR / SOT_AGAINST / POSSESSION /
    YELLOW_CARDS / XG_FOR / XG_AGAINST all set to None (not available in
    this dataset).
    """
    try:
        df = _load_csv()
    except Exception as e:
        print(f"  [github_stats] Could not load CSV: {e}")
        return None

    # Resolve dataset team name
    dataset_name = _INTERNAL_TO_DATASET.get(team_name, team_name)

    # Filter to this team's completed matches (non-null scores)
    mask = (
        ((df["home_team"] == dataset_name) | (df["away_team"] == dataset_name))
        & df["home_score"].notna()
        & df["away_score"].notna()
    )
    team_df = df[mask].copy()

    if team_df.empty:
        print(f"  [github_stats] No matches found for '{team_name}' (dataset name: '{dataset_name}')")
        return None

    team_df = team_df.sort_values("date")
    recent = team_df.tail(15)

    match_data = []
    for _, row in recent.iterrows():
        is_home  = (row["home_team"] == dataset_name)
        gf       = float(row["home_score"] if is_home else row["away_score"])
        ga       = float(row["away_score"] if is_home else row["home_score"])
        opp_name = row["away_team"] if is_home else row["home_team"]

        pts     = 3 if gf > ga else (1 if gf == ga else 0)
        raw_elo = _lookup_elo(opp_name)
        eff_elo = raw_elo if raw_elo else BASELINE_OPPONENT_ELO

        match_data.append({
            "name":     opp_name,
            "elo":      raw_elo,
            "eff_elo":  eff_elo,
            "gf":       gf,
            "ga":       ga,
            "pts":      pts,
        })

    if not match_data:
        return None

    games = len(match_data)

    raw_forma   = round(sum(m["pts"]      for m in match_data) / games, 2)
    raw_gf      = round(sum(m["gf"]       for m in match_data) / games, 2)
    raw_ga      = round(sum(m["ga"]       for m in match_data) / games, 2)
    avg_opp_elo = sum(m["eff_elo"] for m in match_data) / games

    wf_sum = wg_sum = wga_sum = 0.0
    wf_pts = wg_gf  = wga_ga  = 0.0

    for m in match_data:
        w_forma, w_gf, w_ga_inv = _match_weights(m["eff_elo"])
        wf_sum  += w_forma;  wf_pts += m["pts"] * w_forma
        wg_sum  += w_gf;     wg_gf  += m["gf"]  * w_gf
        wga_sum += w_ga_inv; wga_ga += m["ga"]  * w_ga_inv

    adj_forma = round(max(0.5, min(3.0, wf_pts / wf_sum)), 2)
    adj_gf    = round(max(0.3, min(4.0, wg_gf  / wg_sum)), 2)
    adj_ga    = round(max(0.2, min(3.0, wga_ga / wga_sum)), 2)
    sos_ratio = round(avg_opp_elo / BASELINE_OPPONENT_ELO, 4)

    return {
        # ── Core model inputs (SOS-weighted) ──────────────────────────────
        "FORMA":            adj_forma,
        "GF_AVG":           adj_gf,
        "GA_AVG":           adj_ga,
        # ── Extended stats — not available from this source ───────────────
        "CORNERS_FOR":      None,
        "CORNERS_AGAINST":  None,
        "SOT_FOR":          None,
        "SOT_AGAINST":      None,
        "POSSESSION":       None,
        "YELLOW_CARDS":     None,
        "XG_FOR":           None,
        "XG_AGAINST":       None,
        # ── Raw unweighted values ─────────────────────────────────────────
        "RAW_FORMA":        raw_forma,
        "RAW_GF":           raw_gf,
        "RAW_GA":           raw_ga,
        # ── SOS diagnostics ───────────────────────────────────────────────
        "SOS_RATIO":        sos_ratio,
        "AVG_OPP_ELO":      round(avg_opp_elo, 0),
        "SOS_FORMA_MULT":   round((avg_opp_elo / BASELINE_OPPONENT_ELO) ** _ALPHA_FORMA, 4),
        "OPPONENTS": [
            {"name": m["name"], "elo": m["elo"], "gf": int(m["gf"]), "ga": int(m["ga"])}
            for m in match_data
        ],
    }


if __name__ == "__main__":
    print("=== GitHub CSV Stats Test ===\n")
    test_teams = ["Argentina", "Spain", "USA", "Japan", "Mexico", "South Korea",
                  "Switzerland", "Senegal", "Ivory Coast", "DR Congo", "Bosnia & Herzegovina"]
    for team in test_teams:
        stats = get_team_stats(team)
        if stats:
            print(f"{team}:")
            print(f"  Raw:      FORMA={stats['RAW_FORMA']}  GF={stats['RAW_GF']}  GA={stats['RAW_GA']}")
            print(f"  Weighted: FORMA={stats['FORMA']}  GF={stats['GF_AVG']}  GA={stats['GA_AVG']}")
            print(f"  SOS ratio={stats['SOS_RATIO']}  avg_opp_elo={stats['AVG_OPP_ELO']}")
            print(f"  Last {len(stats['OPPONENTS'])} opponents: {[o['name'] for o in stats['OPPONENTS'][-3:]]}")
        else:
            print(f"{team}: no data")
        print()
