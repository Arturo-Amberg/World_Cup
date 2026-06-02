"""
statsbomb_collector.py — Downloads and processes StatsBomb Open Data.

Pulls free open-data JSON directly from GitHub (no auth required).
Aggregates event-level data into match-level stats per team.

Competitions covered:
  - FIFA World Cup: 2022, 2018, 1990, 1986, 1974, 1970, 1962, 1958
  - UEFA Euro: 2024, 2020
  - Copa America: 2024
  - African Cup of Nations: 2023

Stats extracted per match (home + away):
  shots, shots_on_target, corners, free_kicks, fouls, yellow_cards,
  red_cards, possession_pct, pressures, passes_attempted, passes_completed

Output: data/statsbomb_match_stats.csv
"""

import requests
import json
import time
import pandas as pd
from pathlib import Path
from collections import defaultdict

BASE_RAW = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
OUTPUT   = Path("data/statsbomb_match_stats.csv")

# Competition/season pairs to pull (competition_id, season_id, label)
TARGETS = [
    (43,  106, "FIFA World Cup", "2022"),
    (43,    3, "FIFA World Cup", "2018"),
    (43,   55, "FIFA World Cup", "1990"),
    (43,   54, "FIFA World Cup", "1986"),
    (43,   51, "FIFA World Cup", "1974"),
    (43,  272, "FIFA World Cup", "1970"),
    (43,  270, "FIFA World Cup", "1962"),
    (43,  269, "FIFA World Cup", "1958"),
    (55,  282, "UEFA Euro",      "2024"),
    (55,   43, "UEFA Euro",      "2020"),
    (223, 282, "Copa America",   "2024"),
    (1267,107, "AFCON",          "2023"),
]

DELAY = 0.3   # seconds between requests — GitHub CDN is generous


def _get_json(url: str, retries: int = 3):
    for i in range(retries):
        try:
            r = requests.get(url, timeout=15)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 404:
                return None
            time.sleep(2)
        except Exception as e:
            print(f"    Retry {i+1}: {e}")
            time.sleep(3)
    return None


def get_matches(competition_id: int, season_id: int) -> list:
    url = f"{BASE_RAW}/matches/{competition_id}/{season_id}.json"
    data = _get_json(url)
    return data or []


def aggregate_events(match_id: int) -> dict | None:
    """
    Download event JSON for one match, aggregate into per-team box stats.
    Returns dict keyed by team_id with stats, or None on failure.
    """
    url = f"{BASE_RAW}/events/{match_id}.json"
    events = _get_json(url)
    if events is None:
        return None

    teams: dict[int, str] = {}  # id -> name
    stats: dict[int, dict] = defaultdict(lambda: defaultdict(int))

    for ev in events:
        team = ev.get("team")
        if not team:
            continue
        tid  = team["id"]
        tname = team["name"]
        teams[tid] = tname

        ev_type = ev.get("type", {}).get("name", "")
        player  = ev.get("player")

        # ── Shots ────────────────────────────────────────────────────────────
        if ev_type == "Shot":
            stats[tid]["shots"] += 1
            outcome = ev.get("shot", {}).get("outcome", {}).get("name", "")
            if outcome in ("Goal", "Saved", "Saved To Post"):
                stats[tid]["shots_on_target"] += 1

        # ── Passes (corners, free kicks, passes completed) ────────────────
        elif ev_type == "Pass":
            pass_data = ev.get("pass", {})
            pass_type = pass_data.get("type", {}).get("name", "")  # Corner, Free Kick, Goal Kick, etc.
            outcome   = pass_data.get("outcome", {}).get("name", "")

            if pass_type == "Corner":
                stats[tid]["corners"] += 1
            elif pass_type == "Free Kick":
                stats[tid]["free_kicks"] += 1

            stats[tid]["passes_attempted"] += 1
            if outcome not in ("Incomplete", "Out", "Pass Offside"):
                stats[tid]["passes_completed"] += 1

        # ── Fouls & Cards ──────────────────────────────────────────────────
        elif ev_type == "Foul Committed":
            stats[tid]["fouls"] += 1
            card = ev.get("foul_committed", {}).get("card", {}).get("name", "")
            if "Yellow" in card and "Second" not in card:
                stats[tid]["yellow_cards"] += 1
            elif "Red" in card or "Second Yellow" in card:
                stats[tid]["red_cards"] += 1

        # ── Bad Behaviour (cards given separately) ────────────────────────
        elif ev_type == "Bad Behaviour":
            card = ev.get("bad_behaviour", {}).get("card", {}).get("name", "")
            if "Yellow" in card:
                stats[tid]["yellow_cards"] += 1
            elif "Red" in card or "Second Yellow" in card:
                stats[tid]["red_cards"] += 1

        # ── Pressure ──────────────────────────────────────────────────────
        elif ev_type == "Pressure":
            stats[tid]["pressures"] += 1

    # ── Possession % from carry+pass+shot+dribble events ─────────────────────
    poss_events: dict[int, int] = defaultdict(int)
    total_poss = 0
    for ev in events:
        team = ev.get("team")
        if not team:
            continue
        ev_type = ev.get("type", {}).get("name", "")
        if ev_type in ("Pass", "Carry", "Shot", "Dribble", "Clearance",
                       "Interception", "Ball Receipt*", "Pressure", "Duel"):
            poss_events[team["id"]] += 1
            total_poss += 1

    if total_poss > 0:
        for tid in teams:
            stats[tid]["possession_pct"] = round(100.0 * poss_events[tid] / total_poss, 1)

    return {tid: dict(stats[tid]) for tid in teams}, teams


def process_competition(comp_id: int, season_id: int, comp_name: str, season: str) -> list[dict]:
    print(f"\n  {comp_name} {season}  (comp={comp_id}, season={season_id})")
    matches = get_matches(comp_id, season_id)
    print(f"    {len(matches)} matches found")

    rows = []
    for m in matches:
        mid       = m["match_id"]
        date      = m.get("match_date", "")
        home_team = m.get("home_team", {}).get("home_team_name", "")
        away_team = m.get("away_team", {}).get("away_team_name", "")
        home_id   = m.get("home_team", {}).get("home_team_id")
        away_id   = m.get("away_team", {}).get("away_team_id")
        home_score= m.get("home_score")
        away_score= m.get("away_score")

        time.sleep(DELAY)
        result = aggregate_events(mid)
        if result is None:
            print(f"      ✗ {date}  {home_team} vs {away_team}")
            continue

        ev_stats, _ = result

        def _get(tid, key):
            return ev_stats.get(tid, {}).get(key, 0)

        row = {
            "match_id":              mid,
            "date":                  date,
            "home_team":             home_team,
            "away_team":             away_team,
            "home_score":            home_score,
            "away_score":            away_score,
            "competition":           comp_name,
            "season":                season,
            # ── Home stats ────────────────────────────────────────────────
            "home_shots":            _get(home_id, "shots"),
            "home_shots_on_target":  _get(home_id, "shots_on_target"),
            "home_corners":          _get(home_id, "corners"),
            "home_free_kicks":       _get(home_id, "free_kicks"),
            "home_fouls":            _get(home_id, "fouls"),
            "home_yellow_cards":     _get(home_id, "yellow_cards"),
            "home_red_cards":        _get(home_id, "red_cards"),
            "home_possession_pct":   _get(home_id, "possession_pct"),
            "home_pressures":        _get(home_id, "pressures"),
            "home_passes_attempted": _get(home_id, "passes_attempted"),
            "home_passes_completed": _get(home_id, "passes_completed"),
            # ── Away stats ────────────────────────────────────────────────
            "away_shots":            _get(away_id, "shots"),
            "away_shots_on_target":  _get(away_id, "shots_on_target"),
            "away_corners":          _get(away_id, "corners"),
            "away_free_kicks":       _get(away_id, "free_kicks"),
            "away_fouls":            _get(away_id, "fouls"),
            "away_yellow_cards":     _get(away_id, "yellow_cards"),
            "away_red_cards":        _get(away_id, "red_cards"),
            "away_possession_pct":   _get(away_id, "possession_pct"),
            "away_pressures":        _get(away_id, "pressures"),
            "away_passes_attempted": _get(away_id, "passes_attempted"),
            "away_passes_completed": _get(away_id, "passes_completed"),
        }
        rows.append(row)
        print(f"      ✓ {date}  {home_team} {home_score}-{away_score} {away_team}"
              f"  Sh:{_get(home_id,'shots')}-{_get(away_id,'shots')}"
              f"  Cor:{_get(home_id,'corners')}-{_get(away_id,'corners')}"
              f"  Poss:{_get(home_id,'possession_pct')}%-{_get(away_id,'possession_pct')}%")
    return rows


def main():
    print("=" * 65)
    print("StatsBomb Open Data Collector")
    print("=" * 65)

    # Load already collected match IDs
    existing_ids: set[int] = set()
    if OUTPUT.exists():
        existing = pd.read_csv(OUTPUT)
        existing_ids = set(existing["match_id"].dropna().astype(int))
        print(f"Resuming — {len(existing_ids)} matches already saved.")

    all_rows = []
    for comp_id, season_id, comp_name, season in TARGETS:
        rows = process_competition(comp_id, season_id, comp_name, season)
        new_rows = [r for r in rows if r["match_id"] not in existing_ids]
        all_rows.extend(new_rows)

    if all_rows:
        df_new = pd.DataFrame(all_rows)
        if OUTPUT.exists():
            df_new.to_csv(OUTPUT, mode="a", index=False, header=False)
        else:
            df_new.to_csv(OUTPUT, index=False)
        total = len(existing_ids) + len(all_rows)
        print(f"\n{'='*65}")
        print(f"Done! Added {len(all_rows)} new rows → {total} total in {OUTPUT}")
    else:
        print("\nNo new rows to add.")


if __name__ == "__main__":
    main()
