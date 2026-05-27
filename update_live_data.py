"""
update_live_data.py — Smart live data updater for WC 2026.

Respects the 100 req/day API-Football free-tier limit by:
  1. Checking current usage before fetching
  2. Prioritizing top contenders first
  3. Estimating calls needed per team (1 search + up to 4 season queries)
  4. Stopping gracefully when quota is close to exhausted

Usage:
    python3 update_live_data.py             # Update top priority teams
    python3 update_live_data.py --all       # All teams (may hit quota)
    python3 update_live_data.py --status    # Show current quota + cached data
    python3 update_live_data.py --elos      # Refresh ELO ratings only
    python3 update_live_data.py --injuries  # Refresh injuries (manual fallback)
"""

import os, sys, json, time, requests
from dotenv import load_dotenv

load_dotenv()

API_KEY  = os.getenv("API_FOOTBALL_KEY")
BASE_URL = "https://v3.football.api-sports.io"
HEADERS  = {"x-rapidapi-key": API_KEY or "", "x-rapidapi-host": "v3.football.api-sports.io"}
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# ─── Priority order: most impactful teams for the simulation ───────────────
PRIORITY_TEAMS = [
    # Top contenders — fetch first, most impactful on simulation
    "Spain", "France", "Argentina", "Brazil", "England",
    "Portugal", "Netherlands", "Germany", "Colombia", "Belgium",
    "Croatia", "Italy", "Japan", "Uruguay", "Morocco", "Sweden",
    # Second tier
    "USA", "Mexico", "Canada", "Switzerland", "Senegal", "Ecuador",
    "South Korea", "Australia", "Serbia", "Turkey", "Iran", "Austria",
    # Third tier
    "Nigeria", "Ghana", "Ivory Coast", "Tunisia", "Algeria", "Egypt",
    "Poland", "Ukraine", "Slovakia", "Romania", "Paraguay", "Bolivia",
    "Venezuela", "Panama", "Costa Rica", "Honduras", "Jamaica",
    "El Salvador", "Guatemala", "Indonesia", "Uzbekistan", "South Africa",
    "Iraq", "New Zealand",
]

CALLS_PER_TEAM = 5  # 1 search + 4 season fixtures calls (conservative estimate)
SAFETY_MARGIN  = 10  # Reserve this many calls as buffer


def check_quota() -> dict:
    """Query current API usage."""
    try:
        r = requests.get(f"{BASE_URL}/status", headers=HEADERS, timeout=8)
        r.raise_for_status()
        resp = r.json().get("response", {})
        reqs = resp.get("requests", {})
        return {
            "used":      reqs.get("current", 0),
            "limit":     reqs.get("limit_day", 100),
            "remaining": reqs.get("limit_day", 100) - reqs.get("current", 0),
            "plan":      resp.get("subscription", {}).get("plan", "Unknown"),
        }
    except Exception as e:
        print(f"  ⚠ Could not check quota: {e}")
        return {"used": 0, "limit": 100, "remaining": 100, "plan": "Unknown"}


def print_status():
    """Print current quota and cached data state."""
    quota = check_quota()
    print(f"\n{'='*52}")
    print(f"  API-Football  |  Plan: {quota['plan']}")
    print(f"  Used today:   {quota['used']}/{quota['limit']}  ({quota['remaining']} remaining)")
    print(f"  Teams fetchable today: ~{quota['remaining'] // CALLS_PER_TEAM}")
    print(f"{'='*52}")

    # Stats cache
    stats_path = os.path.join(DATA_DIR, "team_stats.json")
    try:
        with open(stats_path) as f:
            stats = json.load(f)
        now = time.time()
        print(f"\n  Cached team stats ({len(stats)} teams):")
        for team in PRIORITY_TEAMS:
            if team in stats:
                age_h = int((now - stats[team].get("ts", 0)) / 3600)
                d = stats[team]["data"]
                flag = "✅" if age_h < 48 else "🟡" if age_h < 168 else "🔴"
                print(f"    {flag} {team:<20} FORMA={d['FORMA']}  GF={d['GF_AVG']}  GA={d['GA_AVG']}  ({age_h}h ago)")
    except FileNotFoundError:
        print("  No stats cache found.")

    # Injuries
    inj_path = os.path.join(DATA_DIR, "injuries.json")
    try:
        with open(inj_path) as f:
            inj = json.load(f)
        injured = {k: v for k, v in inj.items()
                   if not k.startswith("_") and v.get("count", 0) > 0}
        if injured:
            print(f"\n  Teams with injuries:")
            for team, info in sorted(injured.items(), key=lambda x: -x[1]["count"]):
                names = ", ".join(p["name"] for p in info.get("players", [])[:3])
                print(f"    ⚠  {team}: {info['count']} — {names}")
        else:
            print(f"\n  Injuries: none recorded (or API returned empty)")
    except FileNotFoundError:
        print("  No injuries file found.")

    print()


def fetch_team_stats_smart(team_name: str) -> dict | None:
    """Fetch team stats with smarter season handling and fewer API calls."""
    from stats_client import get_team_stats
    return get_team_stats(team_name)


def update_stats(teams: list, quota: dict):
    """Fetch stats for teams, stopping before quota exhausted."""
    available = quota["remaining"] - SAFETY_MARGIN
    can_fetch  = max(0, available // CALLS_PER_TEAM)

    if can_fetch == 0:
        print(f"  ⛔ Not enough quota to fetch any teams today "
              f"({quota['remaining']} calls left, need {CALLS_PER_TEAM}+ per team).")
        print(f"     Quota resets at midnight UTC.")
        return

    print(f"  Quota: {quota['remaining']} calls remaining → can fetch ~{can_fetch} teams today")
    if len(teams) > can_fetch:
        print(f"  ⚠  Will fetch top {can_fetch} priority teams (of {len(teams)} requested)")
        teams = teams[:can_fetch]

    stats_path = os.path.join(DATA_DIR, "team_stats.json")
    try:
        with open(stats_path) as f:
            cache = json.load(f)
    except FileNotFoundError:
        cache = {}

    ok, fail = 0, []
    for i, team in enumerate(teams, 1):
        print(f"  ({i:2}/{len(teams)}) {team:<22}", end="", flush=True)
        stats = fetch_team_stats_smart(team)
        if stats:
            cache[team] = {"ts": time.time(), "data": stats}
            with open(stats_path, "w") as f:
                json.dump(cache, f, indent=2, ensure_ascii=False)
            print(f"✅  FORMA={stats['FORMA']}  GF={stats['GF_AVG']}  GA={stats['GA_AVG']}")
            ok += 1
        else:
            print("❌  no data (not found or API error)")
            fail.append(team)

        # Delay to stay within rate limits
        time.sleep(8)
        # Extra pause every 5 teams
        if i % 5 == 0 and i < len(teams):
            print(f"     ⏳ Pausing 20s (rate limit buffer)...")
            time.sleep(20)

    print(f"\n  ✅ Stats updated: {ok}/{len(teams)} teams")
    if fail:
        print(f"  ❌ Failed: {', '.join(fail)}")


def update_injuries_manual():
    """
    Fallback: since API-Football free tier doesn't provide national team
    injury data, this function allows you to enter known injuries manually.
    
    Edit the KNOWN_INJURIES dict below with current real-world information.
    Source: official team press conferences, trusted sports outlets (BBC Sport,
    Marca, L'Equipe, ESPN) before each tournament phase.
    """
    # ─── EDIT THIS DICT with real pre-tournament injury news ──────────────────
    # Format: "TeamName": ["Player Name", "Player Name 2"]
    # Leave empty list [] if no known injuries.
    KNOWN_INJURIES = {
        "Spain":       [],                                  # Full squad available
        "France":      [],                                  # Mbappe fit
        "Argentina":   [],                                  # Messi confirmed
        "Brazil":      [],
        "England":     [],
        "Portugal":    [],
        "Netherlands": [],
        "Germany":     [],
        "Colombia":    [],
        "Belgium":     [],
        "Croatia":     [],
        "Italy":       [],
        "Japan":       [],
        "Uruguay":     [],
        "Morocco":     [],
        "Sweden":      [],
        "USA":         [],
        "Mexico":      [],
        "Canada":      [],
    }
    # ─────────────────────────────────────────────────────────────────────────

    inj_path = os.path.join(DATA_DIR, "injuries.json")
    try:
        with open(inj_path) as f:
            data = json.load(f)
    except FileNotFoundError:
        data = {}

    updated_count = 0
    for team, players in KNOWN_INJURIES.items():
        data[team] = {
            "count":   len(players),
            "players": [{"name": p, "position": "Unknown", "reason": "Manual"} for p in players],
            "updated": time.strftime("%Y-%m-%d %H:%M"),
        }
        updated_count += 1

    data["_updated"] = time.strftime("%Y-%m-%d %H:%M")
    with open(inj_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"  ✅ Manual injury data saved for {updated_count} teams.")
    injured = {t: p for t, p in KNOWN_INJURIES.items() if p}
    if injured:
        for team, players in injured.items():
            print(f"     ⚠  {team}: {', '.join(players)}")
    else:
        print("     ℹ  No injuries recorded (all teams at full strength).")


def refresh_elos():
    """Refresh ELO ratings from eloratings.net (free, no quota cost)."""
    from scraper import fetch_elo_ratings
    from cache import get_elos
    print("  Fetching from eloratings.net...")
    elos = get_elos(fetch_elo_ratings)
    if elos:
        matched = [t for t in PRIORITY_TEAMS if t in elos]
        print(f"  ✅ ELOs refreshed — {len(matched)}/{len(PRIORITY_TEAMS)} priority teams found")
        print(f"\n  Top 10 ELOs:")
        top = sorted(elos.items(), key=lambda x: -x[1])[:10]
        for i, (name, elo) in enumerate(top, 1):
            print(f"    {i:2}. {name:<20} {elo}")
    else:
        print("  ❌ Failed to fetch ELOs.")


def main():
    args = sys.argv[1:]

    print("=" * 52)
    print("  WC 2026 — Live Data Updater")
    print("=" * 52)

    if "--status" in args or not args:
        print_status()
        if not args:
            print("  Run with --stats, --injuries, --elos, or --all")
            print("  Example: python3 update_live_data.py --stats")
        return

    if "--elos" in args:
        print("\n[ELOs] Refreshing from eloratings.net (free, no quota)...")
        refresh_elos()

    if "--injuries" in args:
        print("\n[Injuries] Saving manual injury data...")
        print("  ℹ  API-Football free tier doesn't provide national team injuries.")
        print("  ℹ  Edit KNOWN_INJURIES in this script with current press info.")
        update_injuries_manual()

    if "--stats" in args or "--all" in args:
        print("\n[Stats] Fetching team FORMA/GF/GA from API-Football...")
        quota = check_quota()
        print(f"  Plan: {quota['plan']}  |  Used: {quota['used']}/{quota['limit']}  |  "
              f"Remaining: {quota['remaining']}")

        teams = PRIORITY_TEAMS if "--all" in args else PRIORITY_TEAMS[:20]
        update_stats(teams, quota)

    print("\n  Done. Run with --status to see updated cache.")


if __name__ == "__main__":
    main()
