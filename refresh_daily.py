#!/usr/bin/env python3
"""
refresh_daily.py — Morning stats refresh for World Cup 2026 predictor.

Runs automatically every morning via crontab.
Each run:
  1. Fetches extended stats (goals, corners, SOT, possession, cards)
     for the next BATCH_SIZE uncached teams, in priority order.
  2. Regenerates docs/static_data.js via the Flask test client.
  3. Commits and pushes the updated snapshot to GitHub Pages.

API budget: ~100 calls/day free tier → ~4 teams/day with fixture stats.
Once a team is cached it is never re-fetched until STATS_TTL (7 days) expires.
"""

import json
import os
import subprocess
import sys
import time
import logging
from datetime import date, datetime
from pathlib import Path

# ── setup ─────────────────────────────────────────────────────────────────────
os.chdir(Path(__file__).parent)

LOG_PATH = Path("data/refresh_daily.log")
LOG_PATH.parent.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ── config ────────────────────────────────────────────────────────────────────
BATCH_SIZE  = 4          # teams to fetch per run (keeps API calls ≤ ~80/day)
STATS_TTL   = 7 * 86400  # re-fetch after 7 days
DELAY       = 1.5        # seconds between team fetches
STOP_DATE   = date(2026, 6, 3)  # self-destruct after 4 daily runs (May 30–June 2)

# Priority order — most prediction-relevant teams first
PRIORITY = [
    "Spain", "France", "England", "Germany", "Brazil", "Argentina",
    "Portugal", "Netherlands", "Belgium", "Croatia", "Morocco", "Japan",
    "USA", "Mexico", "Colombia", "Uruguay", "South Korea", "Switzerland",
    "Senegal", "Austria", "Turkey", "Norway", "Sweden", "Scotland",
    "Czech Republic", "Serbia", "Ukraine", "Ecuador", "Canada", "Poland",
    "Australia", "Iran", "Tunisia", "Algeria", "Ivory Coast", "Ghana",
    "Saudi Arabia", "Egypt", "Nigeria", "Cameroon", "Paraguay",
    "Costa Rica", "Panama", "Honduras", "Jamaica", "Bolivia", "Venezuela",
    "South Africa", "Iraq", "Indonesia", "Uzbekistan", "Slovakia", "Romania",
    "New Zealand", "El Salvador", "Guatemala", "Qatar", "Bosnia & Herzegovina",
    "DR Congo", "Cape Verde", "Jordan", "Haiti", "Curaçao",
]

STATS_FILE   = Path("data/team_stats.json")
BRACKET_FILE = Path("data/bracket_consensus.json")


# ── helpers ───────────────────────────────────────────────────────────────────

def load_cache() -> dict:
    try:
        return json.loads(STATS_FILE.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_cache(cache: dict) -> None:
    STATS_FILE.write_text(json.dumps(cache, indent=2, ensure_ascii=False))


def pick_next_teams(cache: dict, n: int) -> list[str]:
    """Return up to n teams that are missing or whose cache is expired."""
    now = time.time()
    due = []
    for team in PRIORITY:
        entry = cache.get(team)
        age   = now - entry["ts"] if entry else float("inf")
        if age >= STATS_TTL:
            due.append(team)
        if len(due) == n:
            break
    return due


def fetch_teams(teams: list[str], cache: dict) -> tuple[list, list]:
    from stats_client import get_team_stats

    ok, failed = [], []
    for i, team in enumerate(teams):
        log.info(f"  [{i+1}/{len(teams)}] {team} …")
        try:
            stats = get_team_stats(team)
            if stats:
                cache[team] = {"ts": time.time(), "data": stats}
                save_cache(cache)
                has_ext = stats.get("CORNERS_FOR") is not None
                ext = (
                    f"Corners {stats['CORNERS_FOR']:.1f}/{stats['CORNERS_AGAINST']:.1f}  "
                    f"SOT {stats['SOT_FOR']:.1f}/{stats['SOT_AGAINST']:.1f}  "
                    f"Poss {stats['POSSESSION']:.0f}%  YC {stats['YELLOW_CARDS']:.2f}"
                ) if has_ext else "(goals only — no fixture stats yet)"
                log.info(f"    ✓ GF={stats['GF_AVG']}  GA={stats['GA_AVG']}  {ext}")
                ok.append(team)
            else:
                log.warning(f"    — no data returned (API miss or team not found)")
                failed.append(team)
        except Exception as e:
            log.error(f"    ✗ {e}")
            failed.append(team)

        if i < len(teams) - 1:
            time.sleep(DELAY)

    return ok, failed


def regenerate_static() -> bool:
    """Re-run generate_static_data.py via subprocess to get a clean import state."""
    log.info("Regenerating docs/static_data.js …")
    BRACKET_FILE.unlink(missing_ok=True)  # force fresh bracket sim
    result = subprocess.run(
        [sys.executable, "generate_static_data.py"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        log.error(f"generate_static_data.py failed:\n{result.stderr}")
        return False
    for line in result.stdout.strip().splitlines():
        log.info(f"  {line}")
    return True


def shutdown() -> None:
    """Remove crontab entry and cancel the pmset wake schedule, then exit."""
    log.info("STOP_DATE reached — removing cron job and wake schedule.")

    # Remove this script's line from crontab
    result = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
    if result.returncode == 0:
        new_tab = "\n".join(
            line for line in result.stdout.splitlines()
            if "refresh_daily.py" not in line
        )
        subprocess.run(["crontab", "-"], input=new_tab, text=True)
        log.info("  Crontab entry removed.")

    # Cancel the pmset wake schedule (requires sudo — tries passwordless first)
    r = subprocess.run(
        ["sudo", "-n", "pmset", "repeat", "cancel"],
        capture_output=True, text=True
    )
    if r.returncode == 0:
        log.info("  pmset wake schedule cancelled.")
    else:
        log.warning(
            "  Could not cancel pmset automatically (needs sudo password).\n"
            "  Run manually:  sudo pmset repeat cancel"
        )

    log.info("All done — refresh_daily.py has shut itself down.")
    sys.exit(0)


def git_push(teams_fetched: list[str]) -> bool:
    """Stage, commit, and push docs/static_data.js."""
    log.info("Pushing to GitHub Pages …")
    team_list = ", ".join(teams_fetched) if teams_fetched else "no new teams"
    msg = (
        f"Daily stats refresh: {team_list}\n\n"
        f"Auto-generated by refresh_daily.py  {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}\n\n"
        f"Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
    )
    cmds = [
        ["git", "add", "docs/static_data.js"],
        ["git", "commit", "-m", msg],
        ["git", "push"],
    ]
    for cmd in cmds:
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            # "nothing to commit" is fine
            if "nothing to commit" in r.stdout + r.stderr:
                log.info("  Nothing new to commit.")
                return True
            log.error(f"  git command failed: {' '.join(cmd)}\n  {r.stderr.strip()}")
            return False
    log.info("  Pushed ✓")
    return True


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("refresh_daily.py — starting")

    if date.today() >= STOP_DATE:
        shutdown()  # removes itself and exits

    cache = load_cache()
    log.info(f"Cache: {len(cache)} teams already have stats")

    teams = pick_next_teams(cache, BATCH_SIZE)
    if not teams:
        log.info("All teams are up to date — nothing to fetch today.")
    else:
        log.info(f"Fetching {len(teams)} teams: {teams}")
        ok, failed = fetch_teams(teams, cache)
        log.info(f"Fetched: {ok}  |  Failed/skipped: {failed}")

    if not regenerate_static():
        log.error("Static generation failed — aborting push.")
        sys.exit(1)

    git_push(teams)
    log.info("refresh_daily.py — done")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
