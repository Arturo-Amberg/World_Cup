"""
fbref_scraper.py — Scrapes historical international match stats from FBref.

Uses curl_cffi to spoof Chrome's TLS fingerprint and bypass Cloudflare.

Covers:
  - FIFA World Cups (2010-2022)
  - UEFA Euros (2012-2024)
  - Copa America (2015-2024)
  - Africa Cup of Nations (2013-2023)
  - World Cup Qualifiers: UEFA / CONMEBOL / CONCACAF
  - UEFA Nations League (2018-2024)
  - CONCACAF Gold Cup / AFC Asian Cup

Output:
  data/fbref_match_stats.csv

Columns produced (per match):
  date, home_team, away_team, home_score, away_score, competition, season,
  home_poss, away_poss,
  home_shots, away_shots,
  home_shots_on_target, away_shots_on_target,
  home_corners, away_corners,
  home_fouls, away_fouls,
  home_yellows, away_yellows,
  home_reds, away_reds,
  match_url

Rate-limits to 4-7 seconds between requests to be polite to FBref.
Resumes from existing CSV if run again.
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import re
import os
from pathlib import Path
from datetime import datetime
from curl_cffi import requests as cffi_requests

# ─── Config ───────────────────────────────────────────────────────────────────
OUTPUT_PATH = Path("data/fbref_match_stats.csv")
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}
BASE_URL = "https://fbref.com"
MIN_DELAY = 4.0   # seconds between requests
MAX_DELAY = 7.0

# ─── FBref competition fixture list URLs ─────────────────────────────────────
# Format: (competition_name, season_label, fixtures_url)
COMPETITIONS = [
    # ── FIFA World Cups ────────────────────────────────────────────────────────
    ("FIFA World Cup", "2022", "/en/comps/1/2022/schedule/2022-FIFA-World-Cup-Scores-and-Fixtures"),
    ("FIFA World Cup", "2018", "/en/comps/1/2018/schedule/2018-FIFA-World-Cup-Scores-and-Fixtures"),
    ("FIFA World Cup", "2014", "/en/comps/1/2014/schedule/2014-FIFA-World-Cup-Scores-and-Fixtures"),
    ("FIFA World Cup", "2010", "/en/comps/1/2010/schedule/2010-FIFA-World-Cup-Scores-and-Fixtures"),

    # ── UEFA European Championship ────────────────────────────────────────────
    ("UEFA Euro", "2024", "/en/comps/676/2024/schedule/2024-European-Championship-Scores-and-Fixtures"),
    ("UEFA Euro", "2020", "/en/comps/676/2020/schedule/2020-European-Championship-Scores-and-Fixtures"),
    ("UEFA Euro", "2016", "/en/comps/676/2016/schedule/2016-European-Championship-Scores-and-Fixtures"),
    ("UEFA Euro", "2012", "/en/comps/676/2012/schedule/2012-European-Championship-Scores-and-Fixtures"),

    # ── Copa America ──────────────────────────────────────────────────────────
    ("Copa America", "2024", "/en/comps/685/2024/schedule/2024-Copa-America-Scores-and-Fixtures"),
    ("Copa America", "2021", "/en/comps/685/2021/schedule/2021-Copa-America-Scores-and-Fixtures"),
    ("Copa America", "2019", "/en/comps/685/2019/schedule/2019-Copa-America-Scores-and-Fixtures"),
    ("Copa America", "2016", "/en/comps/685/2016/schedule/2016-Copa-America-Centenario-Scores-and-Fixtures"),
    ("Copa America", "2015", "/en/comps/685/2015/schedule/2015-Copa-America-Scores-and-Fixtures"),

    # ── Africa Cup of Nations ─────────────────────────────────────────────────
    ("Africa Cup of Nations", "2023", "/en/comps/656/2023/schedule/2023-Africa-Cup-of-Nations-Scores-and-Fixtures"),
    ("Africa Cup of Nations", "2022", "/en/comps/656/2021/schedule/2021-Africa-Cup-of-Nations-Scores-and-Fixtures"),
    ("Africa Cup of Nations", "2019", "/en/comps/656/2019/schedule/2019-Africa-Cup-of-Nations-Scores-and-Fixtures"),
    ("Africa Cup of Nations", "2017", "/en/comps/656/2017/schedule/2017-Africa-Cup-of-Nations-Scores-and-Fixtures"),
    ("Africa Cup of Nations", "2015", "/en/comps/656/2015/schedule/2015-Africa-Cup-of-Nations-Scores-and-Fixtures"),

    # ── UEFA Nations League ───────────────────────────────────────────────────
    ("UEFA Nations League", "2024-25", "/en/comps/189/2024-2025/schedule/2024-2025-UEFA-Nations-League-Scores-and-Fixtures"),
    ("UEFA Nations League", "2022-23", "/en/comps/189/2022-2023/schedule/2022-2023-UEFA-Nations-League-Scores-and-Fixtures"),
    ("UEFA Nations League", "2020-21", "/en/comps/189/2020-2021/schedule/2020-2021-UEFA-Nations-League-Scores-and-Fixtures"),
    ("UEFA Nations League", "2018-19", "/en/comps/189/2018-2019/schedule/2018-2019-UEFA-Nations-League-Scores-and-Fixtures"),

    # ── World Cup Qualifiers: UEFA ────────────────────────────────────────────
    ("WC Qualifying UEFA", "2022", "/en/comps/68/2022/schedule/2022-FIFA-World-Cup-Qualifying-UEFA-Scores-and-Fixtures"),
    ("WC Qualifying UEFA", "2018", "/en/comps/68/2018/schedule/2018-FIFA-World-Cup-Qualifying-UEFA-Scores-and-Fixtures"),

    # ── World Cup Qualifiers: CONMEBOL ────────────────────────────────────────
    ("WC Qualifying CONMEBOL", "2022", "/en/comps/96/2022/schedule/2022-FIFA-World-Cup-Qualifying-CONMEBOL-Scores-and-Fixtures"),
    ("WC Qualifying CONMEBOL", "2018", "/en/comps/96/2018/schedule/2018-FIFA-World-Cup-Qualifying-CONMEBOL-Scores-and-Fixtures"),

    # ── World Cup Qualifiers: CONCACAF ────────────────────────────────────────
    ("WC Qualifying CONCACAF", "2022", "/en/comps/98/2022/schedule/2022-FIFA-World-Cup-Qualifying-CONCACAF-Scores-and-Fixtures"),

    # ── CONCACAF Gold Cup ─────────────────────────────────────────────────────
    ("CONCACAF Gold Cup", "2023", "/en/comps/679/2023/schedule/2023-CONCACAF-Gold-Cup-Scores-and-Fixtures"),
    ("CONCACAF Gold Cup", "2021", "/en/comps/679/2021/schedule/2021-CONCACAF-Gold-Cup-Scores-and-Fixtures"),
    ("CONCACAF Gold Cup", "2019", "/en/comps/679/2019/schedule/2019-CONCACAF-Gold-Cup-Scores-and-Fixtures"),

    # ── AFC Asian Cup ─────────────────────────────────────────────────────────
    ("AFC Asian Cup", "2023", "/en/comps/717/2023/schedule/2023-AFC-Asian-Cup-Scores-and-Fixtures"),
    ("AFC Asian Cup", "2019", "/en/comps/717/2019/schedule/2019-AFC-Asian-Cup-Scores-and-Fixtures"),
]


# ─── Request helper ───────────────────────────────────────────────────────────
_session = cffi_requests.Session(impersonate="chrome110")

def _get(url: str, max_retries: int = 3) -> BeautifulSoup | None:
    for attempt in range(max_retries):
        try:
            resp = _session.get(url, timeout=20, headers=HEADERS)
            if resp.status_code == 429:
                wait = 60 + random.uniform(0, 30)
                print(f"  ⚠ Rate-limited. Sleeping {wait:.0f}s...")
                time.sleep(wait)
                continue
            if resp.status_code == 200:
                return BeautifulSoup(resp.text, "lxml")
            print(f"  HTTP {resp.status_code} — {url}")
            return None
        except Exception as e:
            print(f"  Error (attempt {attempt+1}): {e}")
            time.sleep(10)
    return None

def _sleep():
    time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))


# ─── Fixtures page parser ──────────────────────────────────────────────────────
def get_match_urls(fixtures_url: str) -> list[tuple[str, str, str, str, str]]:
    """
    Parse the Scores & Fixtures page and return list of:
      (date, home_team, away_team, score_str, match_report_url)
    Only rows that have a Match Report link are returned.
    """
    soup = _get(BASE_URL + fixtures_url)
    if soup is None:
        return []

    table = soup.find("table", id=re.compile("sched"))
    if table is None:
        # Try without id filter
        tables = soup.find_all("table")
        table = tables[0] if tables else None
    if table is None:
        print(f"  No schedule table found at {fixtures_url}")
        return []

    results = []
    for row in table.find("tbody").find_all("tr"):
        if "thead" in row.get("class", []):
            continue
        cells = row.find_all(["td", "th"])
        if len(cells) < 5:
            continue

        # Find the Match Report link
        report_link = row.find("a", string=re.compile(r"Match Report", re.I))
        if not report_link:
            continue

        report_href = report_link.get("href", "")
        if not report_href:
            continue

        # Parse cells — FBref fixture tables have consistent order:
        # [Wk, Day, Date, Time, Home, Score, Away, ...]
        cell_texts = [c.get_text(strip=True) for c in cells]

        date_str = ""
        home = ""
        score = ""
        away = ""

        # Try to find date, home, score, away by cell position
        # FBref standard: col0=wk, col1=day, col2=date, col3=time, col4=home, col5=score, col6=away
        try:
            date_str = cell_texts[2]
            home     = cell_texts[4]
            score    = cell_texts[5]
            away     = cell_texts[6]
        except IndexError:
            continue

        if not date_str or not home or not away or "–" in score.replace("–","").strip() == "":
            pass

        results.append((date_str, home, away, score, BASE_URL + report_href))

    return results


# ─── Match report parser ───────────────────────────────────────────────────────
def parse_match_report(url: str) -> dict | None:
    """
    Parse an FBref match report page and extract box-score stats.
    Returns a dict of stats, or None if parsing failed.
    """
    soup = _get(url)
    if soup is None:
        return None

    stats = {}

    # ── Scorebox: home/away team names and score ───────────────────────────────
    scorebox = soup.find("div", class_="scorebox")
    if scorebox is None:
        return None

    team_divs = scorebox.find_all("div", recursive=False)
    teams = [d.find("a") for d in team_divs if d.find("a") and d.find("a").get("href","").startswith("/en/national/")]
    if len(teams) < 2:
        # fallback: any strong tag in scorebox
        strongs = scorebox.find_all("strong")
        team_names = [s.get_text(strip=True) for s in strongs if len(s.get_text(strip=True)) > 1]
        if len(team_names) >= 2:
            stats["home_team"] = team_names[0]
            stats["away_team"] = team_names[1]
    else:
        stats["home_team"] = teams[0].get_text(strip=True)
        stats["away_team"] = teams[1].get_text(strip=True)

    scores = scorebox.find_all("div", class_="score")
    if len(scores) >= 2:
        try:
            stats["home_score"] = int(scores[0].get_text(strip=True))
            stats["away_score"] = int(scores[1].get_text(strip=True))
        except ValueError:
            pass

    # ── Date ─────────────────────────────────────────────────────────────────
    date_span = scorebox.find("span", class_="venuetime") or scorebox.find("h2")
    meta_div = soup.find("div", class_=re.compile("scorebox_meta"))
    if meta_div:
        for strong in meta_div.find_all("strong"):
            txt = strong.get_text(strip=True)
            if re.match(r"\w+ \d+, \d{4}", txt):
                try:
                    stats["date"] = datetime.strptime(txt, "%B %d, %Y").strftime("%Y-%m-%d")
                except ValueError:
                    pass
                break

    # ── Team Stats table (possession, shots, corners, fouls, cards) ───────────
    # FBref puts a "Team Stats" section with a summary table
    team_stats_div = soup.find("div", id="team_stats")
    if team_stats_div:
        rows = team_stats_div.find_all("tr")
        for row in rows:
            tds = row.find_all("td")
            th = row.find("th")
            if not th or len(tds) < 2:
                continue
            label = th.get_text(strip=True).lower()
            home_val = tds[0].get_text(strip=True)
            away_val = tds[1].get_text(strip=True)

            def _num(s):
                s = re.sub(r"[^\d.]", "", s)
                try: return float(s)
                except: return None

            if "possession" in label:
                stats["home_poss"] = _num(home_val)
                stats["away_poss"] = _num(away_val)
            elif "shots on target" in label:
                stats["home_shots_on_target"] = _num(home_val)
                stats["away_shots_on_target"] = _num(away_val)
            elif "shots" in label:
                stats["home_shots"] = _num(home_val)
                stats["away_shots"] = _num(away_val)
            elif "corner" in label:
                stats["home_corners"] = _num(home_val)
                stats["away_corners"] = _num(away_val)
            elif "foul" in label:
                stats["home_fouls"] = _num(home_val)
                stats["away_fouls"] = _num(away_val)
            elif "yellow" in label:
                stats["home_yellows"] = _num(home_val)
                stats["away_yellows"] = _num(away_val)
            elif "red" in label:
                stats["home_reds"] = _num(home_val)
                stats["away_reds"] = _num(away_val)

    # Fallback: try the extra stats table that sometimes appears instead
    extra_div = soup.find("div", id="team_stats_extra")
    if extra_div:
        for row in extra_div.find_all("tr"):
            tds = row.find_all("td")
            if len(tds) < 3:
                continue
            label = tds[1].get_text(strip=True).lower()
            home_val = tds[0].get_text(strip=True)
            away_val = tds[2].get_text(strip=True)
            def _num(s):
                s = re.sub(r"[^\d.]", "", s)
                try: return float(s)
                except: return None

            if "corner" in label and "home_corners" not in stats:
                stats["home_corners"] = _num(home_val)
                stats["away_corners"] = _num(away_val)
            elif "foul" in label and "home_fouls" not in stats:
                stats["home_fouls"] = _num(home_val)
                stats["away_fouls"] = _num(away_val)
            elif "yellow" in label and "home_yellows" not in stats:
                stats["home_yellows"] = _num(home_val)
                stats["away_yellows"] = _num(away_val)
            elif "red" in label and "home_reds" not in stats:
                stats["home_reds"] = _num(home_val)
                stats["away_reds"] = _num(away_val)
            elif "possession" in label and "home_poss" not in stats:
                stats["home_poss"] = _num(home_val)
                stats["away_poss"] = _num(away_val)

    stats["match_url"] = url
    return stats


# ─── Main orchestrator ─────────────────────────────────────────────────────────
COLUMNS = [
    "date", "home_team", "away_team", "home_score", "away_score",
    "competition", "season",
    "home_poss", "away_poss",
    "home_shots", "away_shots",
    "home_shots_on_target", "away_shots_on_target",
    "home_corners", "away_corners",
    "home_fouls", "away_fouls",
    "home_yellows", "away_yellows",
    "home_reds", "away_reds",
    "match_url",
]

def load_existing() -> set:
    """Return set of already-scraped match URLs."""
    if OUTPUT_PATH.exists():
        df = pd.read_csv(OUTPUT_PATH)
        return set(df["match_url"].dropna())
    return set()

def scrape_all():
    print("=" * 65)
    print("FBref International Match Stats Scraper")
    print("=" * 65)
    print(f"Output: {OUTPUT_PATH}")
    print(f"Competitions: {len(COMPETITIONS)}")
    print(f"Rate limit: {MIN_DELAY}–{MAX_DELAY}s per request\n")

    already_scraped = load_existing()
    print(f"Resuming — {len(already_scraped)} matches already saved.\n")

    total_saved = len(already_scraped)

    for comp_name, season, fixture_url in COMPETITIONS:
        print(f"\n{'─'*60}")
        print(f"  {comp_name} {season}")
        print(f"{'─'*60}")

        _sleep()
        match_list = get_match_urls(fixture_url)
        print(f"  Found {len(match_list)} match report links")

        for date_str, home, away, score, report_url in match_list:
            if report_url in already_scraped:
                continue

            print(f"  Scraping: {date_str}  {home} vs {away}  ({score})  ...", end="", flush=True)
            _sleep()

            row = parse_match_report(report_url)
            if row is None:
                print(" FAILED")
                continue

            # Fill from fixtures page if report parsing missed these
            if "home_team" not in row or not row.get("home_team"):
                row["home_team"] = home
            if "away_team" not in row or not row.get("away_team"):
                row["away_team"] = away
            if "date" not in row or not row.get("date"):
                try:
                    row["date"] = pd.to_datetime(date_str).strftime("%Y-%m-%d")
                except Exception:
                    row["date"] = date_str

            # Parse score from fixtures page as fallback
            if "home_score" not in row:
                m = re.search(r"(\d+)[^\d]+(\d+)", score)
                if m:
                    row["home_score"] = int(m.group(1))
                    row["away_score"] = int(m.group(2))

            row["competition"] = comp_name
            row["season"] = season

            # Write one row at a time so we don't lose progress
            out_row = {c: row.get(c, None) for c in COLUMNS}
            df_row = pd.DataFrame([out_row])
            write_header = not OUTPUT_PATH.exists()
            df_row.to_csv(OUTPUT_PATH, mode="a", index=False, header=write_header)

            already_scraped.add(report_url)
            total_saved += 1
            print(f" ✓  [total={total_saved}]")

    print(f"\n{'='*65}")
    print(f"Done! {total_saved} total match records in {OUTPUT_PATH}")


if __name__ == "__main__":
    scrape_all()
