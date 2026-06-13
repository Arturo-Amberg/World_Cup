"""
Coolbet odds scraper for World Cup 2026.

Coolbet uses GeeTest CAPTCHA / Incapsula anti-bot protection on its API.
The only reliable method is to piggyback on a real Chrome session where the
user has already passed the bot challenge.

HOW TO USE
----------
1. Open Google Chrome with remote-debugging enabled:

     /Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome \
         --remote-debugging-port=9222

2. In that Chrome window, navigate to:
     https://www.coolbet.com/en/sports/football/world-cup/matches
   and wait for the page to fully load.

3. Then run:
     python3 -m src.scrapers.coolbet_scraper

The scraper connects to Chrome via CDP, extracts the Incapsula session
cookies, and uses them to call the JSON APIs directly.

Pages scraped:
  - matches              (category 46803)  — regular match odds
  - nation_specials      (category 61857)  — per-country outright specials
  - group_specials       (category 61856)  — group stage outright specials
  - wc_specials          (category 51930)  — tournament-level outright specials

Usage:
    python3 -m src.scrapers.coolbet_scraper        # scrape and save
    python3 -m src.scrapers.coolbet_scraper --json # print JSON to stdout
"""

import asyncio
import json
import csv
import sys
import time
from pathlib import Path
from datetime import datetime

import requests
from playwright.async_api import async_playwright

# ── config ────────────────────────────────────────────────────────────────────
CDP_URL = "http://localhost:9222"

CATEGORIES = {
    "matches":         {"id": 46803, "type": "matches"},
    "match_sidebets":  {"id": 46803, "type": "match_sidebets"},
    "nation_specials": {"id": 61857, "type": "outright"},
    "group_specials":  {"id": 61856, "type": "outright"},
    "wc_specials":     {"id": 51930, "type": "outright"},
}

BASE     = "https://www.coolbet.com"
SBGATE   = f"{BASE}/s/sbgate/sports/fo-category/"
SIDEBETS = f"{BASE}/s/sbgate/sports/fo-market/sidebets"
PRICES   = f"{BASE}/s/sb-odds/odds/current/fo-line/"
PRICES2  = f"{BASE}/s/sb-odds/odds/current/fo"

OUTPUT_DIR = Path("data/coolbet")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent":      "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept":          "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Origin":          BASE,
    "Referer":         f"{BASE}/en/sports/football/world-cup/matches",
    "Sec-Fetch-Site":  "same-origin",
    "Sec-Fetch-Mode":  "cors",
    "Sec-Fetch-Dest":  "empty",
    "X-Device":        "DESKTOP",
}


# ── Step 1: get Incapsula cookies from running Chrome ─────────────────────────

async def _get_chrome_cookies() -> dict:
    async with async_playwright() as p:
        browser = await p.chromium.connect_over_cdp(CDP_URL)
        ctx     = browser.contexts[0]
        cookies = await ctx.cookies(BASE)
        return {c["name"]: c["value"] for c in cookies}


def get_session() -> requests.Session:
    """Connect to Chrome via CDP, extract cookies, return authenticated session."""
    print("Connecting to Chrome (CDP)...")
    try:
        cookies = asyncio.run(_get_chrome_cookies())
    except Exception as e:
        print(
            f"\n  ✗ Could not connect to Chrome: {e}\n\n"
            "  To scrape Coolbet you need a real Chrome session:\n\n"
            "    1. Open Chrome with:\n"
            "       /Applications/Google\\ Chrome.app/Contents/MacOS/Google\\ Chrome"
            " --remote-debugging-port=9222\n\n"
            "    2. Navigate to:\n"
            "       https://www.coolbet.com/en/sports/football/world-cup/matches\n"
            "       and wait for the page to load fully.\n\n"
            "    3. Then re-run this scraper.\n"
        )
        sys.exit(1)

    key_cookies = [k for k in ["reese84", "visid_incap_723517"] if k in cookies]
    print(f"  ✓ Got {len(cookies)} cookies  ({', '.join(key_cookies) or 'WARNING: no Incapsula cookies!'})")

    s = requests.Session()
    for k, v in cookies.items():
        s.cookies.set(k, v, domain=".coolbet.com")
    s.headers.update(HEADERS)
    return s


# ── Step 2: fetch data via the JSON APIs ──────────────────────────────────────

def fetch_category(session: requests.Session, cat_id: int, limit: int = 500) -> list[dict]:
    r = session.get(SBGATE, params={
        "categoryId": cat_id, "country": "CL", "isMobile": "0",
        "language": "en", "layout": "EUROPEAN", "limit": limit,
    }, timeout=20)
    r.raise_for_status()
    data = r.json()
    if isinstance(data, list) and data:
        return data[0].get("matches", [])
    if isinstance(data, dict):
        cats = data.get("categories", [])
        if cats:
            return cats[0].get("matches", [])
    return []


def collect_match_market_ids(matches: list[dict]) -> tuple[list[list[int]], dict[int, dict]]:
    outcome_index: dict[int, dict] = {}
    for match in matches:
        for market in match.get("markets", []):
            for outcome in market.get("outcomes", []):
                outcome_index[outcome["id"]] = {"outcome": outcome, "market": market, "match": match}

    market_id_groups: list[list[int]] = []
    for match in matches:
        ids = [m["id"] for m in match.get("markets", [])]
        if not ids:
            continue
        market_id_groups.append([ids[0]])
        if len(ids) > 1:
            market_id_groups.append(ids[1:])

    return market_id_groups, outcome_index


def fetch_match_prices(session: requests.Session, market_id_groups: list[list[int]]) -> dict[int, float]:
    if not market_id_groups:
        return {}
    r = session.post(PRICES, json={"marketIds": market_id_groups}, timeout=30)
    r.raise_for_status()
    return {
        int(k): v.get("value")
        for k, v in r.json().items()
        if isinstance(v, dict) and v.get("status") == "OPEN"
    }


def fetch_sidebets(session: requests.Session, match_id: int) -> dict:
    r = session.get(SIDEBETS, params={
        "country": "CL", "language": "en", "layout": "EUROPEAN",
        "matchId": match_id, "matchStatus": "OPEN",
    }, timeout=15)
    r.raise_for_status()
    return r.json()


def collect_outright_market_ids(sidebets: dict, match: dict) -> tuple[list[int], dict[int, dict]]:
    market_ids:    list[int]      = []
    outcome_index: dict[int, dict] = {}
    for group in sidebets.get("markets", []):
        for mkt in group.get("markets", []):
            mid = mkt["id"]
            if mid not in market_ids:
                market_ids.append(mid)
            for outcome in mkt.get("outcomes", []):
                outcome_index[outcome["id"]] = {
                    "outcome":     outcome,
                    "market":      mkt,
                    "match":       match,
                    "market_name": mkt.get("name") or group.get("market_type_name", ""),
                    "group_name":  group.get("market_type_name", ""),
                }
    return market_ids, outcome_index


def fetch_outright_prices(session: requests.Session, market_ids: list[int]) -> dict[int, float]:
    if not market_ids:
        return {}
    r = session.post(PRICES2, json={"where": {"market_id": {"in": market_ids}}}, timeout=30)
    r.raise_for_status()
    return {
        int(k): v.get("value")
        for k, v in r.json().items()
        if isinstance(v, dict) and v.get("status") == "OPEN"
    }


def build_rows(page_key: str, outcome_index: dict[int, dict], prices: dict[int, float]) -> list[dict]:
    rows = []
    for oid, ctx in outcome_index.items():
        odds = prices.get(oid)
        if odds is None:
            continue
        match  = ctx["match"]
        market = ctx["market"]
        rows.append({
            "page":        page_key,
            "match":       match.get("name", ""),
            "home":        match.get("home_team_name", ""),
            "away":        match.get("away_team_name", ""),
            "match_start": match.get("match_start", ""),
            "market":      ctx.get("market_name") or market.get("name", ""),
            "market_type": ctx.get("group_name") or market.get("market_type_name") or market.get("name", ""),
            "line":        market.get("line", ""),
            "selection":   ctx["outcome"].get("name", ""),
            "odds":        odds,
        })
    return rows


# ── Step 3: orchestrate ───────────────────────────────────────────────────────

def scrape_matches(session: requests.Session, cat_id: int, page_key: str) -> list[dict]:
    matches = fetch_category(session, cat_id)
    print(f"  {len(matches)} matches")
    if not matches:
        return []
    groups, outcome_index = collect_match_market_ids(matches)
    print(f"  {sum(len(g) for g in groups)} markets, {len(outcome_index)} outcomes — fetching prices...")
    prices = fetch_match_prices(session, groups)
    print(f"  {len(prices)} priced outcomes")
    return build_rows(page_key, outcome_index, prices)


def scrape_match_sidebets(session: requests.Session, cat_id: int, page_key: str) -> list[dict]:
    """Fetch the full sidebets (corners, cards, shots, halves, player props) for every match."""
    matches = fetch_category(session, cat_id)
    print(f"  {len(matches)} matches")
    if not matches:
        return []
    all_rows: list[dict] = []
    for i, match in enumerate(matches):
        match_id   = match["id"]
        match_name = match.get("name", f"id={match_id}")
        try:
            sidebets            = fetch_sidebets(session, match_id)
            market_ids, out_idx = collect_outright_market_ids(sidebets, match)
            if not market_ids:
                continue
            prices = fetch_outright_prices(session, market_ids)
            rows   = build_rows(page_key, out_idx, prices)
            all_rows.extend(rows)
            print(f"  [{i+1}/{len(matches)}] {match_name}: {len(rows)} odds")
            time.sleep(0.15)
        except Exception as e:
            print(f"  [{i+1}/{len(matches)}] {match_name}: ERROR {e}")
    return all_rows


def scrape_outrights(session: requests.Session, cat_id: int, page_key: str) -> list[dict]:
    matches = fetch_category(session, cat_id)
    print(f"  {len(matches)} outright events")
    if not matches:
        return []
    all_rows: list[dict] = []
    for i, match in enumerate(matches):
        match_id   = match["id"]
        match_name = match.get("name", f"id={match_id}")
        try:
            sidebets            = fetch_sidebets(session, match_id)
            market_ids, out_idx = collect_outright_market_ids(sidebets, match)
            if not market_ids:
                continue
            prices = fetch_outright_prices(session, market_ids)
            rows   = build_rows(page_key, out_idx, prices)
            all_rows.extend(rows)
            print(f"  [{i+1}/{len(matches)}] {match_name}: {len(rows)} odds")
            time.sleep(0.15)
        except Exception as e:
            print(f"  [{i+1}/{len(matches)}] {match_name}: ERROR {e}")
    return all_rows


def scrape() -> list[dict]:
    session  = get_session()
    all_rows: list[dict] = []

    for page_key, cat in CATEGORIES.items():
        cat_id   = cat["id"]
        cat_type = cat["type"]
        print(f"\n[{page_key}] Fetching category {cat_id} ({cat_type})...")
        try:
            if cat_type == "matches":
                rows = scrape_matches(session, cat_id, page_key)
            elif cat_type == "match_sidebets":
                rows = scrape_match_sidebets(session, cat_id, page_key)
            else:
                rows = scrape_outrights(session, cat_id, page_key)
            print(f"  → {len(rows)} rows extracted")
            all_rows.extend(rows)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()

    return all_rows


# ── Step 4: save ──────────────────────────────────────────────────────────────

def save(rows: list[dict], timestamp: str) -> None:
    if not rows:
        print("\nNo data to save.")
        return

    json_path   = OUTPUT_DIR / f"coolbet_odds_{timestamp}.json"
    csv_path    = OUTPUT_DIR / f"coolbet_odds_{timestamp}.csv"
    latest_json = OUTPUT_DIR / "latest.json"
    latest_csv  = OUTPUT_DIR / "latest.csv"

    for p in (json_path, latest_json):
        with open(p, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)

    fieldnames = list(rows[0].keys())
    for p in (csv_path, latest_csv):
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)

    print(f"\nSaved {len(rows)} rows:")
    print(f"  JSON    → {json_path}")
    print(f"  CSV     → {csv_path}")
    print(f"  Latest  → {latest_json}")

    print("\nPer-page breakdown:")
    from collections import Counter
    for key, n in Counter(r["page"] for r in rows).items():
        print(f"  {key}: {n} odds")


def main():
    print_json = "--json" in sys.argv
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    rows       = scrape()

    if print_json:
        print(json.dumps(rows, indent=2))
    else:
        save(rows, timestamp)
        print("\n--- Sample (first 10 rows) ---")
        for r in rows[:10]:
            line = f" [{r['line']}]" if r["line"] else ""
            print(f"  [{r['page']}] {r['match']} | {r['market']}{line} | {r['selection']}: {r['odds']}")

    return rows


if __name__ == "__main__":
    main()
