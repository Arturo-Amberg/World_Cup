import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import os

LEAGUES = [
    "fifa.friendlies",
    "uefa.nations",
    "uefa.nations_league",
    "concacaf.gold",
    "afc.asian.cup",
    "caf.nations",
    "conmebol.america",
    "uefa.euro",
    "fifa.worldq.conmebol",
    "fifa.worldq.uefa",
    "fifa.worldq.concacaf",
    "fifa.worldq.afc",
    "fifa.worldq.caf"
]

def fetch_espn_stats(year: int):
    results = []
    
    # ESPN allows fetching a whole year using dates=YYYY0101-YYYY1231
    start_date = f"{year}0101"
    end_date = f"{year}1231"
    
    for league in LEAGUES:
        url = f"http://site.api.espn.com/apis/site/v2/sports/soccer/{league}/scoreboard?dates={start_date}-{end_date}&limit=1000"
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code != 200:
                continue
            
            data = resp.json()
            events = data.get("events", [])
            
            for event in events:
                try:
                    date_str = event["date"]
                    match_date = datetime.strptime(date_str, "%Y-%m-%dT%H:%MZ").strftime("%Y-%m-%d")
                    comp = event["competitions"][0]
                    
                    home_team = None
                    away_team = None
                    home_stats = {}
                    away_stats = {}
                    
                    for competitor in comp["competitors"]:
                        team_name = competitor["team"]["name"]
                        is_home = competitor["homeAway"] == "home"
                        
                        stats = {}
                        if "statistics" in competitor:
                            for stat in competitor["statistics"]:
                                stats[stat["name"]] = stat.get("displayValue", "0")
                        
                        if is_home:
                            home_team = team_name
                            home_stats = stats
                        else:
                            away_team = team_name
                            away_stats = stats
                    
                    if home_team and away_team and "possessionPct" in home_stats:
                        results.append({
                            "date": match_date,
                            "home_team": home_team,
                            "away_team": away_team,
                            "home_possession_pct": float(home_stats.get("possessionPct", 50)),
                            "home_shots": float(home_stats.get("totalShots", 0)),
                            "home_corners": float(home_stats.get("wonCorners", 0)),
                            "away_possession_pct": float(away_stats.get("possessionPct", 50)),
                            "away_shots": float(away_stats.get("totalShots", 0)),
                            "away_corners": float(away_stats.get("wonCorners", 0))
                        })
                except Exception as e:
                    continue
        except Exception as e:
            continue
            
    return results

if __name__ == "__main__":
    print("Fetching historical ESPN tactical data...")
    all_data = []
    
    # Fetch last 4 years of data (2021-2024)
    for y in range(2021, 2025):
        print(f"Fetching {y}...")
        all_data.extend(fetch_espn_stats(y))
        
        
    df = pd.DataFrame(all_data)
    print(f"Fetched {len(df)} tactical records from ESPN.")
    
    existing_file = "data/statsbomb_match_stats.csv"
    if os.path.exists(existing_file):
        existing_df = pd.read_csv(existing_file)
        combined = pd.concat([existing_df, df]).drop_duplicates(subset=["date", "home_team", "away_team"])
    else:
        combined = df
        
    combined.to_csv("data/statsbomb_match_stats.csv", index=False)
    print(f"Saved {len(combined)} total tactical records.")
