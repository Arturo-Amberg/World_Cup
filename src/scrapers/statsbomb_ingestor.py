import requests
import json
import pandas as pd
import time
import os

TOURNAMENTS = [
    (43, 106),   # WC 2022
    (43, 3),     # WC 2018
    (55, 282),   # Euro 2024
    (55, 43),    # Euro 2020
    (223, 282),  # Copa America 2024
    (1267, 107)  # AFCON 2023
]

# Team name mapping to align with intl_results.csv
TEAM_MAP = {
    "USA": "United States",
    "United States of America": "United States",
    "Korea Republic": "South Korea",
    "Republic of Ireland": "Ireland",
    "IR Iran": "Iran",
    "Türkiye": "Turkey",
    "Czech Republic": "Czechia",
    "Bosnia and Herzegovina": "Bosnia-Herzegovina"
}

def map_team(name):
    return TEAM_MAP.get(name, name)

def main():
    print("="*50)
    print("STATSBOMB TACTICAL DATA INGESTOR")
    print("="*50)

    records = []
    
    for comp_id, season_id in TOURNAMENTS:
        print(f"\nFetching matches for Comp: {comp_id}, Season: {season_id}...")
        url_matches = f"https://raw.githubusercontent.com/statsbomb/open-data/master/data/matches/{comp_id}/{season_id}.json"
        
        try:
            resp = requests.get(url_matches)
            if resp.status_code != 200:
                print(f"  Failed to fetch matches. HTTP {resp.status_code}")
                continue
            matches = resp.json()
        except Exception as e:
            print(f"  Error fetching matches: {e}")
            continue
            
        print(f"  Found {len(matches)} matches. Downloading events...")
        
        for i, match in enumerate(matches):
            match_id = match['match_id']
            home_team = map_team(match['home_team']['home_team_name'])
            away_team = map_team(match['away_team']['away_team_name'])
            match_date = match['match_date']
            
            # Print progress every 10 matches
            if i % 10 == 0 and i > 0:
                print(f"    Downloaded {i}/{len(matches)} events...")
                
            events_url = f"https://raw.githubusercontent.com/statsbomb/open-data/master/data/events/{match_id}.json"
            try:
                resp_events = requests.get(events_url)
                if resp_events.status_code != 200:
                    continue
                events = resp_events.json()
            except Exception:
                continue
                
            home_shots = 0
            home_corners = 0
            home_passes = 0
            away_shots = 0
            away_corners = 0
            away_passes = 0
            
            for e in events:
                team = map_team(e.get('team', {}).get('name'))
                e_type = e.get('type', {}).get('name')
                
                is_home = (team == home_team)
                
                if e_type == 'Shot':
                    if is_home: home_shots += 1
                    else: away_shots += 1
                elif e_type == 'Pass':
                    # A completed pass in StatsBomb lacks an 'outcome' field
                    outcome = e.get('pass', {}).get('outcome')
                    if outcome is None:
                        if is_home: home_passes += 1
                        else: away_passes += 1
                        
                    p_type = e.get('pass', {}).get('type', {}).get('name')
                    if p_type == 'Corner':
                        if is_home: home_corners += 1
                        else: away_corners += 1
            
            total_passes = home_passes + away_passes
            if total_passes > 0:
                home_possession = round((home_passes / total_passes) * 100, 1)
                away_possession = round((away_passes / total_passes) * 100, 1)
            else:
                home_possession = 50.0
                away_possession = 50.0
                        
            records.append({
                'date': match_date,
                'home_team': home_team,
                'away_team': away_team,
                'home_corners': home_corners,
                'away_corners': away_corners,
                'home_shots': home_shots,
                'away_shots': away_shots,
                'home_possession': home_possession,
                'away_possession': away_possession
            })
            
            # Brief sleep to avoid hammering github too fast
            time.sleep(0.05)
            
    df = pd.DataFrame(records)
    print(f"\nSuccessfully parsed {len(df)} matches with tactical data!")
    
    out_path = "data/statsbomb_match_stats.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    main()
