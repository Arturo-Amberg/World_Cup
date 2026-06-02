import requests
import json

comps = requests.get("https://raw.githubusercontent.com/statsbomb/open-data/master/data/competitions.json").json()
for c in comps:
    if c.get("competition_international"):
        print(f"{c['competition_id']} - {c['competition_name']} - {c['season_name']} ({c['season_id']})")
