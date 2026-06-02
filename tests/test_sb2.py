import requests
import json

match_id = 3869685 # Example World Cup 2022 match (Argentina vs France maybe, just an ID from WC)
matches = requests.get("https://raw.githubusercontent.com/statsbomb/open-data/master/data/matches/43/106.json").json()
match_id = matches[0]['match_id']

events = requests.get(f"https://raw.githubusercontent.com/statsbomb/open-data/master/data/events/{match_id}.json").json()
shots = 0
corners = 0
for e in events:
    t = e.get('type', {}).get('name')
    if t == 'Shot':
        shots += 1
    elif t == 'Pass':
        pt = e.get('pass', {}).get('type', {}).get('name')
        if pt == 'Corner':
            corners += 1

print(f"Match: {matches[0]['home_team']['home_team_name']} vs {matches[0]['away_team']['away_team_name']}")
print(f"Total Shots: {shots}, Total Corners: {corners}")
