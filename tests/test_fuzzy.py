import pandas as pd
import difflib
from src.pipelines.training_pipeline import load_data

df = load_data()
valid_teams = pd.concat([df['home_team'], df['away_team']]).unique()

coolbet = pd.read_csv('data/coolbet/latest.csv')
coolbet_teams = pd.concat([coolbet['home'], coolbet['away']]).unique()

for t in coolbet_teams:
    match = difflib.get_close_matches(t, valid_teams, n=1, cutoff=0.6)
    if match:
        print(f"{t} -> {match[0]}")
    else:
        print(f"{t} -> NO MATCH")
