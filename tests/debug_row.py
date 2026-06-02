import pandas as pd
import json
from src.analysis.coolbet_evaluator import map_team
from src.pipelines.training_pipeline import load_data, load_elos, RollingStatsComputer, H2HComputer, build_training_dataset

df = load_data()
elos = load_elos()
rolling = RollingStatsComputer(df)
h2h = H2HComputer(df)

coolbet = pd.read_csv("data/coolbet/latest.csv")
coolbet['home_mapped'] = coolbet['home'].apply(map_team)
coolbet['away_mapped'] = coolbet['away'].apply(map_team)
upcoming = coolbet[['home_mapped', 'away_mapped', 'match_start']].drop_duplicates().copy()
upcoming['date'] = pd.to_datetime(upcoming['match_start']).dt.tz_localize(None)

df_upcoming = pd.DataFrame({
    'date': upcoming['date'],
    'home_team': upcoming['home_mapped'],
    'away_team': upcoming['away_mapped'],
    'home_score': 0, 'away_score': 0, 'tournament': 'FIFA World Cup',
    'city': 'Unknown', 'country': 'United States', 'neutral': True
}).reset_index(drop=True)

idx = df_upcoming[(df_upcoming['home_team'] == 'Spain') & (df_upcoming['away_team'] == 'Cape Verde')].index[0]
print(df_upcoming.iloc[idx])
