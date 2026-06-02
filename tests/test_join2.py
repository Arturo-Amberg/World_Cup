import pandas as pd
import json
from src.pipelines.training_pipeline import load_data, load_elos, TacticalStatsComputer, RollingStatsComputer, H2HComputer, build_training_dataset

df = load_data()
df_tactical = pd.read_csv("data/statsbomb_match_stats.csv")

df_tactical['date'] = pd.to_datetime(df_tactical['date']).dt.strftime('%Y-%m-%d')
df['date'] = df['date'].dt.strftime('%Y-%m-%d')

df_tactical_indexed = df_tactical.set_index(['date', 'home_team'])
df_targets_indexed = df.set_index(['date', 'home_team'])

joined = df_targets_indexed.join(df_tactical_indexed[['home_corners']], how='inner')
print("Join length:", len(joined))

