import pandas as pd
from src.pipelines.training_pipeline import load_data, load_elos, TacticalStatsComputer, build_training_dataset

df = load_data()
df_tactical = pd.read_csv("data/statsbomb_match_stats.csv")
tactical = TacticalStatsComputer(df_tactical)

print(tactical.get_features("Canada", pd.to_datetime('2022-12-01'), "_a"))
