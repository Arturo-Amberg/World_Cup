import pandas as pd
from src.pipelines.training_pipeline import load_data, load_elos, TacticalStatsComputer, RollingStatsComputer, H2HComputer, build_training_dataset

df = load_data()
df_tactical = pd.read_csv("data/statsbomb_match_stats.csv")

elos = load_elos()
tactical = TacticalStatsComputer(df_tactical)
rolling = RollingStatsComputer(df)
h2h = H2HComputer(df)

X_df, _, _ = build_training_dataset(df, elos, rolling, h2h, tactical=tactical)

df_targets = df.loc[X_df.index].copy()

df_tactical['date'] = df_tactical['date'].astype(str)
df_targets['date'] = df_targets['date'].dt.strftime('%Y-%m-%d')

print("Is Canada vs Morocco in df_targets?")
print(df_targets[(df_targets['home_team']=='Canada') & (df_targets['date']=='2022-12-01')][['date', 'home_team']])

print("Is Canada vs Morocco in df_tactical?")
print(df_tactical[(df_tactical['home_team']=='Canada') & (df_tactical['date']=='2022-12-01')][['date', 'home_team']])

