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

df_tactical_indexed = df_tactical.set_index(['date', 'home_team'])
df_targets_indexed = df_targets.reset_index().set_index(['date', 'home_team'])

joined = df_targets_indexed.join(df_tactical_indexed[['home_corners']], how='inner')
print("Join length on df_targets:", len(joined))

