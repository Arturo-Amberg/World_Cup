import pandas as pd
from src.pipelines.training_pipeline import load_data, load_elos, TacticalStatsComputer, RollingStatsComputer, H2HComputer, build_training_dataset

df = load_data()
df_tactical = pd.read_csv("data/statsbomb_match_stats.csv")

elos = load_elos()
tactical = TacticalStatsComputer(df_tactical)
rolling = RollingStatsComputer(df)
h2h = H2HComputer(df)

candidates = df[df["date"].dt.year >= 2005].copy()
valid_idx = []
rows = []
for i, (idx, row) in enumerate(candidates.iterrows()):
    home = str(row["home_team"])
    date = row["date"]
    if rolling.n_prior(home, date) < 10:
        continue
    valid_idx.append(idx)
    rows.append({"a": 1})

print("Length of rows:", len(rows))
print("Max valid index:", max(valid_idx))
