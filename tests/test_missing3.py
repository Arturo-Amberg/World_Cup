import pandas as pd
from src.pipelines.training_pipeline import load_data, load_elos, TacticalStatsComputer, RollingStatsComputer, H2HComputer, build_training_dataset
import time

df = load_data()
df_tactical = pd.read_csv("data/statsbomb_match_stats.csv")

elos = load_elos()
tactical = TacticalStatsComputer(df_tactical)
rolling = RollingStatsComputer(df)
h2h = H2HComputer(df)

candidates = df[df["date"].dt.year >= 2005].copy()
rows = []
for i, (_, row) in enumerate(candidates.iterrows()):
    home = str(row["home_team"])
    if home == "Canada" and row["date"] == pd.to_datetime("2022-12-01"):
        date = row["date"]
        away = str(row["away_team"])
        
        feat = {}
        feat.update(rolling.get_features(home, date, "_a"))
        feat.update(rolling.get_features(away, date, "_b"))
        feat.update(h2h.get_h2h(home, away, as_of_date=date))
        feat.update(tactical.get_features(home, date, "_a"))
        feat.update(tactical.get_features(away, date, "_b"))
        
        nans = [k for k, v in feat.items() if pd.isna(v)]
        print(f"Date {date} missing features: {nans}")
        break
