import pandas as pd
from src.pipelines.training_pipeline import load_data, load_elos, RollingStatsComputer

df = load_data()
rolling = RollingStatsComputer(df)
date = pd.to_datetime('2022-12-01')

print("Canada n_prior:", rolling.n_prior("Canada", date))
print("Morocco n_prior:", rolling.n_prior("Morocco", date))

