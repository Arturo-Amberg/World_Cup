import pickle
import json
import pandas as pd
import numpy as np

with open("data/models/feature_names.json", "r") as f:
    features = json.load(f)

# Load the single row for Spain vs Cape Verde
df_ev = pd.read_csv("data/coolbet/model_value_bets.csv")

# Wait, X_features is what we want.
from src.analysis.coolbet_evaluator import map_team, TEAM_MAP
coolbet = pd.read_csv("data/coolbet/latest.csv")
coolbet['home_mapped'] = coolbet['home'].apply(map_team)
coolbet['away_mapped'] = coolbet['away'].apply(map_team)
upcoming = coolbet[['home_mapped', 'away_mapped', 'match_start']].drop_duplicates().copy()
upcoming['date'] = pd.to_datetime(upcoming['match_start']).dt.tz_localize(None)

from src.pipelines.training_pipeline import load_data, load_elos, RollingStatsComputer, H2HComputer, build_training_dataset
df = load_data()
elos = load_elos()
rolling = RollingStatsComputer(df)
h2h = H2HComputer(df)

df_upcoming = pd.DataFrame({
    'date': upcoming['date'],
    'home_team': upcoming['home_mapped'],
    'away_team': upcoming['away_mapped'],
    'home_score': 0, 'away_score': 0, 'tournament': 'FIFA World Cup',
    'city': 'Unknown', 'country': 'United States', 'neutral': True
}).reset_index(drop=True)

X_upcoming, _, _ = build_training_dataset(df_upcoming, elos, rolling, h2h, tactical=None)

for col in features:
    if col not in X_upcoming:
        X_upcoming[col] = 0.0
X_features = X_upcoming[features].copy()

# Find Spain vs Cape Verde row
idx = df_upcoming[(df_upcoming['home_team'] == 'Spain') & (df_upcoming['away_team'] == 'Cape Verde')].index[0]
x_row_features = X_features.iloc[[idx]]

with open("data/models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
x_row_scaled = scaler.transform(x_row_features)

print("--- Base Model Predictions (Spain vs Cape Verde) ---")
with open("data/models/training_report.json", "r") as f:
    report = json.load(f)
base_names = report["base_model_names"]

base_preds = []
for m in base_names:
    with open(f"data/models/base_{m}.pkl", "rb") as f:
        model = pickle.load(f)
    if m in {"poisson", "rolling_mean"}:
        p = np.full((1, 3), 1.0 / 3.0)
    elif hasattr(model, "predict_proba"):
        p = model.predict_proba(x_row_scaled)
    else:
        p = np.zeros((1, 3))
    print(f"{m:12s} | Home: {p[0,0]:.3f} | Draw: {p[0,1]:.3f} | Away: {p[0,2]:.3f}")
    base_preds.append(p)

X_meta = np.hstack(base_preds)
with open("data/models/meta_scaler.pkl", "rb") as f:
    meta_scaler = pickle.load(f)
X_meta_scaled = meta_scaler.transform(X_meta)

with open("data/models/meta_lgbm.pkl", "rb") as f:
    meta_lgbm = pickle.load(f)
meta_p = meta_lgbm.predict_proba(X_meta_scaled)
print(f"META_LGBM    | Home: {meta_p[0,0]:.3f} | Draw: {meta_p[0,1]:.3f} | Away: {meta_p[0,2]:.3f}")

