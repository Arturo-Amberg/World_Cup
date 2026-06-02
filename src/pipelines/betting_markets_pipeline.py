import pandas as pd
import numpy as np
import time
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, accuracy_score
import xgboost as xgb
import lightgbm as lgb

# Import the powerful feature engine and loading functions from our main pipeline
from src.pipelines.training_pipeline import (
    load_data, load_elos, RollingStatsComputer, H2HComputer, 
    _sample_weight, coords_map, haversine, ROLLING_WINDOWS, EWMA_ALPHA, MIN_YEAR, MIN_PRIOR_MATCHES
)

MODELS_DIR = Path("data/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def build_betting_dataset(df: pd.DataFrame, elos: dict, rolling: RollingStatsComputer, h2h: H2HComputer):
    t0 = time.time()
    rows = []
    y_ou25 = []  # Over/Under 2.5 goals (1 = Over, 0 = Under)
    y_btts = []  # Both Teams To Score (1 = Yes, 0 = No)
    weights = []

    print(f"Building betting features for candidates...")
    candidates = df[df["date"].dt.year >= MIN_YEAR].copy()
    
    for i, (_, row) in enumerate(candidates.iterrows()):
        if i % 10000 == 0 and i > 0:
            elapsed = time.time() - t0
            rem = (len(candidates) - i) * (elapsed / i)
            print(f"  {i:,}/{len(candidates):,}  ({elapsed:.0f}s elapsed, ~{rem:.0f}s remaining)")

        home, away = row["home_team"], row["away_team"]
        date = row["date"]

        if rolling.n_prior(home, date) < MIN_PRIOR_MATCHES or rolling.n_prior(away, date) < MIN_PRIOR_MATCHES:
            continue

        feat_a = rolling.get_features(home, date, suffix="_a")
        feat_b = rolling.get_features(away, date, suffix="_b")
        feat_h2h = h2h.get_h2h(home, away, as_of_date=date, n=5)

        feat = {}
        feat.update(feat_a)
        feat.update(feat_b)
        feat.update(feat_h2h)

        # Elo
        feat["elo_a"] = elos.get(home, 1500.0)
        feat["elo_b"] = elos.get(away, 1500.0)
        feat["elo_diff"] = feat["elo_a"] - feat["elo_b"]

        # Location/Tournament
        neutral = bool(row["neutral"])
        feat["is_neutral"] = 1.0 if neutral else 0.0
        feat["is_home_a"]  = 0.0 if neutral else 1.0
        feat["is_home_b"]  = 0.0
        feat["match_month"] = float(date.month)

        # Rest days
        rd_a = rolling.get_rest_days(home, date)
        rd_b = rolling.get_rest_days(away, date)
        feat["rest_days_a"] = rd_a
        feat["rest_days_b"] = rd_b
        feat["rest_days_diff"] = rd_a - rd_b

        # Distance
        dist_a = 0.0
        dist_b = 0.0
        match_country = row.get("country", home)
        if match_country in coords_map:
            c_lat, c_lon = coords_map[match_country]
            if home in coords_map:
                dist_a = haversine(c_lat, c_lon, coords_map[home][0], coords_map[home][1])
            if away in coords_map:
                dist_b = haversine(c_lat, c_lon, coords_map[away][0], coords_map[away][1])
        feat["travel_dist_a"] = dist_a
        feat["travel_dist_b"] = dist_b

        for w in ROLLING_WINDOWS:
            feat[f"gf_diff_{w}"]  = feat[f"gf_avg_{w}_a"]  - feat[f"gf_avg_{w}_b"]
            feat[f"ga_diff_{w}"]  = feat[f"ga_avg_{w}_a"]  - feat[f"ga_avg_{w}_b"]
            feat[f"pts_diff_{w}"] = feat[f"pts_avg_{w}_a"] - feat[f"pts_avg_{w}_b"]
            feat[f"win_diff_{w}"] = feat[f"win_rate_{w}_a"]- feat[f"win_rate_{w}_b"]
        feat["gf_ewma_diff"] = feat["gf_ewma_a"] - feat["gf_ewma_b"]
        feat["ga_ewma_diff"] = feat["ga_ewma_a"] - feat["ga_ewma_b"]

        feat["momentum_pts_a"] = feat["pts_avg_5_a"] - feat["pts_avg_30_a"]
        feat["momentum_pts_b"] = feat["pts_avg_5_b"] - feat["pts_avg_30_b"]
        feat["momentum_gd_a"] = (feat["gf_avg_5_a"] - feat["ga_avg_5_a"]) - (feat["gf_avg_30_a"] - feat["ga_avg_30_a"])
        feat["momentum_gd_b"] = (feat["gf_avg_5_b"] - feat["ga_avg_5_b"]) - (feat["gf_avg_30_b"] - feat["ga_avg_30_b"])
        feat["win_streak_diff"] = feat["win_streak_a"] - feat["win_streak_b"]
        feat["unbeaten_streak_diff"] = feat["unbeaten_streak_a"] - feat["unbeaten_streak_b"]

        rows.append(feat)

        hs, as_ = int(row["home_score"]), int(row["away_score"])
        total_goals = hs + as_
        
        # BETTING TARGETS
        y_ou25.append(1 if total_goals > 2.5 else 0)
        y_btts.append(1 if (hs > 0 and as_ > 0) else 0)

        year  = date.year
        tourn = str(row.get("tournament", "Friendly"))
        weights.append(_sample_weight(year, tourn))

    print(f"  Done: {len(rows):,} training rows in {time.time() - t0:.0f}s")
    X_df = pd.DataFrame(rows)
    
    y_ou25 = np.array(y_ou25, dtype=int)
    y_btts = np.array(y_btts, dtype=int)
    
    w = np.array(weights, dtype=float)
    w /= w.mean()
    return X_df, y_ou25, y_btts, w

def train_betting_markets():
    print("=" * 70)
    print("BETTING MARKETS PIPELINE — Over/Under & BTTS")
    print("=" * 70)

    df = load_data()
    elos = load_elos()
    rolling = RollingStatsComputer(df)
    h2h = H2HComputer(df)

    X_df, y_ou25, y_btts, weights = build_betting_dataset(df, elos, rolling, h2h)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_df.values)
    pickle.dump(scaler, open(MODELS_DIR / "scaler_betting.pkl", "wb"))

    print("\n[+] Training Over/Under 2.5 Goals Model (XGBoost)")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    ou_model = xgb.XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.03, 
        eval_metric="logloss", random_state=42, verbosity=0
    )
    
    ou_oof = np.zeros(len(y_ou25))
    for tr, val in skf.split(X_scaled, y_ou25):
        ou_model.fit(X_scaled[tr], y_ou25[tr], sample_weight=weights[tr])
        ou_oof[val] = ou_model.predict_proba(X_scaled[val])[:, 1]
    
    ou_pred = (ou_oof > 0.5).astype(int)
    print(f"  OU2.5 Log Loss: {log_loss(y_ou25, ou_oof):.4f}")
    print(f"  OU2.5 Accuracy: {accuracy_score(y_ou25, ou_pred):.4f}")
    
    ou_model.fit(X_scaled, y_ou25, sample_weight=weights)
    pickle.dump(ou_model, open(MODELS_DIR / "model_ou25.pkl", "wb"))

    print("\n[+] Training Both Teams To Score (BTTS) Model (LightGBM)")
    btts_model = lgb.LGBMClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.03, 
        random_state=42, verbose=-1
    )
    
    btts_oof = np.zeros(len(y_btts))
    for tr, val in skf.split(X_scaled, y_btts):
        btts_model.fit(X_scaled[tr], y_btts[tr], sample_weight=weights[tr])
        btts_oof[val] = btts_model.predict_proba(X_scaled[val])[:, 1]
        
    btts_pred = (btts_oof > 0.5).astype(int)
    print(f"  BTTS Log Loss: {log_loss(y_btts, btts_oof):.4f}")
    print(f"  BTTS Accuracy: {accuracy_score(y_btts, btts_pred):.4f}")
    
    btts_model.fit(X_scaled, y_btts, sample_weight=weights)
    pickle.dump(btts_model, open(MODELS_DIR / "model_btts.pkl", "wb"))

    print("\nModels successfully trained and saved!")

if __name__ == "__main__":
    train_betting_markets()
