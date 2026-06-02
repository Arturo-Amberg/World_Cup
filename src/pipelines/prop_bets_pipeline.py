import pandas as pd
import numpy as np
import os
import json
import pickle
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import log_loss, accuracy_score, mean_squared_error
from src.pipelines.training_pipeline import load_data, load_elos, TacticalStatsComputer, RollingStatsComputer, H2HComputer, build_training_dataset

def main():
    print("Loading data...")
    df = load_data()
    elos = load_elos()
    
    tactical = None
    if os.path.exists("data/statsbomb_match_stats.csv"):
        df_tactical = pd.read_csv("data/statsbomb_match_stats.csv")
        tactical = TacticalStatsComputer(df_tactical)

    rolling = RollingStatsComputer(df)
    h2h = H2HComputer(df)
    
    print("Building base dataset...")
    X_df, _, _ = build_training_dataset(df, elos, rolling, h2h, tactical=tactical)
    
    with open("data/models/feature_names.json", "r") as f:
        features = json.load(f)
        
    X_features = X_df[features].copy()
    
    # Target Construction
    # For goals, we can use the full df matching X_features index
    df_targets = df.loc[X_features.index].copy()
    y_over_2_5 = ((df_targets['home_score'] + df_targets['away_score']) > 2.5).astype(int)
    y_btts = ((df_targets['home_score'] > 0) & (df_targets['away_score'] > 0)).astype(int)
    
    print("\n" + "="*50)
    print("PROP BET PIPELINES")
    print("="*50)

    # 1. BTTS
    print("\nTraining BTTS (Both Teams To Score)...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_btts = np.zeros(len(y_btts))
    X_np = X_features.values
    for train_idx, val_idx in skf.split(X_np, y_btts):
        model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, random_state=42, verbose=-1)
        model.fit(X_np[train_idx], y_btts.iloc[train_idx])
        oof_btts[val_idx] = model.predict_proba(X_np[val_idx])[:, 1]
        
    btts_acc = accuracy_score(y_btts, (oof_btts > 0.5).astype(int))
    btts_ll = log_loss(y_btts, oof_btts)
    print(f"BTTS OOF -> LogLoss: {btts_ll:.4f} | Accuracy: {btts_acc:.4f}")
    
    print("Saving Final BTTS Model...")
    final_btts = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, random_state=42, verbose=-1)
    final_btts.fit(X_np, y_btts)
    with open("data/models/model_btts.pkl", "wb") as f:
        pickle.dump(final_btts, f)

    # 2. Over 2.5
    print("\nTraining Over/Under 2.5 Goals...")
    oof_over = np.zeros(len(y_over_2_5))
    for train_idx, val_idx in skf.split(X_np, y_over_2_5):
        model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, random_state=42, verbose=-1)
        model.fit(X_np[train_idx], y_over_2_5.iloc[train_idx])
        oof_over[val_idx] = model.predict_proba(X_np[val_idx])[:, 1]
        
    over_acc = accuracy_score(y_over_2_5, (oof_over > 0.5).astype(int))
    over_ll = log_loss(y_over_2_5, oof_over)
    print(f"Over 2.5 OOF -> LogLoss: {over_ll:.4f} | Accuracy: {over_acc:.4f}")
    
    print("Saving Final Over 2.5 Model...")
    final_over = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, random_state=42, verbose=-1)
    final_over.fit(X_np, y_over_2_5)
    with open("data/models/model_over25.pkl", "wb") as f:
        pickle.dump(final_over, f)

    # 3. Tactical Targets (Corners, Shots)
    if tactical is not None:
        print("\nAligning Tactical Targets (Corners & Shots)...")
        # We need to map tactical data using date and home_team.
        # statsbomb_match_stats.csv has 'date' and 'home_team'
        # Or match_id? statsbomb might have different match_ids.
        # Ensure dates are strings for joining
        df_tactical['date'] = df_tactical['date'].astype(str)
        df_targets['date'] = df_targets['date'].dt.strftime('%Y-%m-%d')
        
        # We need the X_features for these specific matches!
        # The index of df_targets is original df index. The merge on columns will lose the index.
        
        df_tactical_indexed = df_tactical.set_index(['date', 'home_team'])
        df_targets_indexed = df_targets.reset_index().set_index(['date', 'home_team'])
        
        print("df_tactical_indexed length:", len(df_tactical_indexed))
        print("df_targets_indexed length:", len(df_targets_indexed))
        
        joined = df_targets_indexed.join(df_tactical_indexed[['home_corners', 'away_corners', 'home_shots', 'away_shots', 'home_possession', 'away_possession']], how='inner')
        print("joined length:", len(joined))
        
        valid_indices = joined['index'].values
        
        X_tact = X_features.loc[valid_indices]
        y_corners = joined['home_corners'].values + joined['away_corners'].values
        y_shots = joined['home_shots'].values + joined['away_shots'].values
        
        print(f"Found {len(X_tact)} matches with tactical targets.")
        
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        X_tact_np = X_tact.values
        
        # Corners
        oof_corners = np.zeros(len(y_corners))
        for train_idx, val_idx in kf.split(X_tact_np):
            model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, random_state=42, verbose=-1)
            model.fit(X_tact_np[train_idx], y_corners[train_idx])
            oof_corners[val_idx] = model.predict(X_tact_np[val_idx])
            
        rmse_corners = np.sqrt(mean_squared_error(y_corners, oof_corners))
        print(f"Total Corners OOF -> RMSE: {rmse_corners:.2f} (Mean: {y_corners.mean():.2f})")

        # Train and save final Corners model
        final_corners = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, random_state=42, verbose=-1)
        final_corners.fit(X_tact_np, y_corners)
        with open("data/models/model_corners.pkl", "wb") as f:
            pickle.dump(final_corners, f)

        # Shots
        oof_shots = np.zeros(len(y_shots))
        for train_idx, val_idx in kf.split(X_tact_np):
            model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, random_state=42, verbose=-1)
            model.fit(X_tact_np[train_idx], y_shots[train_idx])
            oof_shots[val_idx] = model.predict(X_tact_np[val_idx])
            
        rmse_shots = np.sqrt(mean_squared_error(y_shots, oof_shots))
        print(f"Total Shots OOF   -> RMSE: {rmse_shots:.2f} (Mean: {y_shots.mean():.2f})")
        
        # Train and save final Shots model
        final_shots = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, random_state=42, verbose=-1)
        final_shots.fit(X_tact_np, y_shots)
        with open("data/models/model_shots.pkl", "wb") as f:
            pickle.dump(final_shots, f)

        # Possession
        y_poss = joined['home_possession'].values
        oof_poss = np.zeros(len(y_poss))
        for train_idx, val_idx in kf.split(X_tact_np):
            model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, random_state=42, verbose=-1)
            model.fit(X_tact_np[train_idx], y_poss[train_idx])
            oof_poss[val_idx] = model.predict(X_tact_np[val_idx])
            
        rmse_poss = np.sqrt(mean_squared_error(y_poss, oof_poss))
        print(f"Home Possession OOF -> RMSE: {rmse_poss:.2f}% (Mean: {y_poss.mean():.2f}%)")
        
        # Train and save final Possession model
        final_poss = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, random_state=42, verbose=-1)
        final_poss.fit(X_tact_np, y_poss)
        with open("data/models/model_possession.pkl", "wb") as f:
            pickle.dump(final_poss, f)

if __name__ == "__main__":
    main()
