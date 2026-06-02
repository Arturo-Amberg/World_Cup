import json
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
from src.pipelines.training_pipeline import load_data, load_elos, TacticalStatsComputer, RollingStatsComputer, H2HComputer, build_training_dataset
import os

def main():
    print("Running Segment Analysis (5-Fold CV OOF)...")
    df = load_data()
    elos = load_elos()
    
    tactical = None
    if os.path.exists("data/statsbomb_match_stats.csv"):
        df_tactical = pd.read_csv("data/statsbomb_match_stats.csv")
        tactical = TacticalStatsComputer(df_tactical)

    rolling = RollingStatsComputer(df)
    h2h = H2HComputer(df)
    
    print("Building dataset...")
    X_df, y, weights = build_training_dataset(df, elos, rolling, h2h, tactical=tactical)
    
    with open("data/models/feature_names.json", "r") as f:
        features = json.load(f)
        
    X_features = X_df[features].copy()
    
    # We want to keep some metadata for segments
    # df has tournament, etc. But build_training_dataset skips early matches.
    # X_df has the same index as the matches kept.
    # So we can just join df back to X_df using index.
    
    metadata = df.loc[X_df.index, ['tournament', 'home_team', 'away_team', 'date']].copy()
    metadata['squad_rating_diff'] = X_df['squad_rating_diff'] if 'squad_rating_diff' in X_df else 0
    metadata['altitude_shock_a'] = X_df['altitude_shock_a'] if 'altitude_shock_a' in X_df else 0
    metadata['altitude_shock_b'] = X_df['altitude_shock_b'] if 'altitude_shock_b' in X_df else 0
    metadata['altitude_diff'] = (metadata['altitude_shock_a'] - metadata['altitude_shock_b']).abs()
    
    # KFold CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros((len(y), 3))
    
    print("Training LightGBM on folds to gather OOF predictions...")
    X_np = X_features.values
    
    for train_idx, val_idx in skf.split(X_np, y):
        X_tr, y_tr = X_np[train_idx], y[train_idx]
        X_val, y_val = X_np[val_idx], y[val_idx]
        
        # Simple LGBM for analysis
        model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, random_state=42, verbose=-1)
        model.fit(X_tr, y_tr)
        
        oof_preds[val_idx] = model.predict_proba(X_val)
        
    metadata['y_true'] = y
    metadata['y_pred'] = np.argmax(oof_preds, axis=1)
    for i in range(3):
        metadata[f'prob_{i}'] = oof_preds[:, i]
        
    # Helper to calculate metrics
    def calc_metrics(sub_df):
        if len(sub_df) == 0: return np.nan, np.nan
        ll = log_loss(sub_df['y_true'], sub_df[['prob_0', 'prob_1', 'prob_2']], labels=[0,1,2])
        acc = accuracy_score(sub_df['y_true'], sub_df['y_pred'])
        return ll, acc, len(sub_df)

    print("\n" + "="*50)
    print("SEGMENT ANALYSIS RESULTS")
    print("="*50)
    
    # 1. Overall
    ll, acc, count = calc_metrics(metadata)
    print(f"Overall           | LogLoss: {ll:.4f} | Acc: {acc:.4f} | Count: {count}")
    print("-" * 50)
    
    # 2. By Tournament Type
    print("By Tournament:")
    comps = metadata['tournament'].value_counts().head(5).index
    for c in comps:
        ll, acc, count = calc_metrics(metadata[metadata['tournament'] == c])
        print(f"{c[:15]:<17} | LogLoss: {ll:.4f} | Acc: {acc:.4f} | Count: {count}")
    print("-" * 50)
        
    # 3. By Squad Rating Disparity
    print("By Squad Rating Disparity (Absolute):")
    bins = [0, 5, 10, 20, 100]
    labels = ["0-5 (Tight)", "5-10 (Slight)", "10-20 (Clear Fav)", "20+ (Mismatch)"]
    metadata['squad_bin'] = pd.cut(metadata['squad_rating_diff'].abs(), bins=bins, labels=labels)
    for b in labels:
        ll, acc, count = calc_metrics(metadata[metadata['squad_bin'] == b])
        print(f"{b:<17} | LogLoss: {ll:.4f} | Acc: {acc:.4f} | Count: {count}")
    print("-" * 50)

    # 4. By Altitude Shock
    print("By Altitude Shock Differential:")
    bins = [-1, 100, 500, 1000, 5000]
    labels = ["0-100m (None)", "100-500m (Low)", "500-1km (Med)", "1km+ (Extreme)"]
    metadata['alt_bin'] = pd.cut(metadata['altitude_diff'], bins=bins, labels=labels)
    for b in labels:
        ll, acc, count = calc_metrics(metadata[metadata['alt_bin'] == b])
        print(f"{b:<17} | LogLoss: {ll:.4f} | Acc: {acc:.4f} | Count: {count}")
    print("="*50)

if __name__ == "__main__":
    main()
