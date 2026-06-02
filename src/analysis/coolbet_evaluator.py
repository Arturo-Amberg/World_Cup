import pandas as pd
import numpy as np
import json
import pickle
import os
from src.pipelines.training_pipeline import (
    load_data, load_elos, TacticalStatsComputer, RollingStatsComputer, H2HComputer, build_training_dataset,
    PoissonBaseModel, RollingMeanBaseModel
)

# Mappings for Coolbet names to Historical Data names
TEAM_MAP = {
    "USA": "United States"
}

def map_team(t):
    return TEAM_MAP.get(t, t)

def main():
    print("="*60)
    print("COOLBET VALUE BETTING EVALUATOR")
    print("="*60)
    
    # 1. Load Data
    print("Loading historical data to build rolling features...")
    df = load_data()
    elos = load_elos()
    
    tactical = None
    if os.path.exists("data/statsbomb_match_stats.csv"):
        df_tactical = pd.read_csv("data/statsbomb_match_stats.csv")
        tactical = TacticalStatsComputer(df_tactical)

    # We build the Rolling and H2H engines using ONLY the historical data.
    # This prevents future matches from accidentally bleeding into the past.
    rolling = RollingStatsComputer(df)
    h2h = H2HComputer(df)

    # 2. Parse Coolbet Matches
    coolbet = pd.read_csv("data/coolbet/latest.csv")
    coolbet['home_mapped'] = coolbet['home'].apply(map_team)
    coolbet['away_mapped'] = coolbet['away'].apply(map_team)
    
    # Extract unique matchups
    upcoming = coolbet[['home_mapped', 'away_mapped', 'match_start']].drop_duplicates().copy()
    upcoming['date'] = pd.to_datetime(upcoming['match_start']).dt.tz_localize(None)
    
    print(f"Found {len(upcoming)} upcoming matches on Coolbet.")
    
    # 3. Create dummy historical rows to feed into our feature builder
    hosts = {"United States", "Mexico", "Canada"}
    
    records = []
    for _, row in upcoming.iterrows():
        home = row['home_mapped']
        away = row['away_mapped']
        date = row['date']
        
        is_neutral = True
        
        # If either team is a host, it's not neutral, and the host MUST be the home team
        if home in hosts or away in hosts:
            is_neutral = False
            if away in hosts and home not in hosts:
                # Swap them so the host gets the home advantage
                home, away = away, home
                
        records.append({
            'date': date,
            'home_team': home,
            'away_team': away,
            'home_score': 0,
            'away_score': 0,
            'tournament': 'FIFA World Cup',
            'city': 'Unknown',
            'country': 'North America',
            'neutral': is_neutral
        })
        
    df_upcoming = pd.DataFrame(records)
    
    # Pre-filter to prevent desync in build_training_dataset
    keep_indices = []
    for i, row in df_upcoming.iterrows():
        home = str(row['home_team'])
        away = str(row['away_team'])
        if pd.isna(row['home_team']) or pd.isna(row['away_team']): continue
        if home == "nan" or away == "nan": continue
        if pd.isna(row['date']): continue
        if rolling.n_prior(home, row['date']) < 10: continue
        if rolling.n_prior(away, row['date']) < 10: continue
        keep_indices.append(i)
        
    df_upcoming = df_upcoming.loc[keep_indices].reset_index(drop=True)
    
    # 4. Generate Features
    print("Generating features for upcoming matches...")
    X_upcoming, _, _ = build_training_dataset(df_upcoming, elos, rolling, h2h, tactical=tactical)
    
    # Ensure columns match our training data
    with open("data/models/feature_names.json", "r") as f:
        features = json.load(f)
    
    # Missing columns will be 0 (some tactical diffs might be missing if a team had no tactical data)
    missing_cols = []
    for col in features:
        if col not in X_upcoming:
            X_upcoming[col] = 0.0
            missing_cols.append(col)
    if missing_cols:
        print(f"Warning: Missing features filled with 0.0: {missing_cols}")
            
    X_features = X_upcoming[features].copy()
    
    # We need to scale using our saved scaler
    with open("data/models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    X_scaled = scaler.transform(X_features)

    # 5. Predict True Probabilities
    print("\nLoading models and predicting True Probabilities...")
    with open("data/models/meta_lgbm.pkl", "rb") as f:
        meta_lgbm = pickle.load(f)
    
    # Base models for stacking
    with open("data/models/training_report.json", "r") as f:
        report = json.load(f)
    base_names = report["base_model_names"]
    
    base_preds = []
    for m in base_names:
        with open(f"data/models/base_{m}.pkl", "rb") as f:
            model = pickle.load(f)
            
        if m in {"poisson", "rolling_mean"}:
            p = np.full((len(X_scaled), 3), 1.0 / 3.0)
        elif hasattr(model, "predict_proba"):
            p = model.predict_proba(X_scaled)
        else:
            p = np.zeros((len(X_scaled), 3))
        base_preds.append(p)
        
    X_meta = np.hstack(base_preds)
    with open("data/models/meta_scaler.pkl", "rb") as f:
        meta_scaler = pickle.load(f)
    X_meta_scaled = meta_scaler.transform(X_meta)
    
    true_1x2_probs = meta_lgbm.predict_proba(X_meta_scaled)
    
    # Load Prop Models
    with open("data/models/model_over25.pkl", "rb") as f:
        model_over25 = pickle.load(f)
    with open("data/models/model_btts.pkl", "rb") as f:
        model_btts = pickle.load(f)
        
    # Prop models were trained directly on unscaled X_features!
    true_over25_probs = model_over25.predict_proba(X_features)[:, 1]
    true_btts_probs = model_btts.predict_proba(X_features)[:, 1]

    # Assign True Probs back to df_upcoming
    df_upcoming['prob_home_win'] = true_1x2_probs[:, 0]
    df_upcoming['prob_draw'] = true_1x2_probs[:, 1]
    df_upcoming['prob_away_win'] = true_1x2_probs[:, 2]
    df_upcoming['prob_over25'] = true_over25_probs
    df_upcoming['prob_btts'] = true_btts_probs
    df_upcoming['squad_rating_diff'] = X_features['squad_rating_diff'].values
    
    # 6. Evaluate Coolbet Odds
    print("\nEvaluating +EV Bets...")
    value_bets = []
    
    for _, cb_row in coolbet.iterrows():
        match_dt = pd.to_datetime(cb_row['match_start']).tz_localize(None)
        
        # Find corresponding true probs. Teams might have been swapped to give the host the home advantage.
        match_data = df_upcoming[
            ((df_upcoming['home_team'] == cb_row['home_mapped']) & (df_upcoming['away_team'] == cb_row['away_mapped'])) |
            ((df_upcoming['home_team'] == cb_row['away_mapped']) & (df_upcoming['away_team'] == cb_row['home_mapped']))
        ]
        if len(match_data) == 0:
            continue
            
        md = match_data.iloc[0]
        swapped = (md['home_team'] != cb_row['home_mapped'])
        
        odds = float(cb_row['odds'])
        market = cb_row['market']
        sel = cb_row['selection']
        
        true_prob = None
        bet_type = None
        
        # Match Result 1X2
        if market == 'Match Result (1X2)':
            if sel == cb_row['home']:
                true_prob = md['prob_away_win'] if swapped else md['prob_home_win']
                bet_type = '1X2 (Home)'
            elif sel == 'Draw':
                true_prob = md['prob_draw']
                bet_type = '1X2 (Draw)'
            elif sel == cb_row['away']:
                true_prob = md['prob_home_win'] if swapped else md['prob_away_win']
                bet_type = '1X2 (Away)'
                
        # Total Goals O/U 2.5
        elif market == 'Total Goals Over / Under' and float(cb_row['line']) == 2.5:
            if sel == 'Over':
                true_prob = md['prob_over25']
                bet_type = 'O/U 2.5 (Over)'
            elif sel == 'Under':
                true_prob = 1.0 - md['prob_over25']
                bet_type = 'O/U 2.5 (Under)'
                
        # BTTS
        elif market == 'Both Teams to Score':
            if sel == 'Yes':
                true_prob = md['prob_btts']
                bet_type = 'BTTS (Yes)'
            elif sel == 'No':
                true_prob = 1.0 - md['prob_btts']
                bet_type = 'BTTS (No)'
                
        if true_prob is not None:
            ev = (true_prob * odds) - 1.0
            
            value_bets.append({
                'Match': f"{cb_row['home']} vs {cb_row['away']}",
                'Market': bet_type,
                'Odds': odds,
                'True_Prob': true_prob,
                'EV_Pct': ev * 100,
                'Squad_Diff': abs(md['squad_rating_diff'])
            })
            
    # 10. Generate Final Output DataFrame
    df_eval = pd.DataFrame(value_bets)
    if not df_eval.empty:
        df_eval = df_eval[df_eval['EV_Pct'] > 0.0].sort_values("EV_Pct", ascending=False)
        
    if not df_eval.empty:
        # Add Fair Odds and Kelly Criterion
        df_eval["Fair_Odds"] = 1.0 / df_eval["True_Prob"]
        df_eval["Kelly_Pct"] = (df_eval["EV_Pct"] / 100.0) / (df_eval["Odds"] - 1.0) * 100.0
        
        # Cap Kelly to avoid massive recommendations, typically bettors use quarter-Kelly anyway, but we just show the raw % capped at 100.
        df_eval["Kelly_Pct"] = df_eval["Kelly_Pct"].clip(upper=100.0)
        
        df_eval.to_csv("data/coolbet/model_value_bets.csv", index=False)
        
        print(f"\nFound {len(df_eval)} +EV Bets!\n")
        
        print("--- TOP HIGH-CONFIDENCE VALUE BETS (Squad Diff > 15) ---")
        high_conf = df_eval[(df_eval["Squad_Diff"] > 15) & (df_eval["Market"].str.startswith("1X2"))]
        cols_to_print = ["Match", "Market", "Odds", "Fair_Odds", "True_Prob", "EV_Pct", "Kelly_Pct"]
        if not high_conf.empty:
            print(high_conf.head(10)[cols_to_print].to_string(index=False))
        else:
            print("No high-confidence 1X2 mismatches found currently.")
            
        print("\n--- TOP PROP BETS (Over/Under & BTTS) ---")
        props = df_eval[~df_eval["Market"].str.startswith("1X2")]
        if not props.empty:
            print(props.head(10)[cols_to_print].to_string(index=False))
        else:
            print("No +EV prop bets found.")
            
        print("\nFull +EV report saved to data/coolbet/model_value_bets.csv")
    else:
        print("\nNo +EV bets found under the current model thresholds.")

    # 11. Predict Tactical Stats (Corners and Shots)
    print("\n============================================================")
    print("TACTICAL PREDICTIONS (EXPECTED CORNERS, SHOTS & POSSESSION)")
    print("============================================================")
    try:
        with open("data/models/model_corners.pkl", "rb") as f:
            model_corners = pickle.load(f)
        with open("data/models/model_shots.pkl", "rb") as f:
            model_shots = pickle.load(f)
        with open("data/models/model_possession.pkl", "rb") as f:
            model_possession = pickle.load(f)
            
        pred_corners = model_corners.predict(X_scaled)
        pred_shots = model_shots.predict(X_scaled)
        pred_poss = model_possession.predict(X_scaled)
        
        df_tact = pd.DataFrame({
            "Match": df_upcoming["home_team"] + " vs " + df_upcoming["away_team"],
            "Exp_Total_Corners": pred_corners,
            "Exp_Total_Shots": pred_shots,
            "Exp_Home_Poss": pred_poss,
            "Exp_Away_Poss": 100.0 - pred_poss
        })
        
        print(df_tact.to_string(index=False, float_format="%.1f"))
        df_tact.to_csv("data/coolbet/tactical_predictions.csv", index=False)
        print("\nTactical predictions saved to data/coolbet/tactical_predictions.csv")
        
    except Exception as e:
        print(f"\nCould not generate tactical predictions. Error: {e}")

if __name__ == "__main__":
    main()
