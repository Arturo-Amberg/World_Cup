import pandas as pd
import numpy as np

print("--- SANITY CHECK: VALUE BETS ---")
try:
    df_bets = pd.read_csv("data/coolbet/model_value_bets.csv")
    print(f"Loaded {len(df_bets)} bets.")
    
    # Check for NaNs
    nans = df_bets.isna().sum()
    if nans.sum() > 0:
        print("WARNING: Found NaNs in bets data:")
        print(nans[nans > 0])
    else:
        print("No NaNs found in bets data.")
        
    # Check bounds
    if (df_bets['True_Prob'] < 0).any() or (df_bets['True_Prob'] > 1).any():
        print("WARNING: True_Prob out of bounds!")
    else:
        print("True_Prob within [0, 1].")
        
    if (df_bets['Fair_Odds'] <= 0).any():
        print("WARNING: Fair_Odds <= 0 found!")
    else:
        print("Fair_Odds > 0.")
        
    print("\nTop 5 highest EVs:")
    print(df_bets.sort_values("EV_Pct", ascending=False).head(5)[['Match', 'Market', 'Odds', 'Fair_Odds', 'True_Prob', 'EV_Pct']])
    
except Exception as e:
    print("Error checking bets:", e)


print("\n--- SANITY CHECK: TACTICAL PREDICTIONS ---")
try:
    df_tact = pd.read_csv("data/coolbet/tactical_predictions.csv")
    print(f"Loaded {len(df_tact)} tactical predictions.")
    
    nans = df_tact.isna().sum()
    if nans.sum() > 0:
        print("WARNING: Found NaNs in tactical data:")
        print(nans[nans > 0])
    else:
        print("No NaNs found in tactical data.")
        
    # Check reasonable ranges
    if (df_tact['Exp_Total_Corners'] < 0).any() or (df_tact['Exp_Total_Corners'] > 30).any():
        print("WARNING: Extreme corner values!")
        
    if (df_tact['Exp_Total_Shots'] < 0).any() or (df_tact['Exp_Total_Shots'] > 60).any():
        print("WARNING: Extreme shot values!")
        
    if (df_tact['Exp_Home_Poss'] < 0).any() or (df_tact['Exp_Home_Poss'] > 100).any():
        print("WARNING: Home possession out of bounds [0, 100]!")
        
    # Check if possession sums to 100
    poss_sum = df_tact['Exp_Home_Poss'] + df_tact['Exp_Away_Poss']
    if not np.allclose(poss_sum, 100.0, atol=1.0):
        print("WARNING: Possession does not sum to 100%!")
        print(poss_sum)
    else:
        print("Possession sums to 100% correctly.")
        
except Exception as e:
    print("Error checking tactical:", e)

