import pandas as pd
import numpy as np

print("--- SANITY CHECK: 1X2 PROBS ---")
df = pd.read_csv("data/coolbet/model_predictions_full.csv")

if 'prob_home' in df.columns:
    probs = df['prob_home'] + df['prob_draw'] + df['prob_away']
    print(f"Mean sum of 1X2 probs: {probs.mean():.4f}")
    if not np.allclose(probs, 1.0, atol=1e-3):
        print("WARNING: 1X2 probabilities do not sum to 1.0!")
        print(probs[~np.isclose(probs, 1.0, atol=1e-3)])
    else:
        print("1X2 probabilities correctly sum to 1.0 for all matches.")
else:
    print("Full predictions CSV not found or missing columns.")
