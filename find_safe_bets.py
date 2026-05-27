
import json
import os
from stacked_predictor import stack_predict
from tournament import load_team_db, load_groups

def find_strategies():
    team_db = load_team_db()
    groups = load_groups()
    
    print("\nCalculating precise strategies...")
    
    # 1. Group Qualification Probabilities (High Fidelity)
    n_sims = 5000
    qual_counts = {t: 0 for t in team_db}
    for _ in range(n_sims):
        from tournament import sim_group_stage
        results, best_thirds, _, _ = sim_group_stage(groups, team_db)
        qualified = set()
        for letter, res in results.items():
            qualified.add(res['first'])
            qualified.add(res['second'])
        for t in best_thirds:
            qualified.add(t)
        for t in qualified:
            if t in qual_counts:
                qual_counts[t] += 1
                
    qual_probs = {t: count / n_sims for t, count in qual_counts.items()}
    
    # Define Tiers
    spain = qual_probs.get('Spain', 0)
    england = qual_probs.get('England', 0)
    france = qual_probs.get('France', 0)
    argentina = qual_probs.get('Argentina', 0)
    germany = qual_probs.get('Germany', 0)
    brazil = qual_probs.get('Brazil', 0)
    belgium = qual_probs.get('Belgium', 0)
    colombia = qual_probs.get('Colombia', 0)

    # --- 90% STRATEGY ---
    # Strategy 1: The "Elite Trio" (Qualification)
    prob_90 = spain * england * france
    
    # --- 80% STRATEGY ---
    # Strategy 2: The "Big Five" (Qualification)
    prob_80 = spain * england * france * argentina * germany

    print("\n" + "="*50)
    print("   BETTING STRATEGY REPORT (FIFA WC 2026)")
    print("="*50)

    print(f"\n[STRATEGY A] THE 90% ULTRA-SAFE (Qualification Parlay)")
    print(f"Goal: High win rate, lower multiplier.")
    print(f"Components:")
    print(f"  - Spain to qualify     ({spain*100:.1f}%)")
    print(f"  - England to qualify   ({england*100:.1f}%)")
    print(f"  - France to qualify    ({france*100:.1f}%)")
    print(f"COMBINED PROBABILITY: {prob_90*100:.2f}%")
    print(f"Recommended Action: Accumulator (Parlay) or Sequential All-In.")

    print(f"\n[STRATEGY B] THE 80% HIGH-VALUE (Qualification Parlay)")
    print(f"Goal: Balanced risk/reward.")
    print(f"Components:")
    print(f"  - Spain to qualify     ({spain*100:.1f}%)")
    print(f"  - England to qualify   ({england*100:.1f}%)")
    print(f"  - France to qualify    ({france*100:.1f}%)")
    print(f"  - Argentina to qualify ({argentina*100:.1f}%)")
    print(f"  - Germany to qualify   ({germany*100:.1f}%)")
    print(f"COMBINED PROBABILITY: {prob_80*100:.2f}%")
    print(f"Recommended Action: Standard Parlay.")

    print(f"\n[ADD-ON] MATCH SPECIFIC BOOSTERS (90%+ Confidence)")
    print(f"You can swap one of the above for these if odds are better:")
    # Recalculate match DC
    m_ec = stack_predict(team_db['Ecuador'], team_db['Curaçao'])
    p_ec_dc = m_ec['p_win_a'] + m_ec['p_draw']
    print(f"  - Ecuador Win or Draw vs Curaçao ({p_ec_dc*100:.1f}%)")
    
    m_ge = stack_predict(team_db['Germany'], team_db['Curaçao'])
    p_ge_dc = m_ge['p_win_a'] + m_ge['p_draw']
    print(f"  - Germany Win or Draw vs Curaçao ({p_ge_dc*100:.1f}%)")

if __name__ == "__main__":
    find_strategies()
