"""
compare_models.py — Side-by-side comparison of all prediction models.

Usage:
    python compare_models.py          # backtest + quick MC (1000 sims)
    python compare_models.py --full   # backtest + full MC (10 000 sims)
"""
import sys
import json
import os
import math
from collections import defaultdict

from stacked_predictor import elo_model, poisson_model, DEFAULT_WEIGHTS
from ml_predictor import MLPredictor, TRAINING_DATA, RESULT_MAP, _make_features

# ─────────────────────────────────────────────
#  Backtest helpers (reuse calibrate data)
# ─────────────────────────────────────────────
def _log_loss(probs, result):
    eps = 1e-9
    idx = {"A": 0, "D": 1, "B": 2}[result]
    return -math.log(max(eps, probs[idx]))

def _correct(probs, result):
    idx = {"A": 0, "D": 1, "B": 2}[result]
    return 1 if int(probs.index(max(probs))) == idx else 0

def backtest_classical():
    """Run ELO / Poisson / Form / Blend on training data."""
    from calibrate import WC2022_MATCHES, _make_teams, compute_log_loss, accuracy
    results_list = [m["result"] for m in WC2022_MATCHES]
    elo_p, poi_p = [], []
    for m in WC2022_MATCHES:
        ta, tb = _make_teams(m)
        elo_p.append(elo_model(ta, tb))
        poi_p.append(poisson_model(ta, tb))

    # Blend (no ML)
    w = DEFAULT_WEIGHTS
    wt = w["elo"] + w["poisson"]
    blend = []
    for e, po in zip(elo_p, poi_p):
        pa = (w["elo"]*e[0] + w["poisson"]*po[0]) / wt
        pd = (w["elo"]*e[1] + w["poisson"]*po[1]) / wt
        pb = (w["elo"]*e[2] + w["poisson"]*po[2]) / wt
        s  = pa+pd+pb
        blend.append((pa/s, pd/s, pb/s))

    return {
        "ELO":     {"ll": compute_log_loss(elo_p, results_list),  "acc": accuracy(elo_p, results_list)},
        "Poisson": {"ll": compute_log_loss(poi_p, results_list),  "acc": accuracy(poi_p, results_list)},
        "Blend2":  {"ll": compute_log_loss(blend, results_list),  "acc": accuracy(blend, results_list)},
        "n": len(WC2022_MATCHES),
    }


# ─────────────────────────────────────────────
#  Mini Monte Carlo for comparison table
# ─────────────────────────────────────────────
def mini_mc(n_sims: int, label: str) -> list:
    from tournament import load_team_db, load_groups, monte_carlo
    team_db = load_team_db()
    groups  = load_groups()
    print(f"  Running {n_sims:,} sims [{label}]...")
    return monte_carlo(groups, team_db, n=n_sims)


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────
def main():
    full = "--full" in sys.argv
    n_sims = 10000 if full else 5000

    print("=" * 65)
    print("  WORLD CUP 2026 — MODEL COMPARISON REPORT")
    print("=" * 65)

    # ── 1. Classical models (WC 2022 backtest) ──────────────────
    print("\n▶ Classical model backtest (WC 2022 group stage):")
    try:
        cls = backtest_classical()
        print(f"  {'Model':<12} {'Log-loss':>10} {'Accuracy':>10}")
        print("  " + "-" * 34)
        for name, stat in cls.items():
            if name == "n":
                continue
            print(f"  {name:<12} {stat['ll']:>10.4f} {stat['acc']*100:>9.1f}%")
        print(f"  (n={cls['n']} matches)")
    except Exception as e:
        print(f"  ⚠ Classical backtest failed: {e}")

    # ── 2. ML model backtest (all tournament data) ───────────────
    print("\n▶ ML Ensemble backtest (WC 2018/2022 + EURO 2020/2024):")
    ml = MLPredictor()
    ml.fit()
    ml_stats = ml.backtest_accuracy()
    print(f"  RandomForest+GBoost  log-loss={ml_stats['log_loss']:.4f}  "
          f"accuracy={ml_stats['accuracy']*100:.1f}%  n={ml_stats['n_samples']}")

    # ── 3. Key match predictions comparison ─────────────────────
    print("\n▶ Key match predictions (A=win_a  D=draw  B=win_b):")
    matchups = [
        ("Spain",    {"ELO":2070,"FORMA":2.3,"GF_AVG":2.0,"GA_AVG":0.6,"INJURIES":0},
         "France",   {"ELO":2081,"FORMA":2.1,"GF_AVG":1.9,"GA_AVG":0.8,"INJURIES":0}),
        ("Spain",    {"ELO":2070,"FORMA":2.3,"GF_AVG":2.0,"GA_AVG":0.6,"INJURIES":0},
         "Argentina",{"ELO":2113,"FORMA":2.4,"GF_AVG":2.1,"GA_AVG":0.6,"INJURIES":0}),
        ("France",   {"ELO":2081,"FORMA":2.1,"GF_AVG":1.9,"GA_AVG":0.8,"INJURIES":0},
         "England",  {"ELO":2021,"FORMA":1.9,"GF_AVG":1.7,"GA_AVG":0.9,"INJURIES":0}),
        ("Germany",  {"ELO":1923,"FORMA":1.8,"GF_AVG":1.9,"GA_AVG":1.1,"INJURIES":0},
         "Brazil",   {"ELO":2067,"FORMA":2.0,"GF_AVG":1.8,"GA_AVG":0.7,"INJURIES":0}),
    ]
    print(f"  {'Matchup':<30} {'ELO':>12} {'Poisson':>12} {'ML':>12}")
    print("  " + "-" * 68)
    for na, ta, nb, tb in matchups:
        ta["name"] = na; tb["name"] = nb
        ep  = elo_model(ta, tb)
        pp  = poisson_model(ta, tb)
        mp  = ml.predict(ta, tb)
        lbl = f"{na} vs {nb}"
        fmt = lambda p: f"{p[0]:.2f}/{p[1]:.2f}/{p[2]:.2f}"
        print(f"  {lbl:<30} {fmt(ep):>12} {fmt(pp):>12} {fmt(mp):>12}")

    # ── 4. Monte Carlo ───────────────────────────────────────────
    print(f"\n▶ Monte Carlo ({n_sims:,} simulations — uses full blended model):")
    mc_res = mini_mc(n_sims, "Blended (ELO+Poisson+ML)")
    table = mc_res["table"]
    metrics = mc_res["metrics"]

    print(f"\n  {'Team':<22} {'Champion%':>10} {'Final%':>8} {'Semi%':>7} {'Quarter%':>9}")
    print("  " + "-" * 60)
    for row in table[:20]:
        print(f"  {row['team']:<22} {row['champion_%']:>9.2f}%  "
              f"{row['final_%']:>6.2f}%  {row['semi_%']:>5.2f}%  {row['quarter_%']:>7.2f}%")
              
    print("\n  ▶ GOAL METRICS:")
    print(f"    Avg Goals / Tournament : {metrics['avg_goals']}")
    print(f"    Avg Over 2.5 Matches   : {metrics['avg_over_2_5']}")
    print(f"    Expected Over 2.5 %    : {metrics['pct_over_2_5']}%")

    # Save
    out = os.path.join(os.path.dirname(__file__), "data", "mc_results.json")
    with open(out, "w") as f:
        json.dump(mc_res, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved → data/mc_results.json")
    print("=" * 65)


if __name__ == "__main__":
    main()
