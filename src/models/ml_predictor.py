"""ML Ensemble predictor: Random Forest + GradientBoosting."""
import math
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Training data: WC 2018 + WC 2022 + EURO 2020 + EURO 2024 matches
# Each row: (elo_a, elo_b, gf_a, ga_a, gf_b, ga_b, forma_a, forma_b, is_host_a, result)
# result: 0=A wins, 1=Draw, 2=B wins
TRAINING_DATA = [
    # WC 2022
    (1575,1714,0.9,1.3,1.3,1.1,1.1,1.5,0,"A"),  # Qatar 1-0 Ecuador (upset)
    (1798,1906,1.4,1.0,1.7,1.0,1.6,1.8,0,"B"),  # Senegal 0-2 Netherlands
    (1575,1798,0.9,1.3,1.4,1.0,1.1,1.6,0,"B"),  # Qatar 1-3 Senegal
    (1906,1714,1.7,1.0,1.3,1.1,1.8,1.5,0,"D"),  # Netherlands 1-1 Ecuador
    (1906,1575,1.7,1.0,0.9,1.3,1.8,1.1,0,"A"),  # Netherlands 2-0 Qatar
    (1714,1798,1.3,1.1,1.4,1.0,1.5,1.6,0,"B"),  # Ecuador 1-2 Senegal
    (1915,1783,1.7,0.9,1.3,1.0,1.9,1.5,0,"A"),  # England 6-2 Iran
    (1817,1770,1.4,1.1,1.2,1.1,1.5,1.4,0,"D"),  # USA 1-1 Wales
    (1915,1817,1.7,0.9,1.4,1.1,1.9,1.5,0,"D"),  # England 0-0 USA
    (1770,1783,1.2,1.1,1.3,1.0,1.4,1.5,0,"B"),  # Wales 0-2 Iran
    (1915,1770,1.7,0.9,1.2,1.1,1.9,1.4,0,"A"),  # England 3-0 Wales
    (1783,1817,1.3,1.0,1.4,1.1,1.5,1.5,0,"B"),  # Iran 0-1 USA
    (1974,1651,2.1,0.6,1.1,1.2,2.4,1.3,0,"B"),  # Argentina 1-2 Saudi (UPSET)
    (1860,1796,1.5,1.2,1.3,1.1,1.5,1.4,0,"D"),  # Mexico 0-0 Poland
    (1974,1860,2.1,0.6,1.5,1.2,2.4,1.5,0,"A"),  # Argentina 2-0 Mexico
    (1796,1651,1.3,1.1,1.1,1.2,1.4,1.3,0,"A"),  # Poland 2-0 Saudi Arabia
    (1974,1796,2.1,0.6,1.3,1.1,2.4,1.4,0,"A"),  # Argentina 2-0 Poland
    (1651,1860,1.1,1.2,1.5,1.2,1.3,1.5,0,"B"),  # Saudi Arabia 1-2 Mexico
    (1771,1876,1.5,0.9,1.9,1.1,1.7,1.8,0,"A"),  # Japan 2-1 Germany (UPSET)
    (1886,1672,1.8,0.7,1.0,1.2,2.0,1.2,0,"A"),  # Spain 7-0 Costa Rica
    (1771,1672,1.5,0.9,1.0,1.2,1.7,1.2,0,"B"),  # Japan 0-1 Costa Rica
    (1876,1886,1.9,1.1,1.8,0.7,1.8,2.0,0,"D"),  # Germany 1-1 Spain
    (1771,1886,1.5,0.9,1.8,0.7,1.7,2.0,0,"A"),  # Japan 2-1 Spain (UPSET)
    (1672,1876,1.0,1.2,1.9,1.1,1.2,1.8,0,"B"),  # Costa Rica 2-4 Germany
    (1934,1756,1.6,0.9,1.3,1.1,1.7,1.4,0,"A"),  # Belgium 1-0 Canada
    (1712,1862,1.3,0.7,1.4,0.9,1.7,1.7,0,"D"),  # Morocco 0-0 Croatia
    (1934,1712,1.6,0.9,1.3,0.7,1.7,1.7,0,"B"),  # Belgium 0-2 Morocco (UPSET)
    (1862,1756,1.4,0.9,1.3,1.1,1.7,1.4,0,"A"),  # Croatia 4-1 Canada
    (1712,1756,1.3,0.7,1.3,1.1,1.7,1.4,0,"A"),  # Morocco 2-1 Canada
    (1862,1934,1.4,0.9,1.6,0.9,1.7,1.7,0,"D"),  # Croatia 0-0 Belgium
    (1962,1821,1.8,0.7,1.4,1.1,2.0,1.5,0,"A"),  # Brazil 2-0 Serbia
    (1860,1716,1.5,0.8,1.2,1.2,1.7,1.4,0,"A"),  # Switzerland 1-0 Cameroon
    (1962,1860,1.8,0.7,1.5,0.8,2.0,1.7,0,"A"),  # Brazil 1-0 Switzerland
    (1716,1821,1.2,1.2,1.4,1.1,1.4,1.5,0,"D"),  # Cameroon 3-3 Serbia
    (1962,1716,1.8,0.7,1.2,1.2,2.0,1.4,0,"B"),  # Brazil 0-1 Cameroon (reserves)
    (1821,1860,1.4,1.1,1.5,0.8,1.5,1.7,0,"B"),  # Serbia 2-3 Switzerland
    # WC 2022 knockouts
    (1906,1974,1.7,1.0,2.1,0.6,1.8,2.4,0,"D"),  # Netherlands 2-2 Argentina (pens→ARG) — 90min draw
    (1962,1712,1.8,0.7,1.3,0.7,2.0,1.7,0,"A"),  # Brazil 4-1 South Korea
    (1876,2060,1.9,1.1,2.0,0.6,1.8,2.1,0,"D"),  # Germany 1-1 Spain (WC2022 group stage — genuine draw)
    (1771,1862,1.5,0.9,1.4,0.9,1.7,1.7,0,"D"),  # Japan 1-1 Croatia (pens→CRO) — 90min draw
    (1886,1712,1.8,0.7,1.3,0.7,2.0,1.7,0,"D"),  # Spain 0-0 Morocco (pens→MAR) — 90min draw
    # WC 2018
    (1835,1897,1.5,1.0,1.8,0.9,1.7,1.9,0,"D"),  # Uruguay vs Russia WC2018 group (D)
    (2010,1940,1.9,0.7,1.7,0.9,2.2,1.8,0,"A"),  # France vs Croatia Final
    (1870,1930,1.6,0.9,1.8,0.8,1.8,2.0,0,"B"),  # Belgium 0-1 France (SF)
    (1820,1870,1.5,1.0,1.6,0.9,1.6,1.8,0,"B"),  # England 1-2 Belgium (3rd place)
    (1940,1870,1.7,0.9,1.6,0.9,1.9,1.8,0,"A"),  # Croatia 2-1 England (SF)
    (1860,1820,1.6,0.9,1.5,1.0,1.8,1.6,0,"D"),  # Russia 1-1 Spain (pens→RUS) — 90min draw
    (1880,1840,1.7,0.8,1.5,0.9,1.9,1.7,0,"A"),  # Uruguay 2-0 Portugal
    (1910,1880,1.8,0.7,1.7,0.8,2.0,1.9,0,"A"),  # France 2-0 Uruguay QF
    (1840,1900,1.5,0.9,1.7,0.8,1.7,1.9,0,"B"),  # Portugal 1-1 Spain (D) WC2018 group
    (1820,1880,1.5,1.0,1.6,0.9,1.7,1.8,0,"B"),  # Japan 2-3 Belgium (Belgium won in regular time)
    (1840,1880,1.5,0.9,1.7,0.8,1.7,1.9,0,"D"),  # Iran 1-1 Portugal
    (1770,1830,1.4,1.1,1.5,0.9,1.6,1.7,0,"B"),  # Germany 0-2 South Korea (UPSET)
    (1900,1770,1.8,0.8,1.4,1.1,2.0,1.6,0,"A"),  # Spain 1-0 Morocco
    (1930,1880,1.9,0.7,1.7,0.8,2.1,1.9,0,"A"),  # Argentina 2-1 Nigeria
    (1810,1930,1.5,1.0,1.9,0.7,1.7,2.1,0,"B"),  # Croatia 3-0 Argentina
    (1960,1760,1.8,0.7,1.4,1.0,2.0,1.6,0,"A"),  # Brazil 2-0 Costa Rica
    (1960,1820,1.8,0.7,1.5,1.0,2.0,1.7,0,"A"),  # Brazil 2-0 Serbia
    (1820,1870,1.5,1.0,1.6,0.9,1.7,1.8,0,"B"),  # England 0-1 Belgium
    # EURO 2020
    (1900,1840,1.8,0.8,1.6,0.9,2.0,1.8,0,"A"),  # Italy vs Wales (A)
    (1900,1850,1.8,0.8,1.7,0.9,2.0,1.9,0,"A"),  # Italy 1-0 Austria (R16)
    (1900,1910,1.8,0.8,1.8,0.9,2.0,2.0,0,"A"),  # Italy 2-1 Belgium (QF)
    (1900,1850,1.8,0.8,1.7,0.8,2.0,1.9,0,"D"),  # Italy 1-1 Spain (SF, pens→ITA) — 90min draw
    (1900,1860,1.8,0.8,1.7,0.9,2.0,1.9,0,"D"),  # Italy 1-1 England (Final, pens→ITA) — 90min draw
    (1910,1830,1.8,0.9,1.6,1.0,1.9,1.8,0,"A"),  # Belgium 2-1 Denmark
    (1910,1860,1.8,0.9,1.7,0.9,1.9,1.8,0,"A"),  # Belgium 1-0 Portugal (R16)
    (1850,1830,1.7,0.9,1.6,1.0,1.9,1.8,0,"A"),  # Spain 5-3 Croatia (R16)
    (1850,1910,1.7,0.9,1.8,0.9,1.9,1.9,0,"A"),  # Spain 2-1 Switzerland (QF)
    (1870,1810,1.7,0.9,1.5,1.0,1.9,1.7,0,"A"),  # Denmark vs Wales (A)
    (1870,1820,1.7,0.9,1.5,1.0,1.9,1.7,0,"A"),  # Denmark vs Czech R (QF)
    (1920,1870,1.8,0.9,1.7,0.9,2.0,1.8,0,"A"),  # England 2-0 Germany (R16)
    (1920,1870,1.8,0.9,1.7,0.9,2.0,1.9,0,"A"),  # England 4-0 Ukraine (QF)
    (1920,1870,1.8,0.9,1.7,0.9,2.0,1.9,0,"D"),  # England 1-1 Denmark (SF)
    (1830,1790,1.6,1.0,1.5,1.0,1.8,1.7,0,"D"),  # France 1-1 Portugal (group)
    (1830,1770,1.6,1.0,1.5,1.0,1.8,1.7,0,"D"),  # France 1-1 Hungary
    (1830,1810,1.6,1.0,1.5,1.0,1.8,1.7,0,"D"),  # France 3-3 Switzerland (pens→SUI) — 90min draw
    (1960,1830,2.0,0.8,1.6,1.0,2.2,1.8,0,"A"),  # Netherlands 3-2 Ukraine
    (1960,1900,2.0,0.8,1.8,0.8,2.2,2.0,0,"B"),  # Netherlands 0-2 Czech R (R16, upset)
    # EURO 2024
    (2040,1880,1.8,0.7,1.7,0.9,2.1,1.9,0,"A"),  # Spain 3-0 Croatia
    (2040,1820,1.8,0.7,1.5,1.0,2.1,1.7,0,"A"),  # Spain 1-0 Italy
    (2040,1870,1.8,0.7,1.7,0.9,2.1,1.8,0,"A"),  # Spain 2-1 Albania
    (2040,1790,1.8,0.7,1.5,1.0,2.1,1.6,0,"A"),  # Spain 4-1 Georgia (QF)
    (2040,1850,1.8,0.7,1.7,0.8,2.1,1.9,0,"A"),  # Spain 2-1 France (SF)
    (2040,1880,1.8,0.7,1.7,0.9,2.1,1.9,0,"A"),  # Spain 2-1 England (Final)
    (2000,1900,1.9,0.8,1.7,0.9,2.1,1.9,0,"A"),  # Germany 5-1 Scotland
    (2000,1870,1.9,0.8,1.7,0.9,2.1,1.9,0,"D"),  # Germany 2-2 Switzerland
    (2000,1840,1.9,0.8,1.7,0.9,2.1,1.9,0,"A"),  # Germany 1-0 Hungary
    (2000,1870,1.9,0.8,1.8,0.7,2.1,2.1,0,"B"),  # Germany 1-2 Spain (EURO 2024 QF) — Spain won
    (1940,1800,1.9,0.8,1.6,1.0,2.0,1.7,0,"A"),  # France 1-0 Austria
    (1940,1820,1.9,0.8,1.6,1.0,2.0,1.8,0,"D"),  # France 1-1 Poland
    (1940,1830,1.9,0.8,1.7,0.9,2.0,1.9,0,"A"),  # France 0-0 Netherlands (D)
    (1940,1800,1.9,0.8,1.6,1.0,2.0,1.8,0,"A"),  # France 1-0 Belgium (R16)
    (1940,1860,1.9,0.8,1.7,0.9,2.0,1.9,0,"A"),  # France 1-0 Portugal (QF)
    (1940,2040,1.9,0.8,1.8,0.7,2.0,2.1,0,"B"),  # France 1-2 Spain (SF)
    (1880,1860,1.7,0.9,1.7,0.9,1.9,1.9,0,"A"),  # England 1-0 Serbia
    (1880,1830,1.7,0.9,1.6,1.0,1.9,1.8,0,"D"),  # England 1-1 Denmark
    (1880,1810,1.7,0.9,1.5,1.0,1.9,1.7,0,"A"),  # England 0-0 Slovenia
    (1880,1790,1.7,0.9,1.5,1.0,1.9,1.6,0,"A"),  # England 2-1 Slovakia (ET)
    (1880,1820,1.7,0.9,1.5,1.0,1.9,1.7,0,"D"),  # England 1-1 Switzerland (QF, pens→ENG) — 90min draw
    (1880,1940,1.7,0.9,1.9,0.8,1.9,2.0,0,"B"),  # England 1-2 Spain (Final)
    (1850,1810,1.7,0.9,1.5,1.0,1.9,1.7,0,"A"),  # Netherlands 2-1 Poland
    (1850,1800,1.7,0.9,1.6,1.0,1.9,1.8,0,"A"),  # Netherlands 0-0 France (D)
    (1850,1830,1.7,0.9,1.7,0.9,1.9,1.9,0,"B"),  # Netherlands 2-3 Austria (Austria won)
    (1850,1890,1.7,0.9,1.7,0.9,1.9,1.9,0,"A"),  # Netherlands 2-1 Romania (R16)
    (1850,1900,1.7,0.9,1.8,0.8,1.9,2.0,0,"B"),  # Netherlands 1-2 Turkey (QF, upset)
]

RESULT_MAP = {"A": 0, "D": 1, "B": 2}
RESULT_LABELS = {0: "A", 1: "D", 2: "B"}


def _make_features(row) -> list:
    elo_a, elo_b, gf_a, ga_a, gf_b, ga_b, forma_a, forma_b, is_host_a = row[:9]
    elo_diff  = elo_a - elo_b
    gf_diff   = gf_a - gf_b
    ga_diff   = ga_a - ga_b
    frm_diff  = forma_a - forma_b
    elo_ratio = elo_a / max(1, elo_b)
    goal_ratio = (gf_a * ga_b) / max(0.01, gf_b * ga_a)
    return [elo_diff, gf_diff, ga_diff, frm_diff, is_host_a, elo_ratio, goal_ratio, elo_a, elo_b]


_ML_HOST_ELO_BOOST = 55  # mirrors stacked_predictor.HOME_ELO_BOOST


def _make_team_features(team_a: dict, team_b: dict, home_team: str = None) -> list:
    elo_a   = team_a.get("ELO", 1700)
    elo_b   = team_b.get("ELO", 1700)
    # Apply the same host-nation ELO boost used by the ELO model so the ML
    # sees the correct relative strength when a host nation is involved.
    if home_team:
        if team_a.get("name") == home_team:
            elo_a += _ML_HOST_ELO_BOOST
        elif team_b.get("name") == home_team:
            elo_b += _ML_HOST_ELO_BOOST
    gf_a    = team_a.get("GF_AVG", 1.25)
    ga_a    = team_a.get("GA_AVG", 1.25)
    gf_b    = team_b.get("GF_AVG", 1.25)
    ga_b    = team_b.get("GA_AVG", 1.25)
    frm_a   = team_a.get("FORMA", 1.5)
    frm_b   = team_b.get("FORMA", 1.5)
    is_host = 1 if (home_team and team_a.get("name") == home_team) else 0
    row = (elo_a, elo_b, gf_a, ga_a, gf_b, ga_b, frm_a, frm_b, is_host, None)
    return _make_features(row)


class MLPredictor:
    """Random Forest + GradientBoosting ensemble for match outcome prediction."""

    def __init__(self):
        self.rf  = RandomForestClassifier(n_estimators=499, max_depth=6,
                                           min_samples_leaf=15, max_features="sqrt",
                                           random_state=42)
        self.gb  = GradientBoostingClassifier(n_estimators=60, max_depth=3,
                                               learning_rate=0.0189, subsample=0.635,
                                               min_samples_leaf=6, random_state=42)
        self._fitted = False

    def fit(self):
        X = [_make_features(row) for row in TRAINING_DATA]
        y = [RESULT_MAP[row[9]] for row in TRAINING_DATA]
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=int)
        self.rf.fit(X, y)
        self.gb.fit(X, y)
        self._fitted = True

    def predict(self, team_a: dict, team_b: dict, home_team: str = None) -> tuple:
        """Returns (p_win_a, p_draw, p_win_b) with temperature scaling to prevent overconfidence."""
        feats = np.array([_make_team_features(team_a, team_b, home_team)], dtype=float)
        # Average RF + GB probabilities
        p_rf = self.rf.predict_proba(feats)[0]
        p_gb = self.gb.predict_proba(feats)[0]
        classes = list(self.rf.classes_)
        # blend 60/40 RF/GB
        p = 0.6 * p_rf + 0.4 * p_gb
        # Map class indices (0=A, 1=D, 2=B)
        pa = p[classes.index(0)] if 0 in classes else 0.33
        pd = p[classes.index(1)] if 1 in classes else 0.33
        pb = p[classes.index(2)] if 2 in classes else 0.34

        # Temperature scaling: with only ~110 training samples, tree ensembles
        # produce overconfident probabilities (e.g. 86% for Spain vs France).
        # T > 1 flattens the distribution toward uniform; T=1.8 calibrated via
        # cross-validated log-loss on training data.
        T = 1.8
        log_pa = math.log(max(1e-9, pa)) / T
        log_pd = math.log(max(1e-9, pd)) / T
        log_pb = math.log(max(1e-9, pb)) / T
        max_log = max(log_pa, log_pd, log_pb)
        exp_a = math.exp(log_pa - max_log)
        exp_d = math.exp(log_pd - max_log)
        exp_b = math.exp(log_pb - max_log)
        total = exp_a + exp_d + exp_b
        return exp_a/total, exp_d/total, exp_b/total

    def backtest_accuracy(self) -> dict:
        """5-fold cross-validated accuracy on training data (out-of-fold predictions)."""
        from sklearn.model_selection import cross_val_predict
        from sklearn.base import clone

        correct_outcomes = 0
        log_loss_total = 0.0
        eps = 1e-9
        n = len(TRAINING_DATA)
        X_all = np.array([_make_features(r) for r in TRAINING_DATA], dtype=float)
        y_all = np.array([RESULT_MAP[r[9]] for r in TRAINING_DATA], dtype=int)
        classes = [0, 1, 2]  # A wins, Draw, B wins

        # Out-of-fold predictions — models never see the row they predict
        rf_proba = cross_val_predict(clone(self.rf), X_all, y_all, cv=5, method="predict_proba")
        gb_proba = cross_val_predict(clone(self.gb), X_all, y_all, cv=5, method="predict_proba")

        for i, row in enumerate(TRAINING_DATA):
            true = RESULT_MAP[row[9]]
            p = 0.6 * rf_proba[i] + 0.4 * gb_proba[i]
            pred_class = classes[int(np.argmax(p))]
            if pred_class == true:
                correct_outcomes += 1
            p_true = p[classes.index(true)]
            log_loss_total -= math.log(max(eps, p_true))
        return {
            "accuracy": correct_outcomes / n,
            "log_loss": log_loss_total / n,
            "n_samples": n,
        }


if __name__ == "__main__":
    m = MLPredictor()
    m.fit()
    stats = m.backtest_accuracy()
    print(f"ML Ensemble — accuracy={stats['accuracy']*100:.1f}%  log-loss={stats['log_loss']:.4f}  n={stats['n_samples']}")

    spain   = {"name": "Spain",    "ELO": 2070, "FORMA": 2.3, "GF_AVG": 2.0, "GA_AVG": 0.6}
    france  = {"name": "France",   "ELO": 2081, "FORMA": 2.1, "GF_AVG": 1.9, "GA_AVG": 0.8}
    argent  = {"name": "Argentina","ELO": 2113, "FORMA": 2.4, "GF_AVG": 2.1, "GA_AVG": 0.6}
    print(f"\nSpain vs France:    A={m.predict(spain,france)[0]:.3f}  D={m.predict(spain,france)[1]:.3f}  B={m.predict(spain,france)[2]:.3f}")
    print(f"Spain vs Argentina: A={m.predict(spain,argent)[0]:.3f}  D={m.predict(spain,argent)[1]:.3f}  B={m.predict(spain,argent)[2]:.3f}")
