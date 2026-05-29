"""
Modelo de predicción apilado (model stacking) para el Mundial 2026.

Combina tres modelos independientes:
  1. ELO       — probabilidad basada en rating ELO (modelo clásico)
  2. Poisson   — goles esperados vía Dixon-Coles; convierte a 1X2
  3. ML        — Random Forest + GradientBoosting ensemble (ml_predictor.py)

Más H2H histórico cuando hay datos disponibles.

Variables simplificadas: se eliminaron WC_TOURNAMENT_FACTOR y SQUAD_AVG_AGE
(alta correlación con ELO, aportaban ruido). El efecto de lesiones y profundidad
de plantilla se unifica en un único «availability score» (0–1).
"""

import math
import json
import os

# ─────────────────────────────────────────────
#  Constantes del modelo
# ─────────────────────────────────────────────
WC_AVG_GOALS = 1.30          # Promedio de goles por equipo por partido en el Mundial (WC 2018: 1.32, WC 2022: 1.34)
DIXON_COLES_RHO = -0.12      # Dixon-Coles correlation parameter for low-score outcomes
H2H_PATH = os.path.join(os.path.dirname(__file__), "data", "h2h_data.json")

from predictor import VENUES, HIGH_ALTITUDE_TEAMS

# Typical summer competitive temperature (°C) each squad is acclimatised to.
TEAM_COMFORT_TEMP = {
    # Hot-climate nations — comfortable in 26-34 °C
    "Brazil":        28, "Colombia":   27, "Ecuador":    22,
    "Mexico":        26, "Senegal":    30, "Nigeria":    30,
    "Cameroon":      28, "Ghana":      30, "Ivory Coast":30,
    "Costa Rica":    27, "Panama":     30, "Jamaica":    28,
    "Honduras":      27, "Morocco":    24, "Algeria":    28,
    "Tunisia":       26, "Egypt":      30, "Saudi Arabia":34,
    "Iran":          28, "DR Congo":   28, "Mali":       32,
    "Cape Verde":    27, "Curaçao":    28,
    # Temperate / mixed — comfortable in 18-24 °C
    "Argentina":     20, "Uruguay":    20, "Chile":      18,
    "Peru":          20, "USA":        22, "Canada":     18,
    "Australia":     22, "Japan":      22, "South Korea":22,
    "Turkey":        23, "Portugal":   20, "Spain":      22,
    "Italy":         22, "Croatia":    21, "Serbia":     20,
    "Bosnia & Herzegovina": 20, "Albania": 22,
    # Cold-climate nations — comfortable in 10-18 °C
    "Germany":       16, "France":     17, "England":    14,
    "Netherlands":   15, "Belgium":    15, "Switzerland":16,
    "Austria":       16, "Poland":     17, "Czech Republic":16,
    "Norway":        11, "Sweden":     14, "Scotland":   12,
    "Denmark":       14, "Slovakia":   16, "Hungary":    18,
    "Iceland":        9,
}

def climate_altitude_adj(team_name: str, venue_name: str) -> float:
    """
    Returns a goal-output multiplier based on how much the venue's climate
    and altitude deviate from what the team is acclimatised to.
    """
    if not venue_name:
        return 1.0
    venue_data = VENUES.get(venue_name)
    if not venue_data:
        return 1.0

    mult = 1.0
    alt      = venue_data["alt"]
    temp     = venue_data["temp"]
    humidity = venue_data["humidity"]

    # ── Altitude ────────────────────────────────────────────────────────────
    if alt > 500 and team_name not in HIGH_ALTITUDE_TEAMS:
        alt_penalty = min(0.17, (alt - 500) / 11_000)
        mult -= alt_penalty

    # ── Temperature deviation from comfort zone ──────────────────────────────
    comfort = TEAM_COMFORT_TEMP.get(team_name, 20)
    delta   = temp - comfort

    if delta > 0:
        humidity_amp = 1.0 + max(0, (humidity - 50) / 100)
        heat_penalty = min(0.12, delta * 0.005 * humidity_amp)
        mult -= heat_penalty
    elif delta < -4:
        cold_penalty = min(0.06, abs(delta + 4) * 0.003)
        mult -= cold_penalty

    return max(0.72, mult)

# ─────────────────────────────────────────────
#  Model weights  (calibrated on WC 2018 + 2022 + EURO 2020/2024)
# ─────────────────────────────────────────────
DEFAULT_WEIGHTS = {
    "elo":     0.10,   # base ELO signal
    "poisson": 0.10,   # Dixon-Coles physics model
    "ml":      0.80,   # Random Forest + GradBoost ensemble (optimized)
}

# ─────────────────────────────────────────────
#  Host nation advantage
# ─────────────────────────────────────────────
HOST_NATIONS = {"USA", "Mexico", "Canada"}
HOME_ELO_BOOST = 55       # Added to ELO when team is host playing in tournament
HOME_GOALS_BOOST = 0.10   # Multiplier to lam when home team

# ─────────────────────────────────────────────
#  Squad availability score
#  Replaces the old SQUAD_DEPTH + INJURIES combo.
#  availability = depth * (1 - injury_load)
# ─────────────────────────────────────────────
SQUAD_DEPTH = {
    "Spain":        0.97,  # Euro 2024 champions — best squad in the world
    "France":       0.96,  # Elite depth, Nations League 2024 champions
    "England":      0.92,
    "Germany":      0.90,
    "Brazil":       0.89,
    "Portugal":     0.87,
    "Netherlands":  0.86,
    "Belgium":      0.85,
    "Argentina":    0.82,  # Aging core, post-peak transition
    "Italy":        0.83,
    "Croatia":      0.77,
    "Uruguay":      0.76,
    "Colombia":     0.75,
    "Japan":        0.74,
    "South Korea":  0.73,
    "USA":          0.72,
    "Mexico":       0.71,
    "Switzerland":  0.73,
    "Turkey":       0.70,
    "Senegal":      0.67,
    "Norway":       0.77,
    "Sweden":       0.71,
    "Scotland":     0.69,
    "Czech Republic": 0.67,
    "Morocco":      0.71,
}  # default 0.60


def availability_score(team_name: str, injuries: int, depth: float | None = None) -> float:
    """
    Returns a squad availability multiplier in [0.70, 1.0].
    injuries: number of key players absent (0–4+)
    depth: optional override; uses SQUAD_DEPTH lookup if None
    """
    d = depth if depth is not None else SQUAD_DEPTH.get(team_name, 0.60)
    # Each injury reduces output proportionally to how thin the bench is
    injury_load = injuries * 0.045 * (1.0 + (1.0 - d) * 0.4)
    return max(0.70, 1.0 - injury_load)


# ─────────────────────────────────────────────
#  Penalty win rates
# ─────────────────────────────────────────────
PENALTY_WIN_RATES = {
    "Germany":     0.76, "Argentina":   0.60, "France":      0.68,
    "Brazil":      0.48, "Spain":       0.58, "Portugal":    0.56,
    "Italy":       0.54, "Croatia":     0.62, "Netherlands": 0.32,
    "England":     0.42, "Uruguay":     0.55, "Switzerland": 0.52,
    "Mexico":      0.38, "Colombia":    0.46, "USA":         0.54,
    "Japan":       0.50, "South Korea": 0.42, "Senegal":     0.46,
    "Belgium":     0.50, "Poland":      0.46, "Norway":      0.58,
    "Sweden":      0.52, "Scotland":    0.48, "Czech Republic": 0.50,
    "Morocco":     0.50,
}  # default 0.50


def penalty_win_prob(name_a: str, name_b: str) -> float:
    """Returns P(team A wins on penalties). Normalized."""
    rate_a = PENALTY_WIN_RATES.get(name_a, 0.50)
    rate_b = PENALTY_WIN_RATES.get(name_b, 0.50)
    return rate_a / (rate_a + rate_b)


# ─────────────────────────────────────────────
#  Utilidades Poisson
# ─────────────────────────────────────────────
def _poisson_pmf(lam: float, k: int) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * (lam ** k) / math.factorial(k)

def _poisson_3way(lam_a: float, lam_b: float, max_goals: int = 9):
    """
    Calcula P(gana A), P(empate), P(gana B) usando distribución de Poisson
    con corrección Dixon-Coles para resultados de bajo marcador.
    """
    rho = DIXON_COLES_RHO

    p_win_a = p_draw = p_win_b = 0.0
    for i in range(max_goals + 1):
        pa_i = _poisson_pmf(lam_a, i)
        for j in range(max_goals + 1):
            p = pa_i * _poisson_pmf(lam_b, j)

            if i == 0 and j == 0:
                tau = 1.0 - lam_a * lam_b * rho
            elif i == 1 and j == 0:
                tau = 1.0 + lam_b * rho
            elif i == 0 and j == 1:
                tau = 1.0 + lam_a * rho
            elif i == 1 and j == 1:
                tau = 1.0 - rho
            else:
                tau = 1.0

            tau = max(0.01, tau)
            p *= tau

            if i > j:
                p_win_a += p
            elif i == j:
                p_draw += p
            else:
                p_win_b += p

    total = p_win_a + p_draw + p_win_b
    if total <= 0:
        total = 1.0
    return p_win_a / total, p_draw / total, p_win_b / total


# ─────────────────────────────────────────────
#  Modelo 1 — ELO
# ─────────────────────────────────────────────
def elo_model(team_a: dict, team_b: dict, home_team: str = None, venue_name: str = None) -> tuple:
    """Probabilidades basadas en diferencia de ELO con draw dinámico."""
    elo_a = team_a["ELO"]
    elo_b = team_b["ELO"]

    name_a = team_a.get("name", "")
    name_b = team_b.get("name", "")

    # Host nation ELO boost — only applies if the designated home_team
    # is actually playing (not both co-hosts simultaneously)
    if home_team is not None and home_team in HOST_NATIONS:
        if name_a == home_team:
            elo_a += HOME_ELO_BOOST
        elif name_b == home_team:
            elo_b += HOME_ELO_BOOST

    # Climate / Altitude — calibrated so a 10% climate penalty ≈ 28 ELO drop
    elo_a += (climate_altitude_adj(name_a, venue_name) - 1.0) * 280
    elo_b += (climate_altitude_adj(name_b, venue_name) - 1.0) * 280

    # Squad availability — injuries depress effective ELO
    # A full-strength team (avail=1.0) gets no penalty; 3 injuries ≈ -40 to -60 ELO
    avail_a = availability_score(name_a, team_a.get("INJURIES", 0))
    avail_b = availability_score(name_b, team_b.get("INJURIES", 0))
    elo_a += (avail_a - 1.0) * 400
    elo_b += (avail_b - 1.0) * 400

    # Extra multiplier (manual override for external intel)
    elo_a += (team_a.get("EXTRA", 1.0) - 1.0) * 800
    elo_b += (team_b.get("EXTRA", 1.0) - 1.0) * 800

    elo_diff = elo_a - elo_b
    p_elo_a = 1 / (1 + math.pow(10, -elo_diff / 400))
    p_elo_b = 1 - p_elo_a
    # Draw shrinks as mismatch grows; floor 15% consistent with historical WC rates (~16-25%).
    draw = max(0.15, 0.25 - 0.12 * min(1.0, abs(elo_diff) / 400))
    return p_elo_a * (1 - draw), draw, p_elo_b * (1 - draw)


# ─────────────────────────────────────────────
#  Modelo 2 — Poisson (Dixon-Coles style)
# ─────────────────────────────────────────────
def _compute_lambdas(
    team_a: dict,
    team_b: dict,
    home_team: str = None,
    round_number: int = 1,
    venue_name: str = None,
) -> tuple:
    """Shared λ computation used by both poisson_model and expected_goals."""
    mu = WC_AVG_GOALS
    name_a = team_a.get("name", "")
    name_b = team_b.get("name", "")

    gf_a = max(0.3, team_a.get("GF_AVG", mu))
    ga_a = max(0.3, team_a.get("GA_AVG", mu))
    gf_b = max(0.3, team_b.get("GF_AVG", mu))
    ga_b = max(0.3, team_b.get("GA_AVG", mu))

    lam_a = (gf_a / mu) * (ga_b / mu) * mu
    lam_b = (gf_b / mu) * (ga_a / mu) * mu

    # FORMA adjustment: scale goal output by form relative to baseline (1.5)
    # A team with FORMA 2.5 (exceptional) gets ~7% boost; FORMA 1.0 (poor) gets ~3% cut
    forma_baseline = 1.5
    forma_a = team_a.get("FORMA", forma_baseline)
    forma_b = team_b.get("FORMA", forma_baseline)
    lam_a *= 1.0 + (forma_a - forma_baseline) * 0.07
    lam_b *= 1.0 + (forma_b - forma_baseline) * 0.07

    # Host nation goals boost — only the designated home_team gets it
    if home_team is not None and home_team in HOST_NATIONS:
        if name_a == home_team:
            lam_a *= (1.0 + HOME_GOALS_BOOST)
        elif name_b == home_team:
            lam_b *= (1.0 + HOME_GOALS_BOOST)

    # Climate and altitude penalties
    lam_a *= climate_altitude_adj(name_a, venue_name)
    lam_b *= climate_altitude_adj(name_b, venue_name)

    # Extra info multiplier
    lam_a *= team_a.get("EXTRA", 1.0)
    lam_b *= team_b.get("EXTRA", 1.0)

    # Unified availability (squad depth + injuries)
    avail_a = availability_score(name_a, team_a.get("INJURIES", 0))
    avail_b = availability_score(name_b, team_b.get("INJURIES", 0))
    
    # Injuries hurt attack (own lambda drops) AND defense (opponent lambda increases)
    lam_a = lam_a * (avail_a / avail_b)
    lam_b = lam_b * (avail_b / avail_a)

    lam_a = max(0.15, min(5.0, lam_a))
    lam_b = max(0.15, min(5.0, lam_b))
    return lam_a, lam_b


def poisson_model(
    team_a: dict,
    team_b: dict,
    home_team: str = None,
    round_number: int = 1,
    venue_name: str = None,
) -> tuple:
    """Calcula P(gana A), P(empate), P(gana B) usando Dixon-Coles."""
    lam_a, lam_b = _compute_lambdas(team_a, team_b, home_team, round_number, venue_name)
    return _poisson_3way(lam_a, lam_b)


def expected_goals(
    team_a: dict,
    team_b: dict,
    home_team: str = None,
    round_number: int = 1,
    venue_name: str = None,
) -> tuple:
    """Devuelve (λ_A, λ_B) — goles esperados por equipo."""
    return _compute_lambdas(team_a, team_b, home_team, round_number, venue_name)



# ─────────────────────────────────────────────
#  Modelo 4 — H2H histórico
# ─────────────────────────────────────────────
_h2h_db = None

def _load_h2h():
    global _h2h_db
    if _h2h_db is None:
        try:
            with open(H2H_PATH, "r") as f:
                _h2h_db = json.load(f)
        except FileNotFoundError:
            _h2h_db = {}
    return _h2h_db

def h2h_model(name_a: str, name_b: str) -> tuple | None:
    """Retorna (p_win_a, p_draw, p_win_b) desde datos históricos, o None si no hay."""
    db = _load_h2h()
    key1 = f"{name_a} vs {name_b}"
    key2 = f"{name_b} vs {name_a}"

    if key1 in db:
        wr_a = db[key1]
    elif key2 in db:
        wr_a = 1 - db[key2]
    else:
        return None

    # wr_a is points percentage: P(Win A) + 0.5 * P(Draw) = wr_a
    draw = min(0.30, 0.50 - abs(wr_a - 0.5) * 0.80)
    
    p_win_a = wr_a - 0.5 * draw
    p_win_b = 1.0 - wr_a - 0.5 * draw
    
    # Safety bounds
    p_win_a = max(0.0, p_win_a)
    p_win_b = max(0.0, p_win_b)
    
    total = p_win_a + draw + p_win_b
    if total <= 0:
        return 0.33, 0.34, 0.33
    return p_win_a / total, draw / total, p_win_b / total


# ─────────────────────────────────────────────
#  Modelo 5 — ML Ensemble  (lazy import)
# ─────────────────────────────────────────────
_ml_predictor = None

def _get_ml_predictor():
    global _ml_predictor
    if _ml_predictor is None:
        try:
            from ml_predictor import MLPredictor
            _ml_predictor = MLPredictor()
            _ml_predictor.fit()
        except Exception:
            _ml_predictor = False   # mark as unavailable
    return _ml_predictor if _ml_predictor is not False else None



# ─────────────────────────────────────────────
#  Meta-learner — combinación ponderada de todos los modelos
# ─────────────────────────────────────────────
_PREDICTION_CACHE = {}
_ML_CACHE = {}
_CACHE_MAX = 4096  # bounded to prevent unbounded growth during large MC runs

def stack_predict(
    team_a: dict,
    team_b: dict,
    weights: dict = None,
    home_team: str = None,
    round_number: int = 1,
    venue_name: str = None,
) -> dict:
    """
    Predice probabilidades 1X2 combinando todos los modelos disponibles.
    Usa una caché interna para acelerar millones de llamadas en Monte Carlo.
    """
    # Generar clave para la caché
    _use_default_weights = weights is None
    cache_key = (
        team_a.get("name"), team_b.get("name"),
        int(team_a.get("ELO", 1600)), int(team_b.get("ELO", 1600)),
        round(team_a.get("FORMA", 1.0), 2), round(team_b.get("FORMA", 1.0), 2),
        round(team_a.get("GF_AVG", 1.0), 2), round(team_b.get("GF_AVG", 1.0), 2),
        round(team_a.get("GA_AVG", 1.0), 2), round(team_b.get("GA_AVG", 1.0), 2),
        team_a.get("INJURIES", 0), team_b.get("INJURIES", 0),
        home_team, round_number, venue_name
    )
    if _use_default_weights and cache_key in _PREDICTION_CACHE:
        return _PREDICTION_CACHE[cache_key]

    if _use_default_weights:
        weights = dict(DEFAULT_WEIGHTS)

    elo_p = elo_model(team_a, team_b, home_team=home_team, venue_name=venue_name)
    poi_p = poisson_model(team_a, team_b, home_team=home_team, round_number=round_number, venue_name=venue_name)

    models = {
        "elo":     elo_p,
        "poisson": poi_p,
    }

    active_w = {
        "elo":     weights.get("elo",     0.35),
        "poisson": weights.get("poisson", 0.30),
    }

    # H2H model — add if data available
    name_a = team_a.get("name", "")
    name_b = team_b.get("name", "")
    h2h_p = h2h_model(name_a, name_b)
    if h2h_p is not None:
        models["h2h"] = h2h_p
        h2h_share = 0.08  # small weight — historical H2H is noisy for national teams
        other_total = sum(active_w.values())
        if other_total > 0:
            scale = (1.0 - h2h_share) / other_total
            for k in active_w:
                active_w[k] *= scale
        active_w["h2h"] = h2h_share

    # ML model — add if available
    ml = _get_ml_predictor()
    if ml is not None:
        try:
            ml_key = (team_a.get("name"), team_b.get("name"), home_team)
            if ml_key in _ML_CACHE:
                ml_p = _ML_CACHE[ml_key]
            else:
                ml_p = ml.predict(team_a, team_b, home_team=home_team)
                if len(_ML_CACHE) < _CACHE_MAX:
                    _ML_CACHE[ml_key] = ml_p
            
            models["ml"] = ml_p
            ml_share = weights.get("ml", 0.35)
            other_total = sum(active_w.values())
            if other_total > 0:
                scale = (1.0 - ml_share) / other_total
                for k in active_w:
                    active_w[k] *= scale
            active_w["ml"] = ml_share
        except Exception:
            pass

    total_w = sum(active_w.values())

    pa = sum(active_w[k] * models[k][0] for k in models) / total_w
    pd = sum(active_w[k] * models[k][1] for k in models) / total_w
    pb = sum(active_w[k] * models[k][2] for k in models) / total_w

    # Renormalise
    total = pa + pd + pb
    if total > 0:
        pa, pd, pb = pa / total, pd / total, pb / total
    else:
        pa, pd, pb = 0.333, 0.334, 0.333

    breakdown = {}
    labels = {"elo": "ELO", "poisson": "Poisson", "ml": "ML", "h2h": "H2H"}
    for k, probs in models.items():
        breakdown[labels[k]] = {
            "win_a":  round(probs[0], 3),
            "draw":   round(probs[1], 3),
            "win_b":  round(probs[2], 3),
            "weight": round(active_w.get(k, 0), 3),
        }

    result = {
        "p_win_a":        round(pa, 4),
        "p_draw":         round(pd, 4),
        "p_win_b":        round(pb, 4),
        "model_breakdown": breakdown,
        "weights_used":   {k: round(v / total_w, 3) for k, v in active_w.items()},
    }
    
    # Save to cache if no custom weights were passed in
    if _use_default_weights and len(_PREDICTION_CACHE) < _CACHE_MAX:
        _PREDICTION_CACHE[cache_key] = result

    return result


# ─────────────────────────────────────────────
#  Test rápido
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import json as _json

    spain   = {"name": "Spain",   "ELO": 2165, "FORMA": 2.5, "GF_AVG": 2.2, "GA_AVG": 0.5, "INJURIES": 0}
    france  = {"name": "France",  "ELO": 2095, "FORMA": 2.3, "GF_AVG": 2.1, "GA_AVG": 0.7, "INJURIES": 0}
    argent  = {"name": "Argentina","ELO": 2095, "FORMA": 2.2, "GF_AVG": 1.9, "GA_AVG": 0.7, "INJURIES": 0}

    print("=== Spain vs France ===")
    result = stack_predict(spain, france)
    print(_json.dumps(result, indent=2))

    print("\n=== Argentina vs Spain ===")
    result2 = stack_predict(argent, spain)
    print(_json.dumps(result2, indent=2))

    print("\n=== USA (host) vs England ===")
    usa     = {"name": "USA",     "ELO": 1721, "FORMA": 1.5, "GF_AVG": 1.4, "GA_AVG": 1.1, "INJURIES": 0}
    england = {"name": "England", "ELO": 2021, "FORMA": 1.9, "GF_AVG": 1.7, "GA_AVG": 0.9, "INJURIES": 0}
    r2 = stack_predict(usa, england, home_team="USA")
    print(_json.dumps(r2, indent=2))
