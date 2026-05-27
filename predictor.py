import math
import json
import os
from dotenv import load_dotenv

load_dotenv()

ODDS_API_KEY = os.getenv("ODDS_API_KEY")

VENUES = {
    "Miami": {"country": "USA", "temp": 30, "humidity": 82, "alt": 3},
    "Dallas": {"country": "USA", "temp": 34, "humidity": 55, "alt": 165},
    "Houston": {"country": "USA", "temp": 32, "humidity": 75, "alt": 12},
    "Atlanta": {"country": "USA", "temp": 29, "humidity": 66, "alt": 320},
    "Los Angeles": {"country": "USA", "temp": 23, "humidity": 62, "alt": 88},
    "San Francisco": {"country": "USA", "temp": 19, "humidity": 72, "alt": 18},
    "Seattle": {"country": "USA", "temp": 19, "humidity": 68, "alt": 54},
    "Kansas City": {"country": "USA", "temp": 30, "humidity": 67, "alt": 265},
    "New York / NJ": {"country": "USA", "temp": 26, "humidity": 65, "alt": 8},
    "Boston": {"country": "USA", "temp": 24, "humidity": 66, "alt": 8},
    "Philadelphia": {"country": "USA", "temp": 27, "humidity": 65, "alt": 12},
    "Vancouver": {"country": "Canada", "temp": 19, "humidity": 72, "alt": 5},
    "Toronto": {"country": "Canada", "temp": 23, "humidity": 64, "alt": 100},
    "Ciudad de México": {"country": "Mexico", "temp": 19, "humidity": 52, "alt": 2240},
    "Monterrey": {"country": "Mexico", "temp": 32, "humidity": 58, "alt": 540},
    "Guadalajara": {"country": "Mexico", "temp": 26, "humidity": 62, "alt": 1566}
}

# Equipos de gran altitud nativa (aproximados)
HIGH_ALTITUDE_TEAMS = ["Mexico", "Bolivia", "Ecuador", "Colombia", "Peru"]

# Typical summer competitive temperature (°C) each squad is acclimatised to.
# Mirrors TEAM_COMFORT_TEMP in stacked_predictor.py — keep in sync.
_COMFORT_TEMP = {
    "Brazil": 28, "Colombia": 27, "Ecuador": 22, "Mexico": 26,
    "Senegal": 30, "Nigeria": 30, "Cameroon": 28, "Ghana": 30,
    "Ivory Coast": 30, "Costa Rica": 27, "Panama": 30, "Jamaica": 28,
    "Honduras": 27, "Morocco": 24, "Algeria": 28, "Tunisia": 26,
    "Egypt": 30, "Saudi Arabia": 34, "Iran": 28,
    "Argentina": 20, "Uruguay": 20, "Chile": 18, "Peru": 20,
    "USA": 22, "Canada": 18, "Japan": 22, "South Korea": 22,
    "Turkey": 23, "Portugal": 20, "Spain": 22, "Italy": 22,
    "Croatia": 21, "Serbia": 20,
    "Germany": 16, "France": 17, "England": 14, "Netherlands": 15,
    "Belgium": 15, "Switzerland": 16, "Austria": 16, "Poland": 17,
    "Czech Republic": 16, "Norway": 11, "Sweden": 14, "Scotland": 12,
    "Denmark": 14,
}

def calculate_situational_adj(team_name, venue_name, injuries_count, is_diaspora=False):
    adj = 0.0

    venue_data = VENUES.get(venue_name)
    if not venue_data:
        return 0.0

    # Home crowd / diaspora advantage
    if venue_data["country"] == team_name:
        adj += 0.05
    elif is_diaspora:
        adj += 0.02

    alt      = venue_data["alt"]
    temp     = venue_data["temp"]
    humidity = venue_data["humidity"]

    # Altitude — continuous penalty from 500 m upward
    if alt > 500 and team_name not in HIGH_ALTITUDE_TEAMS:
        adj -= min(0.08, (alt - 500) / 11_000 * 2.0)

    # Temperature deviation from each team's comfort zone
    comfort = _COMFORT_TEMP.get(team_name, 20)
    delta   = temp - comfort

    if delta > 0:
        humidity_amp = 1.0 + max(0, (humidity - 50) / 100)
        adj -= min(0.07, delta * 0.004 * humidity_amp)
    elif delta < -4:
        adj -= min(0.03, abs(delta + 4) * 0.002)

    # Injuries
    injury_penalty = min(0.06, injuries_count * 0.02)
    adj -= injury_penalty

    return adj


def calculate_predictions(match_data):
    elo_a = match_data["ELO_A"]
    elo_b = match_data["ELO_B"]
    
    # PASO 1 - Probabilidad base por Elo con draw dinámico
    elo_diff = elo_a - elo_b
    p_elo_a = 1 / (1 + math.pow(10, -elo_diff / 400))
    p_elo_b = 1 - p_elo_a

    # Draw probability shrinks as ELO gap grows (large mismatches produce fewer draws).
    # At diff=0: ~26% draw. At diff=400: ~18% draw. Floor at 15%.
    draw_base = max(0.15, 0.26 - 0.08 * min(1.0, abs(elo_diff) / 400))
    p_win_a = p_elo_a * (1 - draw_base)
    p_win_b = p_elo_b * (1 - draw_base)
    p_draw = draw_base
    # By construction: p_win_a + p_win_b + p_draw == 1 and all values > 0

    # PASO 2 - Ajuste por forma y ataque
    form_adj = (match_data.get("FORMA_A", 0) - match_data.get("FORMA_B", 0)) * 0.04
    attack_adj = ((match_data.get("GF_AVG_A", 0) - match_data.get("GA_AVG_A", 0)) - (match_data.get("GF_AVG_B", 0) - match_data.get("GA_AVG_B", 0))) * 0.02
    
    h2h_win_rate = match_data.get("H2H_WIN_RATE_A", None)
    h2h_adj = (h2h_win_rate - 0.5) * 0.03 if h2h_win_rate is not None else 0
        
    # PASO 2B - Ajustes Situacionales
    venue = match_data.get("VENUE", "")
    sit_adj_a = calculate_situational_adj(match_data["EQUIPO_A"], venue, match_data.get("INJURIES_A", 0), match_data.get("DIASPORA_A", False))
    sit_adj_b = calculate_situational_adj(match_data["EQUIPO_B"], venue, match_data.get("INJURIES_B", 0), match_data.get("DIASPORA_B", False))
    
    delta = form_adj + attack_adj + h2h_adj + (sit_adj_a - sit_adj_b)

    # Apply delta symmetrically to both teams, then clamp to avoid negatives
    p_win_a_adj = max(0.04, p_win_a + delta)
    p_win_b_adj = max(0.04, p_win_b - delta)

    # Renormalize so the three outcomes still sum to 1
    total_adj = p_win_a_adj + p_win_b_adj + p_draw
    p_win_a_norm = p_win_a_adj / total_adj
    p_win_b_norm = p_win_b_adj / total_adj
    p_draw_norm = p_draw / total_adj

    # PASO 3 - Suavizar con señal ELO pura (80/20 blend)
    p_final_a = 0.80 * p_win_a_norm + 0.20 * p_elo_a
    p_final_draw = 0.80 * p_draw_norm + 0.20 * draw_base
    p_final_b = 1 - p_final_a - p_final_draw

    # Safety floor and re-normalize (guards against extreme edge cases)
    p_final_a = max(0.02, p_final_a)
    p_final_b = max(0.02, p_final_b)
    p_final_draw = max(0.02, p_final_draw)
    total_final = p_final_a + p_final_b + p_final_draw
    p_final_a /= total_final
    p_final_b /= total_final
    p_final_draw /= total_final
    
    # PASO 4 - Odds justas
    odd_justo_a = 1 / p_final_a
    odd_justo_draw = 1 / p_final_draw
    odd_justo_b = 1 / p_final_b
    
    # PASO 5 - Margen de la casa
    odd_bookie_a = match_data["ODDS_BOOKIE_A"]
    odd_bookie_draw = match_data["ODDS_BOOKIE_DRAW"]
    odd_bookie_b = match_data["ODDS_BOOKIE_B"]
    
    margin = (1/odd_bookie_a + 1/odd_bookie_draw + 1/odd_bookie_b) - 1
    p_impl_a = (1/odd_bookie_a) / (1 + margin)
    p_impl_draw = (1/odd_bookie_draw) / (1 + margin)
    p_impl_b = (1/odd_bookie_b) / (1 + margin)
    
    # PASO 6 - Expected Value
    ev_a = (p_final_a * odd_bookie_a) - 1
    ev_draw = (p_final_draw * odd_bookie_draw) - 1
    ev_b = (p_final_b * odd_bookie_b) - 1
    
    value_bets = []
    if ev_a > 0.05 and p_final_a > p_impl_a + 0.03: value_bets.append("team_a")
    if ev_draw > 0.05 and p_final_draw > p_impl_draw + 0.03: value_bets.append("draw")
    if ev_b > 0.05 and p_final_b > p_impl_b + 0.03: value_bets.append("team_b")
        
    # PASO 7 - Kelly
    kelly_a = min(0.05, max(0, (p_final_a * odd_bookie_a - 1) / (odd_bookie_a - 1)) / 4)
    kelly_draw = min(0.05, max(0, (p_final_draw * odd_bookie_draw - 1) / (odd_bookie_draw - 1)) / 4)
    kelly_b = min(0.05, max(0, (p_final_b * odd_bookie_b - 1) / (odd_bookie_b - 1)) / 4)
    
    if abs(elo_diff) >= 200:
        confidence = "high"
    elif abs(elo_diff) >= 50:
        confidence = "medium"
    else:
        confidence = "low"
        
    summary = f"Partido en {venue}. "
    if sit_adj_a > sit_adj_b + 0.02:
        summary += f"Ventaja situacional para {match_data['EQUIPO_A']}. "
    elif sit_adj_b > sit_adj_a + 0.02:
        summary += f"Ventaja situacional para {match_data['EQUIPO_B']}. "
        
    if len(value_bets) > 0:
        summary += f"Valor detectado en {', '.join(value_bets)}."
    else:
        summary += "Sin value bets claras (>5% EV)."
        
    return {
        "match": f"{match_data['EQUIPO_A']} vs {match_data['EQUIPO_B']}",
        "our_probs": {
            "team_a_win": round(p_final_a, 4),
            "draw": round(p_final_draw, 4),
            "team_b_win": round(p_final_b, 4)
        },
        "fair_odds": {
            "team_a": round(odd_justo_a, 2),
            "draw": round(odd_justo_draw, 2),
            "team_b": round(odd_justo_b, 2)
        },
        "bookie_margin_pct": round(margin * 100, 2),
        "expected_value": {
            "team_a": round(ev_a, 4),
            "draw": round(ev_draw, 4),
            "team_b": round(ev_b, 4)
        },
        "value_bets": value_bets,
        "kelly_fraction": {
            "team_a": round(kelly_a, 4),
            "draw": round(kelly_draw, 4),
            "team_b": round(kelly_b, 4)
        },
        "confidence": confidence,
        "summary": summary
    }

if __name__ == "__main__":
    test_data = {
        "EQUIPO_A": "Mexico",
        "EQUIPO_B": "Germany",
        "ELO_A": 1850,
        "ELO_B": 1950,
        "FORMA_A": 2.0,
        "FORMA_B": 2.2,
        "GF_AVG_A": 1.5,
        "GF_AVG_B": 2.1,
        "GA_AVG_A": 1.0,
        "GA_AVG_B": 0.8,
        "H2H_WIN_RATE_A": 0.4,
        "VENUE": "Ciudad de México",
        "DIASPORA_A": False,
        "DIASPORA_B": False,
        "INJURIES_A": 0,
        "INJURIES_B": 1,
        "ODDS_BOOKIE_A": 3.80,
        "ODDS_BOOKIE_DRAW": 3.40,
        "ODDS_BOOKIE_B": 2.00
    }
    
    print("Testing predictor logic...")
    res = calculate_predictions(test_data)
    print(json.dumps(res, indent=2))
