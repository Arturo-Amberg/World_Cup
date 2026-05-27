import os
import requests
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("ODDS_API_KEY")

def get_match_odds(team_a, team_b):
    """
    Intenta obtener cuotas reales desde The Odds API.
    Si no hay datos disponibles, retorna None.
    """
    if not API_KEY or API_KEY == "tu_clave_api_aqui":
        print("API_KEY inválida o no configurada para Odds API.")
        return None
        
    # Usaremos soccer_fifa_world_cup como sport_key (o soccer_international si no esta activo)
    sport_key = "soccer_fifa_world_cup" 
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds"
    
    params = {
        "apiKey": API_KEY,
        "regions": "eu,us",
        "markets": "h2h",
        "oddsFormat": "decimal"
    }
    
    try:
        response = requests.get(url, params=params)
        if response.status_code != 200:
            print(f"Aviso Odds API: {response.status_code} - Quizá el torneo aún no está publicado en la API.")
            return None
            
        matches = response.json()
        for match in matches:
            home_team = match.get("home_team", "")
            away_team = match.get("away_team", "")
            
            if (team_a in home_team or team_a in away_team) and (team_b in home_team or team_b in away_team):
                bookmakers = match.get("bookmakers", [])
                if bookmakers:
                    # Promediamos sobre todas las bookies disponibles para mayor precisión
                    all_odds_a, all_odds_draw, all_odds_b = [], [], []
                    for bookie in bookmakers:
                        for market in bookie.get("markets", []):
                            if market.get("key") != "h2h":
                                continue
                            for outcome in market.get("outcomes", []):
                                name = outcome["name"]
                                price = outcome["price"]
                                if name == "Draw":
                                    all_odds_draw.append(price)
                                elif team_a in name:
                                    all_odds_a.append(price)
                                elif team_b in name:
                                    all_odds_b.append(price)

                    if all_odds_a and all_odds_draw and all_odds_b:
                        # Best available odds across all bookmakers
                        return {
                            "ODDS_BOOKIE_A":    round(max(all_odds_a), 3),
                            "ODDS_BOOKIE_DRAW": round(max(all_odds_draw), 3),
                            "ODDS_BOOKIE_B":    round(max(all_odds_b), 3)
                        }
        print("No se encontraron cuotas para este partido específico.")
    except Exception as e:
        print(f"Error conectando a The Odds API: {e}")
        
    return None

def get_all_odds():
    """
    Fetches ALL available WC 2026 match odds in a single API call.
    Returns dict keyed by frozenset of team names → {a, draw, b, home, away}
    """
    if not API_KEY or API_KEY == "tu_clave_api_aqui":
        return {}

    url = f"https://api.the-odds-api.com/v4/sports/soccer_fifa_world_cup/odds"
    params = {"apiKey": API_KEY, "regions": "eu,us", "markets": "h2h",
              "oddsFormat": "decimal"}
    result = {}
    try:
        resp = requests.get(url, params=params, timeout=8)
        if resp.status_code != 200:
            return {}
        for match in resp.json():
            home = match.get("home_team", "")
            away = match.get("away_team", "")
            all_a, all_d, all_b = [], [], []
            for bookie in match.get("bookmakers", []):
                for mkt in bookie.get("markets", []):
                    if mkt.get("key") != "h2h":
                        continue
                    for o in mkt.get("outcomes", []):
                        n, p = o["name"], o["price"]
                        if n == "Draw":   all_d.append(p)
                        elif n == home:   all_a.append(p)
                        elif n == away:   all_b.append(p)
            if all_a and all_d and all_b:
                result[frozenset([home, away])] = {
                    "home": home, "away": away,
                    "a":    round(max(all_a), 3),
                    "draw": round(max(all_d), 3),
                    "b":    round(max(all_b), 3),
                }
    except Exception as e:
        print(f"Odds bulk fetch error: {e}")
    return result


if __name__ == "__main__":
    print("Testeando Odds API (Si hay partidos de mundial disponibles)...")
    odds = get_match_odds("Argentina", "Mexico")
    print(odds)
