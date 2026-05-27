"""
Módulo de lesiones para el Mundial 2026.

Fuentes:
  1. API-Football — endpoint /injuries (automático)
  2. data/injuries.json — base manual editable

Uso:
    from injuries_client import get_injuries, update_all_injuries
    count = get_injuries("Argentina")   # retorna número de lesionados clave
    update_all_injuries(teams_list)      # actualiza el JSON para todos los equipos
"""

import os
import json
import time
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_FOOTBALL_KEY")
BASE_URL = "https://v3.football.api-sports.io"
HEADERS = {
    "x-rapidapi-key": API_KEY or "",
    "x-rapidapi-host": "v3.football.api-sports.io",
}
INJURIES_FILE = os.path.join(os.path.dirname(__file__), "data", "injuries.json")

# Posiciones consideradas "clave" para el impacto en el rendimiento del equipo
KEY_POSITIONS = {"Goalkeeper", "Defender", "Midfielder", "Attacker",
                 "Portero", "Defensa", "Centrocampista", "Delantero"}


def _read_injuries_file() -> dict:
    try:
        with open(INJURIES_FILE, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _write_injuries_file(data: dict):
    os.makedirs(os.path.dirname(INJURIES_FILE), exist_ok=True)
    with open(INJURIES_FILE, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _get_team_id(team_name: str) -> int | None:
    """Busca el ID de la selección nacional en API-Football."""
    if not API_KEY:
        return None
    try:
        r = requests.get(f"{BASE_URL}/teams", headers=HEADERS, params={"search": team_name}, timeout=8)
        r.raise_for_status()
        resp = r.json().get("response", [])
        for t in resp:
            if t["team"].get("national"):
                return t["team"]["id"]
        if resp:
            return resp[0]["team"]["id"]
    except Exception:
        pass
    return None


def _fetch_injuries_from_api(team_id: int) -> list[dict] | None:
    """
    Obtiene jugadores con lesión o sanción activa desde API-Football.
    Usa el endpoint /injuries de la temporada actual.
    """
    if not API_KEY:
        return None
    current_year = 2026
    injured_players = []
    try:
        for season in [current_year, current_year - 1]:
            r = requests.get(
                f"{BASE_URL}/injuries",
                headers=HEADERS,
                params={"team": team_id, "season": season},
                timeout=10,
            )
            if r.status_code != 200:
                continue
            for item in r.json().get("response", []):
                player = item.get("player", {})
                reason = item.get("reason", "")
                pos    = player.get("type", "")
                name   = player.get("name", "Unknown")
                if reason and reason.lower() not in ("questionable", "day-to-day"):
                    injured_players.append({
                        "name": name,
                        "position": pos,
                        "reason": reason,
                    })
            if injured_players:
                break  # Con datos de la temporada actual es suficiente
    except Exception as e:
        print(f"  API error obteniendo lesiones: {e}")
        return None

    return injured_players


def get_injuries(team_name: str) -> int:
    """
    Retorna el número de jugadores clave lesionados del equipo.
    Lee del archivo local (más rápido y sin consumir API).
    """
    data = _read_injuries_file()
    entry = data.get(team_name, {})
    return entry.get("count", 0)


def get_injury_detail(team_name: str) -> dict:
    """Retorna el detalle completo de lesiones del equipo."""
    data = _read_injuries_file()
    return data.get(team_name, {"count": 0, "players": []})


def update_team_injuries(team_name: str, verbose: bool = True) -> int:
    """
    Actualiza las lesiones de un equipo consultando API-Football.
    Guarda el resultado en data/injuries.json.
    Retorna el conteo de lesionados.
    """
    all_data = _read_injuries_file()

    team_id = _get_team_id(team_name)
    if not team_id:
        if verbose:
            print(f"  {team_name}: sin ID en API-Football, usando datos manuales.")
        return all_data.get(team_name, {}).get("count", 0)

    players = _fetch_injuries_from_api(team_id)

    if players is None:
        if verbose:
            print(f"  {team_name}: API no disponible, datos sin cambio.")
        return all_data.get(team_name, {}).get("count", 0)

    count = len(players)
    all_data[team_name] = {
        "count": count,
        "players": players,
        "updated": time.strftime("%Y-%m-%d %H:%M"),
    }
    _write_injuries_file(all_data)

    if verbose:
        if players:
            names = ", ".join(p["name"] for p in players[:4])
            suffix = f" (+{count-4} más)" if count > 4 else ""
            print(f"  {team_name}: {count} lesionado(s) — {names}{suffix}")
        else:
            print(f"  {team_name}: sin lesiones registradas")

    return count


def update_all_injuries(teams: list[str], verbose: bool = True) -> dict:
    """
    Actualiza lesiones para todos los equipos de la lista.
    Retorna un dict {team_name: injury_count}.
    """
    results = {}
    for i, team in enumerate(teams, 1):
        if verbose:
            print(f"  ({i}/{len(teams)}) {team}...", end=" ", flush=True)
        count = update_team_injuries(team, verbose=False)
        results[team] = count
        if verbose:
            print(f"{count} lesionados")
        time.sleep(0.3)  # Evitar rate-limit de la API
    return results


def set_injury_manual(team_name: str, players: list[str]):
    """
    Registra lesiones manualmente (para cuando no hay datos de API).

    Ejemplo:
        set_injury_manual("France", ["Kylian Mbappé", "Ousmane Dembélé"])
    """
    all_data = _read_injuries_file()
    all_data[team_name] = {
        "count": len(players),
        "players": [{"name": p, "position": "Unknown", "reason": "Manual"} for p in players],
        "updated": time.strftime("%Y-%m-%d %H:%M"),
    }
    _write_injuries_file(all_data)
    print(f"Lesiones de {team_name} guardadas: {len(players)} jugador(es).")


if __name__ == "__main__":
    print("=== Test lesiones — Argentina ===")
    count = update_team_injuries("Argentina")
    print(f"Lesionados encontrados: {count}")
    print(get_injury_detail("Argentina"))
