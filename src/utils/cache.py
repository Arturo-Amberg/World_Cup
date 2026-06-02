import json
import os
import time

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
ELO_CACHE_FILE = os.path.join(CACHE_DIR, "elos.json")
STATS_CACHE_FILE = os.path.join(CACHE_DIR, "team_stats.json")

ELO_TTL = 24 * 3600        # Renueva ELOs cada 24h
STATS_TTL = 7 * 24 * 3600  # Renueva stats cada 7 días


def _read(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _write(path, data):
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def get_elos(fetch_fn):
    cached = _read(ELO_CACHE_FILE)
    age = time.time() - cached.get("ts", 0) if cached else float("inf")
    if cached and age < ELO_TTL:
        print(f"ELOs cargados desde caché ({int(age/3600)}h de antigüedad).")
        return cached["data"]
    print("Actualizando ELOs desde eloratings.net...")
    data = fetch_fn()
    if data:
        _write(ELO_CACHE_FILE, {"ts": time.time(), "data": data})
    elif cached:
        print("Fallo al actualizar, usando caché anterior.")
        return cached["data"]
    return data


def get_stats(team_name, fetch_fn):
    cached = _read(STATS_CACHE_FILE) or {}
    entry = cached.get(team_name)
    age = time.time() - entry.get("ts", 0) if entry else float("inf")
    if entry and age < STATS_TTL:
        print(f"Stats de {team_name} cargadas desde caché ({int(age/86400)}d de antigüedad).")
        return entry["data"]
    print(f"Descargando stats de {team_name} desde API-Football...")
    data = fetch_fn(team_name)
    if data:
        cached[team_name] = {"ts": time.time(), "data": data}
        _write(STATS_CACHE_FILE, cached)
    elif entry:
        print(f"Fallo al actualizar stats de {team_name}, usando caché anterior.")
        return entry["data"]
    return data


def force_refresh_stats(team_name, fetch_fn):
    """Fuerza refresco ignorando caché."""
    cached = _read(STATS_CACHE_FILE) or {}
    print(f"Forzando actualización de stats de {team_name}...")
    data = fetch_fn(team_name)
    if data:
        cached[team_name] = {"ts": time.time(), "data": data}
        _write(STATS_CACHE_FILE, cached)
    return data


def cache_status():
    """Imprime el estado actual de la caché."""
    elos = _read(ELO_CACHE_FILE)
    stats = _read(STATS_CACHE_FILE) or {}
    now = time.time()

    print("\n--- Estado de Caché ---")
    if elos:
        age_h = int((now - elos.get("ts", 0)) / 3600)
        print(f"ELOs: {len(elos['data'])} países, {age_h}h de antigüedad")
    else:
        print("ELOs: sin caché")

    if stats:
        print(f"Stats guardadas: {len(stats)} equipos")
        for team, entry in sorted(stats.items()):
            age_d = int((now - entry.get("ts", 0)) / 86400)
            d = entry["data"]
            print(f"  {team}: Forma={d['FORMA']} GF={d['GF_AVG']} GA={d['GA_AVG']} ({age_d}d)")
    else:
        print("Stats: sin caché")
    print("-----------------------\n")
