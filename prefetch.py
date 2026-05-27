"""
Pre-carga todos los datos necesarios para simular el Mundial 2026.

    python prefetch.py              # ELOs + stats + lesiones
    python prefetch.py --elos       # Solo ELOs
    python prefetch.py --stats      # Solo stats
    python prefetch.py --injuries   # Solo lesiones
    python prefetch.py --status     # Ver estado de la caché
"""
import sys
import time
from scraper import fetch_elo_ratings
from stats_client import get_team_stats
from injuries_client import update_all_injuries
from cache import get_elos, force_refresh_stats, cache_status

WC2026_TEAMS = [
    # CONMEBOL
    "Argentina", "Brazil", "Colombia", "Uruguay", "Ecuador",
    "Paraguay", "Bolivia", "Venezuela",
    # CONCACAF
    "USA", "Mexico", "Canada", "Panama", "Costa Rica",
    "Honduras", "Jamaica", "El Salvador", "Guatemala",
    # Europa
    "France", "England", "Spain", "Germany", "Portugal",
    "Netherlands", "Belgium", "Croatia", "Switzerland", "Italy",
    "Turkey", "Poland", "Serbia", "Austria", "Ukraine",
    "Slovakia", "Romania",
    # África
    "Morocco", "Senegal", "Nigeria", "Cameroon", "Ivory Coast",
    "Ghana", "Algeria", "Tunisia", "Egypt", "South Africa",
    # Asia
    "Japan", "South Korea", "Australia", "Iran", "Saudi Arabia",
    "Iraq", "Uzbekistan", "Indonesia",
    # Oceanía
    "New Zealand",
]


def run_elos():
    print("\n[ELOs] Descargando desde eloratings.net...")
    elos = get_elos(fetch_elo_ratings)
    if elos:
        matched = [t for t in WC2026_TEAMS if t in elos]
        missing = [t for t in WC2026_TEAMS if t not in elos]
        print(f"  OK — {len(matched)}/{len(WC2026_TEAMS)} equipos encontrados.")
        if missing:
            print(f"  Sin ELO: {', '.join(missing)}")
        for t in WC2026_TEAMS:
            if t in elos:
                print(f"    {t}: {elos[t]}")
    else:
        print("  Error: no se pudieron obtener ELOs.")


def run_stats():
    print(f"\n[Stats] Descargando para {len(WC2026_TEAMS)} equipos...")
    ok, fail = 0, []
    for i, team in enumerate(WC2026_TEAMS, 1):
        print(f"  ({i:2}/{len(WC2026_TEAMS)}) {team:<22}", end="", flush=True)
        stats = force_refresh_stats(team, get_team_stats)
        if stats:
            print(f"OK  Forma={stats['FORMA']:.2f}  GF={stats['GF_AVG']:.2f}  GA={stats['GA_AVG']:.2f}")
            ok += 1
        else:
            print("sin datos (API no disponible o equipo no encontrado)")
            fail.append(team)
        time.sleep(7)  # Free tier: ~10 req/min — each team uses ~5 calls
        if i % 5 == 0:
            print(f"  (pausing 15s to avoid rate limit...)")
            time.sleep(15)

    print(f"\n  Stats guardadas: {ok}/{len(WC2026_TEAMS)}")
    if fail:
        print(f"  Sin datos: {', '.join(fail)}")


def run_injuries():
    print(f"\n[Lesiones] Actualizando para {len(WC2026_TEAMS)} equipos...")
    results = update_all_injuries(WC2026_TEAMS, verbose=True)
    injured = {t: c for t, c in results.items() if c > 0}
    if injured:
        print("\n  Equipos con lesiones:")
        for team, count in sorted(injured.items(), key=lambda x: -x[1]):
            print(f"    {team}: {count} lesionado(s)")
    else:
        print("  Ningún equipo con lesiones registradas (o API no disponible).")


def main():
    args = sys.argv[1:]

    if "--status" in args:
        cache_status()
        return

    only_elos     = "--elos"     in args
    only_stats    = "--stats"    in args
    only_injuries = "--injuries" in args
    run_all = not (only_elos or only_stats or only_injuries)

    print("=" * 50)
    print("  PRE-CARGA DE DATOS — Mundial 2026")
    print("=" * 50)

    if run_all or only_elos:
        run_elos()

    if run_all or only_stats:
        run_stats()

    if run_all or only_injuries:
        run_injuries()

    print("\nListo.")
    cache_status()


if __name__ == "__main__":
    main()
