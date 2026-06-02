"""
FIFA World Cup 2026 — Predictor

Modos de uso:
    python main.py                    # Predicción de un partido
    python main.py --tournament       # Simulación completa del torneo (1 vez)
    python main.py --sims 10000       # Monte Carlo (N simulaciones del torneo)
    python main.py --cache            # Ver estado de la caché
    python main.py --injury set       # Registrar lesiones manualmente
"""
import json
import sys
from src.scrapers.scraper import fetch_elo_ratings
from src.utils.odds_client import get_match_odds
from src.utils.stats_client import get_team_stats
from src.models.predictor import VENUES
from src.models.stacked_predictor import stack_predict, expected_goals
from src.utils.injuries_client import get_injuries, get_injury_detail, set_injury_manual
from src.utils.cache import get_elos, get_stats, cache_status

VENUE_LIST = list(VENUES.keys())


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────
def pick_venue() -> str:
    print("\nSedes del Mundial 2026:")
    for i, v in enumerate(VENUE_LIST, 1):
        info = VENUES[v]
        print(f"  {i:2}. {v:<22} ({info['country']}, {info['alt']}m alt, {info['temp']}°C)")
    raw = input("\nElige sede (número o nombre): ").strip()
    if raw.isdigit():
        idx = int(raw) - 1
        if 0 <= idx < len(VENUE_LIST):
            return VENUE_LIST[idx]
    for v in VENUE_LIST:
        if raw.lower() in v.lower():
            return v
    return VENUE_LIST[0]


def _get_team_data(team: str, elos: dict, fetch_fn) -> dict:
    """Construye el dict de datos de un equipo desde caché + defaults."""
    elo = elos.get(team) if elos else None
    if not elo:
        elo = int(input(f"ELO manual para {team}: "))

    stats = get_stats(team, fetch_fn)
    if not stats and team in DEFAULT_STATS:
        print(f"Usando stats de respaldo para {team}.")
        stats = DEFAULT_STATS[team]

    if stats:
        forma  = stats["FORMA"]
        gf_avg = stats["GF_AVG"]
        ga_avg = stats["GA_AVG"]
        print(f"  {team}: ELO={elo}  Forma={forma}  GF={gf_avg}  GA={ga_avg}")
    else:
        forma  = float(input(f"Puntos promedio {team}: ") or "1.5")
        gf_avg = float(input(f"GF promedio {team}: ")    or "1.2")
        ga_avg = float(input(f"GA promedio {team}: ")    or "1.0")

    injuries = get_injuries(team)
    if injuries:
        detail = get_injury_detail(team)
        names  = ", ".join(p["name"] for p in detail.get("players", [])[:3])
        print(f"  ⚠ {team}: {injuries} lesionado(s) — {names}")

    return {
        "name":     team,
        "ELO":      elo,
        "FORMA":    forma,
        "GF_AVG":   gf_avg,
        "GA_AVG":   ga_avg,
        "INJURIES": injuries,
    }


DEFAULT_STATS = {
    "Argentina":   {"FORMA": 2.4, "GF_AVG": 2.1, "GA_AVG": 0.6},
    "France":      {"FORMA": 2.1, "GF_AVG": 1.9, "GA_AVG": 0.8},
    "Brazil":      {"FORMA": 2.0, "GF_AVG": 1.8, "GA_AVG": 0.7},
    "England":     {"FORMA": 1.9, "GF_AVG": 1.7, "GA_AVG": 0.9},
    "Spain":       {"FORMA": 2.0, "GF_AVG": 1.8, "GA_AVG": 0.7},
    "Germany":     {"FORMA": 1.8, "GF_AVG": 1.9, "GA_AVG": 1.1},
    "Portugal":    {"FORMA": 1.9, "GF_AVG": 2.0, "GA_AVG": 1.0},
    "Netherlands": {"FORMA": 1.8, "GF_AVG": 1.7, "GA_AVG": 1.0},
    "Belgium":     {"FORMA": 1.7, "GF_AVG": 1.6, "GA_AVG": 0.9},
    "Croatia":     {"FORMA": 1.7, "GF_AVG": 1.4, "GA_AVG": 0.9},
    "Italy":       {"FORMA": 1.7, "GF_AVG": 1.4, "GA_AVG": 0.8},
    "Colombia":    {"FORMA": 1.8, "GF_AVG": 1.6, "GA_AVG": 0.9},
    "Uruguay":     {"FORMA": 1.7, "GF_AVG": 1.5, "GA_AVG": 0.9},
    "Morocco":     {"FORMA": 1.7, "GF_AVG": 1.3, "GA_AVG": 0.7},
    "Japan":       {"FORMA": 1.7, "GF_AVG": 1.5, "GA_AVG": 0.9},
    "USA":         {"FORMA": 1.5, "GF_AVG": 1.4, "GA_AVG": 1.1},
    "Mexico":      {"FORMA": 1.5, "GF_AVG": 1.5, "GA_AVG": 1.2},
    "Canada":      {"FORMA": 1.4, "GF_AVG": 1.3, "GA_AVG": 1.1},
    "Ecuador":     {"FORMA": 1.5, "GF_AVG": 1.3, "GA_AVG": 1.1},
    "South Korea": {"FORMA": 1.5, "GF_AVG": 1.4, "GA_AVG": 1.0},
    "Senegal":     {"FORMA": 1.6, "GF_AVG": 1.4, "GA_AVG": 1.0},
    "Switzerland": {"FORMA": 1.7, "GF_AVG": 1.5, "GA_AVG": 0.8},
    "Poland":      {"FORMA": 1.4, "GF_AVG": 1.3, "GA_AVG": 1.1},
    "Serbia":      {"FORMA": 1.5, "GF_AVG": 1.4, "GA_AVG": 1.1},
    "Australia":   {"FORMA": 1.4, "GF_AVG": 1.2, "GA_AVG": 1.1},
    "Iran":        {"FORMA": 1.5, "GF_AVG": 1.3, "GA_AVG": 1.0},
}


# ─────────────────────────────────────────────
#  Modo 1: predicción de un partido
# ─────────────────────────────────────────────
def run_match_predictor():
    print("=" * 45)
    print("   FIFA WORLD CUP 2026 — PREDICTOR")
    print("=" * 45)

    team_a = input("\nEquipo A: ").strip()
    team_b = input("Equipo B: ").strip()

    elos = get_elos(fetch_elo_ratings)
    print(f"\nDatos de {team_a}:")
    data_a = _get_team_data(team_a, elos, get_team_stats)
    print(f"\nDatos de {team_b}:")
    data_b = _get_team_data(team_b, elos, get_team_stats)

    venue = pick_venue()
    print(f"Sede: {venue}")

    # Ajustes opcionales
    print("\nAjustes especiales (Enter para omitir):")
    inj_override_a = input(f"  Lesionados adicionales en {team_a} (Enter = usar caché): ").strip()
    inj_override_b = input(f"  Lesionados adicionales en {team_b} (Enter = usar caché): ").strip()
    if inj_override_a.isdigit(): data_a["INJURIES"] = int(inj_override_a)
    if inj_override_b.isdigit(): data_b["INJURIES"] = int(inj_override_b)

    dias_a = input(f"  ¿{team_a} juega con diáspora local? (s/n): ").strip().lower() == "s"
    dias_b = input(f"  ¿{team_b} juega con diáspora local? (s/n): ").strip().lower() == "s"
    data_a["DIASPORA"] = dias_a
    data_b["DIASPORA"] = dias_b

    extra_a = input(f"  Multiplicador extra para {team_a} (ej. 1.05 = +5% boost) (Enter = 1.0): ").strip()
    extra_b = input(f"  Multiplicador extra para {team_b} (ej. 1.05 = +5% boost) (Enter = 1.0): ").strip()
    data_a["EXTRA"] = float(extra_a) if extra_a else 1.0
    data_b["EXTRA"] = float(extra_b) if extra_b else 1.0

    # Odds
    print("\nBuscando cuotas en vivo...")
    odds = get_match_odds(team_a, team_b)
    if not odds:
        print("Sin cuotas en vivo. Introduce manualmente (o Enter para omitir análisis de value):")
        try:
            odd_a    = float(input(f"  Cuota {team_a}: ") or "0")
            odd_draw = float(input("  Cuota Empate: ")    or "0")
            odd_b    = float(input(f"  Cuota {team_b}: ") or "0")
            odds = {"ODDS_BOOKIE_A": odd_a, "ODDS_BOOKIE_DRAW": odd_draw, "ODDS_BOOKIE_B": odd_b}
        except ValueError:
            odds = None

    # Predicción con modelo apilado
    print("\nCalculando (modelo apilado: ELO + Poisson + Forma + H2H)...\n")
    pred = stack_predict(data_a, data_b, venue_name=venue)
    lam_a, lam_b = expected_goals(data_a, data_b, venue_name=venue)

    print("=" * 45)
    print("      RESULTADO DEL PREDICTOR")
    print("=" * 45)
    print(f"\n  {team_a} vs {team_b}  |  Sede: {venue}")
    print(f"\n  Probabilidades:")
    print(f"    {team_a:<20} {pred['p_win_a']*100:>5.1f}%")
    print(f"    {'Empate':<20} {pred['p_draw']*100:>5.1f}%")
    print(f"    {team_b:<20} {pred['p_win_b']*100:>5.1f}%")
    print(f"\n  Goles esperados: {team_a} {lam_a:.2f} — {lam_b:.2f} {team_b}")
    print(f"\n  Cuotas justas:")
    print(f"    {team_a:<20} {1/pred['p_win_a']:>5.2f}")
    print(f"    {'Empate':<20} {1/pred['p_draw']:>5.2f}")
    print(f"    {team_b:<20} {1/pred['p_win_b']:>5.2f}")

    print(f"\n  Desglose de modelos:")
    for model, vals in pred["model_breakdown"].items():
        print(f"    {model:<9} A={vals['win_a']*100:.1f}%  X={vals['draw']*100:.1f}%  "
              f"B={vals['win_b']*100:.1f}%  (w={vals['weight']:.2f})")

    if odds and odds.get("ODDS_BOOKIE_A") and odds.get("ODDS_BOOKIE_A") > 0:
        print(f"\n  Análisis de valor vs bookmaker:")
        for label, p_our, odd in [
            (team_a,  pred["p_win_a"], odds["ODDS_BOOKIE_A"]),
            ("Empate", pred["p_draw"],  odds["ODDS_BOOKIE_DRAW"]),
            (team_b,  pred["p_win_b"], odds["ODDS_BOOKIE_B"]),
        ]:
            ev = p_our * odd - 1
            tag = " ← VALUE BET" if ev > 0.05 else ""
            print(f"    {label:<20} cuota={odd:.2f}  EV={ev:+.3f}{tag}")
    print()


# ─────────────────────────────────────────────
#  Modo 2: simulación del torneo
# ─────────────────────────────────────────────
def run_tournament(n_sims: int = 1):
    import src.models.tournament as t_module
    team_db = t_module.load_team_db()
    groups  = t_module.load_groups()

    if n_sims <= 1:
        print("\n" + "=" * 55)
        print("  SIMULACIÓN DEL MUNDIAL 2026 (modelo apilado)")
        print("=" * 55)
        result = t_module.sim_full_tournament(groups, team_db)
        print(f"\n{'='*55}")
        print(f"  CAMPEÓN DEL MUNDO:  {result['champion'].upper()}")
        print(f"  Finalista:          {result['finalist']}")
        print(f"  3er Puesto:         {result['third']}")
        sf_others = [t for t in result["semifinalists"]
                     if t not in [result["champion"], result["finalist"], result["third"]]]
        if sf_others:
            print(f"  4to Puesto:         {sf_others[0]}")
        print(f"\n  Grupos (1ro / 2do / 3ro / 4to):")
        for letter, ranking in sorted(result["group_results"].items()):
            print(f"    Grupo {letter}: {' > '.join(ranking[:4])}")
        print(f"{'='*55}\n")
    else:
        print(f"\n{'='*55}")
        print(f"  MONTE CARLO — {n_sims:,} sims — MUNDIAL 2026")
        print(f"{'='*55}\n")
        table = t_module.monte_carlo(groups, team_db, n=n_sims)
        print(f"\n{'Equipo':<22} {'Campeón':>9} {'Final':>8} {'Semi':>7} {'3ro':>7}")
        print("-" * 58)
        for row in table[:20]:
            print(f"  {row['team']:<20} {row['champion_%']:>7.2f}%  {row['final_%']:>6.2f}%  "
                  f"{row['semi_%']:>5.2f}%  {row['third_%']:>5.2f}%")
        import os
        out = os.path.join("data", "mc_results.json")
        with open(out, "w") as f:
            json.dump(table, f, indent=2, ensure_ascii=False)
        print(f"\nResultados completos guardados en {out}")


# ─────────────────────────────────────────────
#  Modo 3: registro manual de lesiones
# ─────────────────────────────────────────────
def run_set_injury():
    team = input("Equipo: ").strip()
    print("Ingresa nombres de jugadores lesionados (Enter en blanco para terminar):")
    players = []
    while True:
        p = input(f"  Jugador {len(players)+1}: ").strip()
        if not p:
            break
        players.append(p)
    if players:
        set_injury_manual(team, players)
    else:
        print("Ningún jugador ingresado.")


# ─────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────
def main():
    args = sys.argv[1:]

    if "--cache" in args:
        cache_status()
        return

    if "--injury" in args and "set" in args:
        run_set_injury()
        return

    n_sims = 1
    for i, a in enumerate(args):
        if a == "--sims" and i + 1 < len(args):
            n_sims = int(args[i + 1])

    if "--tournament" in args or "--sims" in args:
        run_tournament(n_sims)
        return

    run_match_predictor()


if __name__ == "__main__":
    main()
