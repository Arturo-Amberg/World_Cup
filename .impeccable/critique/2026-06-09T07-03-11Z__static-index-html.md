---
target: the homepage
total_score: 28
p0_count: 1
p1_count: 1
timestamp: 2026-06-09T07-03-11Z
slug: static-index-html
---
#### Design Health Score
> *Based on Nielsen's 10 Heuristics*

| # | Heuristic | Score | Key Issue |
|---|-----------|-------|-----------|
| 1 | Visibility of System Status | 3/4 | Filtros se aplican de golpe, pero la carga inicial es clara. |
| 2 | Match System / Real World | 4/4 | Terminología de apuestas precisa (Cuota justa, Ventaja). |
| 3 | User Control and Freedom | 2/4 | No hay un botón fácil para "Limpiar filtros" o "Deshacer". |
| 4 | Consistency and Standards | 2/4 | La sección de parlays usa un lenguaje visual distinto y rígido. |
| 5 | Error Prevention | 3/4 | Los filtros están acotados a opciones válidas. |
| 6 | Recognition Rather Than Recall | 4/4 | Todas las opciones relevantes están a la vista. |
| 7 | Flexibility and Efficiency | 2/4 | Falta navegación ágil por teclado para usuarios avanzados. |
| 8 | Aesthetic and Minimalist Design | 3/4 | "Mejores parlays" y tablas redundantes saturan la interfaz. |
| 9 | Error Recovery | 3/4 | N/A |
| 10 | Help and Documentation | 2/4 | Faltan tooltips explicativos para métricas complejas (EV). |
| **Total** | | **28/40** | **Mejorable** |

#### Anti-Patterns Verdict

**LLM assessment**: La estética general ha mejorado (más macOS, menos "AI slop"), pero la estructura de la página de Apuestas aún delata rigidez. La sección de "Mejores Parlays" se siente artificial, como un módulo pegado en lugar de una herramienta orgánica. La tabla inicial de *Inicio* también resulta densa e intimida en el primer impacto.
**Deterministic scan**: Deterministic scan unavailable. Manual review confirms structural friction.
**Visual overlays**: [Skipped: Detector unavailable]

#### Overall Impression
La dirección visual de "terminal analítica profesional" funciona bien, pero la interacción se queda corta. El usuario puede *ver* los datos, pero no puede *actuar* sobre ellos libremente. La oportunidad más grande es transformar los datos estáticos en herramientas interactivas (como un Armador de Parlays).

#### What's Working
1. **Filtros unificados**: El nuevo diseño inline para filtros de apuestas es compacto, profesional y reduce drásticamente el ruido visual.
2. **Jerarquía tipográfica**: La paleta neutra con fuentes de sistema da una sensación nativa y confiable.

#### Priority Issues
- **[P0] "Mejores Parlays" Rígidos**: La sección pre-calculada de parlays es inflexible y se siente desactualizada.
  - *Why it matters*: Los apostadores de alto nivel (nuestro target) quieren crear sus propias combinaciones, no consumir sugerencias genéricas.
  - *Fix*: Eliminar la sección estática y reemplazarla con un "Armador de Parlay" flotante que reaccione al clic en las cuotas.
  - *Suggested command*: `/impeccable craft`
- **[P1] Redundancia en "Inicio"**: La tabla de *Selección/Campeón* en la vista principal es información abrumadora y repetitiva.
  - *Why it matters*: Incrementa la carga cognitiva en el punto de entrada sin ofrecer una acción clara.
  - *Fix*: Eliminar esta tabla del inicio y mantener el foco en la fase de grupos y simulaciones.
  - *Suggested command*: `/impeccable distill`

#### Persona Red Flags
**Alex (Power User)**: Ve cuotas que le gustan pero no puede combinarlas en la plataforma. Tiene que anotar las apuestas manualmente en otra ventana para calcular el parlay. Frustración alta.
**Jordan (First-Timer)**: La tabla de Inicio le tira demasiados números ("Cuota justa", "Semifinal") sin contexto inmediato.

#### Minor Observations
- Faltaría un botón sutil de "Reset" al lado de los filtros.
- Los números de cuotas deberían tener feedback táctil (hover state fuerte) si los vamos a volver clickeables.

#### Questions to Consider
- "¿Qué pasaría si la tabla principal no fuera solo lectura, sino el panel de control para armar estrategias complejas?"
- "¿Necesitamos realmente la pestaña Inicio si toda la acción ocurre en Apuestas?"
