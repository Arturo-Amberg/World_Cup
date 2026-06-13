---
target: static/index.html
total_score: 26
p0_count: 0
p1_count: 2
p2_count: 3
timestamp: 2026-06-13T06-55-12Z
slug: static-index-html
---
## Design Health Score

| # | Heuristic | Score | Key Issue |
|---|-----------|-------|-----------|
| 1 | Visibility of System Status | 3 | Loading spinner + "Calculando…" feedback is good; silent API failures are the gap |
| 2 | Match System / Real World | 3 | "VE", "Kelly%", "xG" never explained inline — assumed knowledge |
| 3 | User Control and Freedom | 3 | Reset button and keyboard shortcuts help; no one-tap filter clear |
| 4 | Consistency and Standards | 3 | Nav tabs use `<div>`, predictor uses `<button>` — divergent semantics |
| 5 | Error Prevention | 2 | Odds inputs accept negatives; no guard against selecting the same team A/B |
| 6 | Recognition Rather Than Recall | 3 | Main nav visible; "VE/Kelly/ELO/xG" have no tooltips |
| 7 | Flexibility and Efficiency | 3 | Keyboard shortcuts, column sort, category filters; no team quick-search |
| 8 | Aesthetic and Minimalist Design | 2 | Vanta.js Three.js animation behind every data page competes with content |
| 9 | Error Recovery | 2 | Notification system exists; API error surface is thin |
| 10 | Help and Documentation | 2 | Shortcuts panel is good; zero contextual tooltips on jargon |
| **Total** | | **26/40** | **Acceptable — significant improvements possible** |

## Anti-Patterns Verdict

**LLM assessment**: The aesthetic holds up — the palette is restrained and purposeful, shadow hierarchy is deliberate, and the data typography (tabular-nums, mono for scores) is genuinely considered. It does not look obviously AI-generated. That said, two second-order tells surface: (1) the 4-up KPI grid at the top of every section is a recurring SaaS template, and (2) the card-badge-progress-bar trio appears identically across every section without variation, flattening the visual rhythm. The group stage page in particular could use compositional variety — all 12 group cards are identical containers.

**Deterministic scan**: CLI detector unavailable (bundled detector not found). Assessment B run on source review only.

**Visual overlays**: Browser injection unavailable in this session. No overlay was created.

## Overall Impression

A technically solid analytical tool that reads as "clean and data-forward" rather than "AI template." The biggest single problem is the Vanta.js animated background — it is the definition of decorative motion behind dense data, and it contradicts the premium-airy aesthetic it's trying to achieve. Remove it and the interface immediately feels more authoritative. The second-biggest opportunity is accessibility: navigation tabs are divs, not interactive elements, which means keyboard-only users cannot use the primary navigation at all (only the number shortcuts work).

## What's Working

**Data typography**: Monospace for all numeric values, `font-variant-numeric: tabular-nums` throughout, weight contrast between labels and values — the numbers are extremely readable. This is where the "professional instrument" feeling comes through most clearly.

**Motion system**: The ease-out-expo easing curve, 200ms base duration, and page-in animation are all correctly calibrated. State transitions feel native. The `prefers-reduced-motion` media query exists and disables animations — rare and appreciated.

**Predictor UX**: The three-column selector (Team A / VS / Team B) with inline odds input is an elegant layout. The results reveal animation is smooth and the tabbed breakdown (Resultado / Marcadores / Cuotas) is the right information architecture.

## Priority Issues

**[P1] Vanta.js: decorative motion on every page**
- **Why it matters**: A spinning Three.js particle field behind 12 group cards and 635 value bet rows is visual noise competing with the content. It contradicts the "precision over decoration" principle and costs ~350KB of JS (Three.js r134 + Vanta). On lower-end hardware it drops the animation below 30fps, causing perceived lag on scrolling.
- **Fix**: Remove Vanta entirely. Replace `#vanta-bg` with a single CSS grain texture or subtle dot-grid background (pure CSS, zero JS). The cool off-white base color already provides elegance without motion.
- **Suggested command**: `/impeccable polish`

**[P1] Navigation is div-based — keyboard inaccessible**
- **Why it matters**: `.page-tab` elements are `<div onclick="...">` with no `tabindex`, no `role`, no `onkeydown`. A user pressing Tab cannot reach the navigation at all. Only the number shortcuts (1–5) work, and those are undiscoverable unless the user presses "?". WCAG 2.4.3 failure.
- **Fix**: Convert `.page-tab` to `<button>` elements (or add `tabindex="0"` + `role="tab"` + `onkeydown` Enter/Space handler). Same fix needed for the header pills (`hdr-pill`) and the "?" button which is currently a bare `<div>`.
- **Suggested command**: `/impeccable audit`

**[P2] Stale keyboard shortcuts panel**
- **Why it matters**: The overlay maps key `4` to "Modelo" — a page that doesn't exist in the current nav. The actual nav is Inicio/Predictor/Apuestas/Marcadores/Cuadro (keys 1–5). This is a direct contradiction that users will notice immediately on pressing "4".
- **Fix**: Update the shortcuts panel to match current nav: 1=Inicio, 2=Predictor, 3=Apuestas, 4=Marcadores, 5=Cuadro. Remove the "Modelo" row.
- **Suggested command**: `/impeccable harden`

**[P2] Value bets filter overload — 6 simultaneous controls**
- **Why it matters**: Category tabs + search + sort dropdown + Kelly min + Prob min + EV max = 6 controls visible at once, exceeding the 4-item working-memory limit. On mobile they collapse into a 2-column grid that wraps inconsistently.
- **Fix**: Collapse Kelly/Prob/EV range inputs into a single "Filtros avanzados" disclosure row (collapsed by default, expands inline). Keep category tabs + search + sort as the primary bar — 3 controls. Add a "Limpiar filtros" single button when any advanced filter is active.
- **Suggested command**: `/impeccable layout`

**[P2] Color-only outcome encoding in probability bars and group cards**
- **Why it matters**: The tricolor prob bar (blue=win A, gold=draw, red=win B) conveys all meaning through hue alone. Group card probabilities use `.c-lime`, `.c-gold`, `.c-red` classes with no adjacent text labels. ~8% of male users have red-green deficiency; blue-yellow variants affect another 2%. "Win / Draw / Loss" cannot be read without color.
- **Fix**: Add micro-labels ("V" / "E" / "D" or "W"/"D"/"L") directly below each probability number in the group match cards. For the prob bar, add a `title` attribute and an accessible legend.
- **Suggested command**: `/impeccable audit`

## Persona Red Flags

**Alex (Power User)**: Keyboard shortcuts exist — 10/10 for discoverability of the "?" panel. But the predictor has no type-ahead search; selecting teams from a full 48-team `<select>` on a keyboard requires scrolling through the entire list. After running a prediction, returning to edit just one team requires clearing and restarting — no partial edit. The value bets table sort is clickable on column headers, which Alex will find and use immediately — this works well.

**Sam (Accessibility-Dependent User)**: Critical failure on the primary navigation. Tab key cannot reach the page tabs at all without the number shortcuts. The prob bars and group card probabilities convey win/draw/loss through color alone with no accessible alternative. The `<div class="vb-cat-tab">` filter tabs inside the value bets section are also divs with `user-select:none` and no keyboard affordance. The `<select>` elements in the predictor are standard HTML — these work correctly. Focus rings via `focus-visible` are correctly implemented on buttons, which is good.

**Project-specific: "The Methodical High-Stakes Bettor" (from PRODUCT.md)**
- Profile: Reviews every column of the value bets table before placing anything. Cross-references Kelly% against their personal bankroll model.
- Red flags: "VE" column header has no tooltip explaining it is Expected Value × 100. "Kelly" column is listed as a decimal (0.28) in the raw data but displayed as a percentage in the UI — new users will see "28%" and wonder "28 of what?". The model breakdown (Poisson 60% / ELO 40%) is buried in the predictor results and has no documentation link. The bettor will want to understand exactly how the model weights work before trusting its output.

## Minor Observations

- `clamp(52px, 8vw, 84px)` on the score display number is a fluid typography technique — the product register notes "fixed rem scale, not fluid" for product UIs. This one instance is harmless but worth standardizing.
- The `--gold` text color is `#c79a00` (WCAG contrast ~4.1:1 on white) — just below the 4.5:1 AA threshold for normal text. Several `.c-gold` usages on white backgrounds may fail contrast.
- The `.vb-kpi-label` uses `font-size: 10px` and `.vb-cat-badge` uses `font-size: 9px` — both below the 11px practical minimum for legibility on retina displays. WCAG requires 4.5:1 at any size; at 9px even passing contrast ratios are difficult to read.
- `transition: all 0.15s` on `.player-card` — "all" transitions every property including layout, which the shared design laws ban. Target `border-color, box-shadow` specifically.
- Bracket page shows a rotate hint on mobile portrait, but the bracket itself doesn't have a minimum touch target height — bracket nodes may be too small to tap accurately.

## Questions to Consider

- "The Vanta background adds weight and motion — what are you replacing it with? A static subtle texture could keep the 'not default white' feeling without the animation cost."
- "The Modelo page appears in shortcuts but not in nav — was it intentionally hidden, or is there unfinished validation/model transparency content worth surfacing?"
- "Six filter controls on value bets: do high-stakes bettors actually use Kelly/Prob/EV range filters independently, or do most users just search by team name and sort by Kelly?"
