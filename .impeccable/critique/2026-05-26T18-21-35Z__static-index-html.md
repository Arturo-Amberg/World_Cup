---
target: static/index.html
total_score: 26
p0_count: 1
p1_count: 2
timestamp: 2026-05-26T18-21-35Z
slug: static-index-html
---
# Design Critique: static/index.html

## Design Health Score
> *Heuristic evaluation based on Nielsen's 10 standards.*

| # | Heuristic | Score | Key Issue |
|---|-----------|-------|-----------|
| 1 | Visibility of System Status | 4 | Excellent use of loading spinners and animated progress bars. |
| 2 | Match System / Real World | 4 | Betting-fluent language ("Home/Away", "Bookie Odds") is effective. |
| 3 | User Control and Freedom | 3 | Clear navigation, but SPA "Back" behavior is a potential risk. |
| 4 | Consistency and Standards | 3 | Consistent color logic (Lime/Red/Gold) and typography weights. |
| 5 | Error Prevention | 3 | Select-based team choice prevents invalid match-ups. |
| 6 | Recognition Rather Than Recall | 3 | Tabs help visibility, but some labels ("Revenue Strategy" vs "Edges") shift. |
| 7 | Flexibility and Efficiency | 2 | No keyboard shortcuts for core actions like "ANALYZE". |
| 8 | Aesthetic and Minimalist Design | 2 | Very noisy; grid background + radial glows + dense tables create friction. |
| 9 | Error Recovery | 1 | No visible handling for fetch failures or missing data. |
| 10 | Help and Documentation | 1 | No explanation of technical terms (EV %, Quarter-Kelly, xG) for non-experts. |
| **Total** | | **26/40** | **Rating: Acceptable** |

## Anti-Patterns Verdict

**Verdict: High-Fidelity Slop.**

**LLM assessment**: The interface is a "Terminal-Native Dark Mode" training-data reflex. While it avoids common SaaS-cream tropes, it leans heavily on the "Hero-Metric Template" (KPI bars with big numbers/tiny labels) and "Identical Card Grids" for both groups and parlays. These are safe choices that miss opportunities for true personality.

**Deterministic scan**: Unavailable. The bundled detector (`detect-antipatterns.mjs`) was missing from the skill directory.

**Manual Evidence**:
- **Side-Stripe Borders (ABSOLUTE BAN)**: Detected in `.val-correct-row` and `.val-wrong-row` (2px colored left borders).
- **Hero-Metric Template**: Used in `.bets-kpi` and `.edges-stat-pill`.
- **Identical Card Grids**: The `.groups-grid` and `.parlay-cards` use repetitive, identical structures that lead to scanning fatigue.

## Overall Impression
The tool feels technically capable and "pro," but it’s dangerously close to becoming a generic "hacker dashboard." It prioritizes density over clarity, making it powerful for experts but intimidating for everyone else. The single biggest opportunity is to **distill the information density** to focus on the most actionable betting insights.

## What's Working
- **Typography Hierarchy**: The combination of **Bebas Neue** (impact), **Barlow Condensed** (density), and **JetBrains Mono** (technicality) is exceptionally well-chosen for a sports analytics tool.
- **Color Strategy**: The "Restrained" tinting and subtle glows (0.04 chroma) create a sense of depth and focus without resorting to cheap glassmorphism.

## Priority Issues

- **[P0] Side-Stripe Borders**: Violates the absolute ban.
  - **Why it matters**: It's a lazy visual shorthand that clutters the interface and signals "AI-generated" or low-effort design.
  - **Fix**: Remove `border-left: 2px` from `.val-correct-row` and `.val-wrong-row`. Use a subtle background tint or a status icon instead.
  - **Suggested command**: `/polish static/index.html`

- **[P1] Information Overload (Memory Bridge)**: The "Betting Sheet" table is too wide (12 columns).
  - **Why it matters**: Users struggle to connect match data on the left with stake values on the right, leading to cognitive strain.
  - **Fix**: Group related columns or use a more card-like structure for individual bets that keeps all relevant data in a tighter visual cluster.
  - **Suggested command**: `/layout static/index.html`

- **[P1] Hero-Metric Cliché**: KPI bars use the banned big-number/small-label template.
  - **Why it matters**: It’s an overused pattern that lacks context and feels like a generic dashboard.
  - **Fix**: Redesign the summary area as a "Statement" or "Status Report" with integrated prose and meaningful hierarchy.
  - **Suggested command**: `/bolder static/index.html`

- **[P2] Missing "Jordan" Scaffolding**: Technical terms are unexplained.
  - **Why it matters**: First-time users (Jordans) will be intimidated by terms like "EV %", "Quarter-Kelly", and "xG".
  - **Fix**: Add a glossary or info-icons with hover definitions to educate the user.
  - **Suggested command**: `/clarify static/index.html`

- **[P3] Grid Interference**: The background grid is too prominent.
  - **Why it matters**: It competes with the actual data grids in tables, making the text feel like it's "vibrating."
  - **Fix**: Reduce opacity of `rgba(198,255,87,0.025)` or remove it to simplify the background.
  - **Suggested command**: `/quieter static/index.html`

## Persona Red Flags

**Alex (Power User)**: The `0.8s` transitions and "floaty" smooth scrolling are too slow for high-frequency analysis. Alex wants instant data, not cinematic transitions.

**Jordan (First-Timer)**: The sheer density of the "Betting Sheet" and the lack of explanations for betting jargon will cause Jordan to abandon the app at the first hurdle.

**Casey (Mobile User)**: The "Group Stage" grid of 12 groups is a vertical scrolling nightmare. Casey needs a "Jump to Group" selector or a more condensed mobile view.

## Minor Observations
- The `.logo span` color (`var(--muted)`) is too low contrast.
- The "Empty State" in the Predictor is cold and doesn't build excitement for the tournament.

## Questions to Consider
- If we rebranded this as a "Tactical Scouting Tool" instead of a "Betting Predictor," what 20% of the UI would feel suddenly absurd?
- Why is the most important number in the app (the predicted score) using a standard scoreboard layout instead of something that reflects model uncertainty?
- What would a version of this look like that focuses *only* on the top 3 most valuable bets of the day?
