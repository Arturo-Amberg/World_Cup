---
target: static/index.html
total_score: 31
p0_count: 0
p1_count: 2
timestamp: 2026-05-26T18-39-36Z
slug: static-index-html
---
# Design Critique: static/index.html

## Design Health Score
> *Heuristic evaluation based on Nielsen's 10 standards.*

| # | Heuristic | Score | Key Issue |
|---|-----------|-------|-----------|
| 1 | Visibility of System Status | 3 | Sticky headers are great; tab transitions lack micro-interactions. |
| 2 | Match System / Real World | 4 | Professional terminology (EV%, xG, Kelly) is perfectly pitched. |
| 3 | User Control and Freedom | 2 | No "clear filters" action; manual reset required for all selects. |
| 4 | Consistency and Standards | 4 | Strict adherence to the JetBrains Mono / Barlow / Bebas palette. |
| 5 | Error Prevention | 3 | Predictor inputs lack range guards (e.g. odds < 1.01). |
| 6 | Recognition Rather Than Recall | 3 | "Jump to Group" chips and tooltips solve previous memory bridges. |
| 7 | Flexibility and Efficiency | 3 | Excellent filtering, but missing keyboard shortcuts for power users. |
| 8 | Aesthetic and Minimalist Design | 3 | Background grid is neutralized, but tables are nearing "data exhaustion." |
| 9 | Error Recovery | 2 | "No bookie odds" is a passive notice, not a guided recovery path. |
| 10 | Help and Documentation | 4 | Tooltips are technically dense and provide genuine advisory value. |
| **Total** | | **31/40** | **Rating: Good** |

## Anti-Patterns Verdict

**Verdict: Expert-Fidelity Design.**

**LLM assessment**: The interface has successfully escaped the "Generic AI Dashboard" gravity. The integrated KPI layout in the Betting Sheet uses purposeful data density rather than card-padding filler. The "Jump to Group" chips feel like tactical utilities rather than templates. Some "Expert-Slop" remains in the `val-stat-card` and `edges-stat-pill` (big-number/tiny-label pattern), but it's used here with actual analytical intent rather than as decoration.

**Deterministic scan**: Unavailable.

**Manual Evidence**:
- **Side-Stripe Borders (FIXED)**: Confirmed removed from validation tables and team panels.
- **Hero-Metric Template (IMPROVED)**: The Betting Sheet summary is now a typography-first "Statement" area.
- **Data Integrity**: Alignment in the new `table-group-headers` is structurally sound.

## Overall Impression
The overhaul has transformed the tool from a generic prototype into a credible "Tactical Scouting Terminal." It now projects confidence and technical authority. However, it is reaching the limits of horizontal space—the interface is powerful but needs to transition from "showing everything" to "guiding the eye" toward the most profitable insights.

## What's Working
- **Technical Authority**: The combination of `JetBrains Mono` and the high-contrast lime-on-black palette creates an immediate sense of professional precision.
- **Contextual Education**: The tooltip implementation transforms the tool from a calculator into an advisor, likely building significant user trust.

## Priority Issues

- **[P1] Predictor Friction (Manual Input)**: Manual entry of 3-4 bookie odds is a hurdle.
  - **Why it matters**: High-stakes bettors value speed. This interaction is the highest-friction point in the primary analysis workflow.
  - **Fix**: Implement "Auto-fill Fair Odds" or "Copy from Market" buttons to give users a baseline to edit from.
  - **Suggested command**: `/polish static/index.html`

- **[P1] Data Exhaustion (Horizontal Scrutiny)**: The Betting Sheet table (12 columns) is too wide for smaller laptop screens.
  - **Why it matters**: Forces horizontal scrolling or extreme density that breaks readability for the "Alex" (Power User) persona.
  - **Fix**: Consolidate `xG A` and `xG B` into a single "xG Spread" column or use progressive disclosure for "Bet Strategy" details.
  - **Suggested command**: `/distill static/index.html`

- **[P2] Memory Bridge in Revenue Strategy**: The Revenue Strategy table lacks the grouped headers used in the Betting Sheet.
  - **Why it matters**: Parsing "MODEL % (A/X/B)" vs "EDGE (A/X/B)" across rows is mentally taxing.
  - **Fix**: Port the `table-group-headers` pattern from the Betting Sheet to the Revenue Strategy table.
  - **Suggested command**: `/layout static/index.html`

- **[P2] Passive Error/Empty States**: Empty states feel like "dead ends."
  - **Why it matters**: "SELECT TWO TEAMS" is cold and doesn't guide the user toward the next tactical step.
  - **Fix**: Add a "Quick Analyze: Top Value Match" button to the Predictor's empty state.
  - **Suggested command**: `/onboard static/index.html`

## Persona Red Flags

**Alex (Power User)**: Lack of keyboard shortcuts (`/` for search, `Enter` to Analyze, `1-6` for page tabs) prevents the high-speed "Terminal" flow Alex expects.

**Jordan (First-Timer)**: While tooltips explain the *how*, they don't explain the *what next*. Jordan might see a "Negative EV" and not realize it means "DO NOT BET."

**Casey (Tactical Analyst)**: Casey cannot export the "Betting Sheet" to CSV/JSON for external modeling, limiting the tool's utility in a broader analytical stack.

## Minor Observations
- The `backdrop-filter: blur(12px)` on the header is hidden by the 95% opacity of the background.
- `init()` lacks try/catch blocks, leaving the app vulnerable to silent failures if an API is down.

## Questions to Consider
- What if the Predictor didn't require two teams to start, but instead suggested the "Value Match of the Day" upon entry?
- Could the "Score Probability Matrix" (Heatmap) be the *primary* interaction point for the Predictor?
- If this is a "Cold-Professional" tool, should we remove the radial glows entirely for a pure "Raw Data" aesthetic?
