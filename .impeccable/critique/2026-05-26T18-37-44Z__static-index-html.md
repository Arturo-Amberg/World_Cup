---
target: static/index.html
total_score: 31
p0_count: 0
p1_count: 2
timestamp: 2026-05-26T18-37-44Z
slug: static-index-html
---
# Critique: static/index.html
**Score: 31/40**
**Date: 2026-05-26**

## AI Slop Verdict
The interface has moved significantly away from the "Generic Dashboard" reflex. The integrated KPI layout in the Betting Sheet uses purposeful data density rather than card-padding filler. The "Jump to Group" navigation (the sticky group-nav-chip bar) feels like a tactical utility rather than a templated component. However, the reliance on the "Big Number + Small Label" pattern in the `val-stat-card` and `edges-stat-pill` still carries a faint SaaS-template scent. It PD-Expert-Slop — a high-fidelity version of a common pattern, but one that is used here with actual analytical intent.

## Heuristics Table

| # | Heuristic | Score | Key Issue |
|---|-----------|-------|-----------|
| 1 | Visibility of System Status | 3 | Sticky headers provide great context, but tab transitions lack micro-interactions. |
| 2 | Match System / Real World | 4 | Expert terminology (EV%, xG, Kelly) perfectly matches the "Tactical" target. |
| 3 | User Control and Freedom | 2 | No "clear filters" global action; users must reset individual selects. |
| 4 | Consistency and Standards | 4 | Rigorous adherence to the JetBrains Mono / Barlow / Bebas pairing. |
| 5 | Error Prevention | 3 | Select-based navigation prevents invalid state, but input[number] lacks range guards. |
| 6 | Recognition Rather Than Recall | 3 | "Jump to Group" chips are excellent; tooltips solve the memory bridge for formulas. |
| 7 | Flexibility and Efficiency | 3 | High density is good for experts, but missing keyboard shortcuts for tab navigation. |
| 8 | Aesthetic and Minimalist Design | 3 | Background grid is subtle, but some tables (Bets) are nearing data-exhaustion. |
| 9 | Error Recovery | 2 | "No bookie odds entered" is a passive notice, not a guided recovery. |
| 10 | Help and Documentation | 4 | Tooltips are technically dense, task-focused, and educational. |
| **Total** | | **31/40** | **Good (Rating: Good)** |

## Cognitive Load Assessment
The redesigned "Betting Sheet" table is a significant improvement. The `table-group-headers` ("MATCH INFO", "MODEL ANALYTICS", "MARKET", "BET STRATEGY") act as the primary cognitive anchors, effectively resolving the "Memory Bridge" issue by grouping related metrics. The use of the `grp-badge` and `bet-market-chip` provides rapid visual scanning. However, the "Model % (A/X/B)" in the Revenue Strategy page is still a high-load element (3 distinct numbers in one cell), requiring the user to mentally parse a slash-separated string.

## Emotional Journey Map
1. **Entry (Groups)**: Confusion at scale (12 groups) → Relief (Sticky nav chips). *Feel: Overwhelmed then Capable.*
2. **Analysis (Predictor)**: High friction (Manual odds entry) → Reward (Scoreline probability & heatmap). *Feel: Tedious then Enlightened.*
3. **Execution (Bets)**: High trust (KPIs, EV%, Parlay logic). *Feel: Strategic & Empowered.*
4. **End (Validation)**: Security (Model transparency). *Feel: Confident & Technical.*

## Strengths
1. **Technical Authority**: The use of `JetBrains Mono` and the high-contrast "Tactical" palette (`#c6ff57` on `#06090f`) creates an immediate sense of professional precision.
2. **Contextual Education**: The tooltip implementation (e.g., "Quarter-Kelly Sizing") transforms the tool from a calculator into an advisor, building long-term user loyalty.

## Priority Issues
1. **[P1] What**: Predictor Friction (Manual Input).
   **Why**: High-stakes bettors value speed. Manually typing 3-4 odds for every match analysis is a significant cognitive and physical hurdle.
   **Fix**: Implement "Auto-fill Fair Odds" or "Copy from Market" buttons in the Predictor panel.
   **Suggested Command**: `impeccable polish predictor`
2. **[P1] What**: Data Exhaustion in `bets-table`.
   **Why**: The 12-column table is too wide for standard 13" laptops, forcing horizontal scrolling or extreme density that breaks readability.
   **Fix**: Consolidate `xG A` and `xG B` into a single "xG Spread" column or use a progressive disclosure pattern for "Bet Strategy" details.
   **Suggested Command**: `impeccable distill bets`
3. **[P2] What**: Passive Error/Empty States.
   **Why**: The "SELECT TWO TEAMS" and "NO BOOKIE ODDS ENTERED" states feel like dead ends rather than invitations to act.
   **Fix**: Add "Quick Analyze: Next Match" or "Load Example" buttons to empty states.
   **Suggested Command**: `impeccable onboard predictor`
4. **[P2] What**: The "Memory Bridge" in Revenue Strategy.
   **Why**: Unlike the Betting Sheet, the Revenue Strategy table lacks grouped headers, making the "MODEL % (A/X/B)" vs "EDGE (A/X/B)" comparison difficult to track across rows.
   **Fix**: Port the `table-group-headers` pattern from the Betting Sheet to the Revenue Strategy table.
   **Suggested Command**: `impeccable layout edges`

## Persona Red Flags
- **Alex (Power User)**: Lack of keyboard shortcuts (`/` for search, `Enter` to Analyze, `1-6` for page tabs) prevents the "Terminal" flow they expect.
- **Jordan (First-Timer)**: The "Kelly Sizing" and "xG" concepts are explained in tooltips, but the *implication* of a "Negative EV" is never explicitly stated as "DO NOT BET."
- **Casey (Tactical Analyst)**: No way to export the "Betting Sheet" or "Tournament Odds" to CSV/JSON for external modeling.

## Minor Observations
- The `backdrop-filter: blur(12px)` on the header is a nice touch, but the `rgba(6,9,15,0.95)` is so opaque that the blur effect is barely visible.
- The `host-pip` in the Group Stage is a great detail for home-advantage analysts.
- The `score-display` font-size (80px) is slightly aggressive, potentially causing layout shifts on smaller screens.

## Questions to Consider
1. What if the Predictor didn"t require two teams to start, but instead suggested the "Value Match of the Day" upon entry?
2. If this is a "Cold-Professional" tool, should we remove the background grid and radial glows entirely to achieve a pure, raw data aesthetic?
3. Could the "Score Probability Matrix" (Heatmap) be the *primary* interaction point for the Predictor, allowing users to hover cells to see odds comparisons?
