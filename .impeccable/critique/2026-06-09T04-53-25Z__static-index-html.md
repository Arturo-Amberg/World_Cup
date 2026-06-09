---
target: static/index.html
total_score: 29
p0_count: 0
p1_count: 0
timestamp: 2026-06-09T04-53-25Z
slug: static-index-html
---
## Design Health Score

| # | Heuristic | Score | Key Issue |
|---|-----------|-------|-----------|
| 1 | Visibility of System Status | 3 | Loading states solid; group match→predictor navigation lacks transition signal |
| 2 | Match Between System / Real World | 3 | Tooltips added for EV, xG, EDGE, KELLY |
| 3 | User Control and Freedom | 3 | Reset button added; tab switching free; filter "clear all" missing but minor |
| 4 | Consistency and Standards | 3 | #007aff eliminated; #fff hardcodes replaced; VB unified |
| 5 | Error Prevention | 3 | Dropdowns, same-team guard, valid constraints — solid |
| 6 | Recognition Rather Than Recall | 3 | Tooltips on technical terms; keyboard shortcuts still undocumented |
| 7 | Flexibility and Efficiency | 3 | Reset button added for faster loop; no export/copy |
| 8 | Aesthetic and Minimalist Design | 3 | Card titles more readable; VB unified; section spacing still flat |
| 9 | Error Recovery | 2 | Raw error strings in live mode unchanged |
| 10 | Help and Documentation | 3 | Tooltips + Model tab; keyboard shortcuts invisible |
| **Total** | | **29/40** | **Good** |

## Anti-Patterns Verdict

Unified. No AI tells. #007aff fully eliminated. #f59e0b amber on .bk-prob.med is the only remaining hardcode outside tokens.

## Priority Issues

**[P2] Section spacing flat — no rhythm between major sections**
- Every .section has same 48px 40px padding, every .card has 24px margin-bottom
- Fix: Increase separation between Overview's Championship Odds and Group Stage sections (64-80px margin or full-width divider)

**[P2] Keyboard shortcuts undiscoverable**
- 1-5 tab switching and Enter to predict are invisible features
- Fix: Small tooltip on header or single line in page headings

**[P2] API error recovery shows raw strings in live mode**
- Backend failures surface as raw JS error text with no recovery guidance
- Fix: Consistent error state with "try reloading" + retry button

**[P3] #f59e0b amber hardcoded in bracket .bk-prob.med**
- Only remaining hardcode outside token system
- Fix: Replace with var(--gold-bg) or introduce --amber token

## Persona Red Flags

**High-Stakes Bettor**: 14px table text is better; probability bars at 7px still too thin for quick comparative scanning.

**Tournament Analyst**: Overview tab lacks visual landmarks between Championship Odds and Group Stage sections — no pause signal between major content areas.

## Minor Observations

- .score-num 84px may overflow at ~900px; consider clamp(52px, 8vw, 84px)
- VB controls compress on narrow screens; two-row layout on mobile
- bk-groups-grid repeat(6,1fr) breaks on tablets; use auto-fit minmax(140px,1fr)
