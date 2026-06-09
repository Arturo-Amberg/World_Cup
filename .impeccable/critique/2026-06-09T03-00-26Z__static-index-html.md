---
target: static/index.html
total_score: 22
p0_count: 0
p1_count: 2
timestamp: 2026-06-09T03-00-26Z
slug: static-index-html
---
## Design Health Score

| # | Heuristic | Score | Key Issue |
|---|-----------|-------|-----------|
| 1 | Visibility of System Status | 3 | Content pops in after spinner — no transition |
| 2 | Match System / Real World | 3 | Betting vocabulary accurate; minor jargon exposure |
| 3 | User Control and Freedom | 2 | No reset in predictor; no export |
| 4 | Consistency and Standards | 2 | #007aff intruder in 4 places; two tab systems |
| 5 | Error Prevention | 1 | Odds inputs accept invalid numbers silently |
| 6 | Recognition Rather Than Recall | 3 | Tab labels clear; no inline tooltips for xG/ELO |
| 7 | Flexibility and Efficiency | 3 | 1-5 keyboard shortcuts present |
| 8 | Aesthetic and Minimalist Design | 2 | Card grid overuse; flat tone throughout |
| 9 | Error Recovery | 1 | Fetch errors shown in loader with no retry action |
| 10 | Help and Documentation | 2 | Model page good; no inline contextual help |
| **Total** | | **22/40** | **Needs work** |

## Priority Issues

- [P1] No enter animations — data snaps in
- [P1] #007aff blue not in design token system (4 occurrences)
- [P2] Empty state: opacity:0.3 uppercase text teaches nothing
- [P2] Identical uppercase card-title pattern everywhere flattens hierarchy
- [P3] Loading → content hard cut
