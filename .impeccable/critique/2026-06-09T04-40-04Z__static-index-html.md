---
target: static/index.html
total_score: 26
p0_count: 0
p1_count: 1
timestamp: 2026-06-09T04-40-04Z
slug: static-index-html
---
## Design Health Score

| # | Heuristic | Score | Key Issue |
|---|-----------|-------|-----------|
| 1 | Visibility of System Status | 3 | Loading states and active tabs are solid; clicking a group match card navigates to the predictor with no visible transition feedback |
| 2 | Match Between System / Real World | 3 | ELO, xG, and EV are domain-standard for this audience; no tooltips for users not fluent in those terms |
| 3 | User Control and Freedom | 2 | No "reset / new prediction" CTA after results appear; Value Bets filters have no "clear all"; predictor selections stick after analysis |
| 4 | Consistency and Standards | 2 | Value Bets section uses #007aff (iOS system blue) as its accent while the entire rest of the app uses --lime (green); several #fff hardcodes outside the token system |
| 5 | Error Prevention | 3 | Dropdowns prevent invalid input; same-team guard fires a notification before submit; solid |
| 6 | Recognition Rather Than Recall | 3 | All options visible; no contextual tooltips on EV/xG/ELO data cells for users who'd benefit from a definition |
| 7 | Flexibility and Efficiency | 3 | Enter key triggers prediction, digit keys switch tabs. No export or copy-to-clipboard on results |
| 8 | Aesthetic and Minimalist Design | 3 | Clean overall; uniform uppercase + letter-spacing card-title pattern across every section flattens hierarchy |
| 9 | Error Recovery | 2 | Notification for invalid selection is clear; API failures surface raw error strings |
| 10 | Help and Documentation | 2 | Model tab explains methodology; no contextual inline help at point of need |
| **Total** | | **26/40** | **Acceptable** |

## Anti-Patterns Verdict

Not AI-generated. The macOS aesthetic is specific and committed. Main issue: Value Bets section uses different design language (#007aff blue, #fff hardcodes, 12px font) — looks pasted in from a different project.

## Priority Issues

**[P1] Value Bets section uses a different design language**
- #007aff blue in 4 locations, #fff hardcodes in 5+ locations, font-size 12px vs app's 14px
- Fix: Replace #007aff with var(--lime), #fff with var(--bg-card), align font sizes

**[P2] No "new prediction" reset path after results**
- No clear CTA to start another prediction after results render
- Fix: Add a "New prediction" reset link near results header

**[P2] .empty-state-text double-dimmed contrast failure**
- color: var(--txt-3) + opacity: 0.3 = ~1.5:1 contrast ratio, fails WCAG AA
- Fix: Remove opacity: 0.3 from .empty-state-text

**[P2] No tooltips on EV, xG, ELO**
- Terms undefined for non-expert users
- Fix: Add title="" tooltips or info icons on first use per page

**[P3] Uniform card-title pattern flattens hierarchy**
- Same 12px uppercase pattern on every card title in every section
- Fix: Reserve uppercase labels for metadata; headings need weight/size differentiation

## Persona Red Flags

**High-Stakes Bettor**: VB table at 12px requires squinting; keyboard shortcut to VB not documented; focus ring color mismatch on keyboard nav.

**Tournament Analyst**: No transition feedback when clicking group match to predictor; no path back to source group after predicting; ELO not shown in group cards for cross-reference.

## Minor Observations

- .score-num at 84px may overflow at ~900px before responsive breakpoint
- VB probability bar track at 5px height is hard to read; 7-8px better
- Notification toast 0.4s transition slower than app's 0.15-0.2s standard
- Bracket page needs landscape orientation hint on mobile
- BETA pill in header: confirm if still accurate or remove
