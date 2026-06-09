---
target: parlay builder
total_score: 32
p0_count: 0
p1_count: 1
timestamp: 2026-06-09T07-19-55Z
slug: static-index-html
---
#### Design Health Score
> *Consult heuristics-scoring*

| # | Heuristic | Score | Key Issue |
|---|-----------|-------|-----------|
| 1 | Visibility of System Status | 3 | Total Cuota updates, but the visual feedback when clicking "Add" is minimal. |
| 2 | Match System / Real World | 3 | Parlay builder uses common betting terminology, but "Parlay Builder" header feels generic. |
| 3 | User Control and Freedom | 4 | Easy removal of individual legs with the "✕" button. |
| 4 | Consistency and Standards | 2 | The builder's dark background clashes with the "macOS Light" theme in DESIGN.md. |
| 5 | Error Prevention | 3 | Users can only click valid odds. |
| 6 | Recognition Rather Than Recall | 4 | The builder keeps selected legs visible at all times. |
| 7 | Flexibility and Efficiency | 4 | 1-click add is very efficient. |
| 8 | Aesthetic and Minimalist Design | 2 | The widget feels cramped and visually heavy for a "premium, airy" app. |
| 9 | Error Recovery | 4 | One-click remove works perfectly. |
| 10 | Help and Documentation | 3 | Self-explanatory, but could use a better empty state. |
| **Total** | | **32/40** | **Solid but Unpolished** |

#### Anti-Patterns Verdict

**LLM assessment**: The new parlay builder feels functional but lacks the high-end Apple-native polish described in DESIGN.md (`macOS Light` visual theme, expansive whitespace, subtle translucency). It currently feels like a dark-mode element injected into a light-mode page. It's structurally sound but visually heavy and cramped.

**Deterministic scan**: Deterministic scan unavailable (detector not found).

#### Overall Impression
The interactive behavior is a massive UX upgrade over the static "Mejores parlays" cards, but the visual execution of the floating widget feels heavy and out of place against the light, airy macOS aesthetic defined for this project.

#### What's Working
- **Interaction Model**: Clicking table odds to instantly populate a sticky widget is the correct pattern for a betting terminal.
- **State Management**: The dynamic Total Cuota multiplier and leg removal logic are robust and keep the user in control.

#### Priority Issues

- **[P1] Visual Theme Mismatch**
  - **Why it matters**: The dark background and tight borders break the premium "native Apple application" feel and draw too much heavy visual weight.
  - **Fix**: Apply a light, slightly translucent background (`backdrop-filter: blur(20px)`) with a soft shadow and increased padding to match the macOS aesthetic.
  - **Suggested command**: `/impeccable layout`

- **[P2] Cramped Typography & Spacing**
  - **Why it matters**: "Precision over Decoration" requires clear typography. Cramped text inside the legs reduces legibility.
  - **Fix**: Increase font size slightly, remove the heavy borders between legs, and use a cleaner edge-to-edge layout for the selected bets list.
  - **Suggested command**: `/impeccable typeset`

- **[P2] Lack of "Delight" in Empty/Full States**
  - **Why it matters**: A premium betting instrument should make the "Total Cuota" feel like a focal point of value.
  - **Fix**: Elevate the typography of the total odds (e.g., using a larger monospace font) and give the header a cleaner look.
  - **Suggested command**: `/impeccable delight`

#### Persona Red Flags

**High-Stakes Bettors**: The total multiplier ("Total Cuota") doesn't stand out enough visually from the rest of the text. They need to see their edge at a glance.

**Tournament Enthusiasts**: The generic "Parlay Builder" header doesn't explain what happens when the builder is empty. A helpful empty state ("Select odds to build your parlay") would guide them better than just showing an empty box.

#### Minor Observations
- The "✕" remove button feels generic. A cleaner icon or softer text button would fit the premium aesthetic better.
- The transition when the builder opens/closes is functional but could use an Apple-like spring or smoother ease.

#### Questions to Consider
- Does the builder need to be dark to stand out, or should it use an Apple-style translucent glass (`backdrop-filter: blur()`) to float lightly above the content?
- How should the "Total Cuota" be emphasized? Should it use the primary Apple Blue color to signal "Action"?
