# Design

## Visual Theme
**macOS Light**. A premium, high-fidelity aesthetic inspired by modern system interfaces. Characterized by expansive whitespace, subtle translucency (vibrancy), and generous border radii.

## Color Strategy: Restrained
The palette is built on tinted neutrals with a single primary accent for action and state.

- **Background (Base)**: `oklch(98% 0.005 240)` (Cool-tinted off-white)
- **Surface (Elevated)**: `oklch(100% 0 0)` (Pure white)
- **Primary (Action)**: `oklch(60% 0.18 250)` (Apple Blue - Trust and Precision)
- **Success**: `oklch(75% 0.15 140)` (Soft Green)
- **Danger**: `oklch(65% 0.22 25)` (Soft Red)
- **Value/Warning**: `oklch(85% 0.18 85)` (Soft Gold)
- **Text (Primary)**: `oklch(25% 0.01 240)` (Deep Charcoal)
- **Text (Secondary)**: `oklch(55% 0.01 240)` (Medium Gray)
- **Border**: `oklch(92% 0.005 240)` (Subtle separation)

## Typography
- **UI Stack**: `system-ui, -apple-system, sans-serif`. High legibility, native feel.
- **Monospace**: `ui-monospace, SFMono-Regular, monospace`. Used for data values, ELO scores, and xG metrics.
- **Hierarchy**: Contrast driven by weight (Regular vs. Semibold) and subtle size steps (1.125 ratio).

## Layout & Rhythm
- **Max-width**: `1280px` for the main terminal.
- **Spacing System**: Increments of 8px. Default container padding is 32px.
- **Elevation**: Use soft, diffused shadows (`0 4px 24px rgba(0,0,0,0.04)`) instead of heavy borders for separation.

## Components

### Group Stage Grids
- **Cards**: Minimal containers with probability chips.
- **Progress Bars**: Slender, rounded bars for probability distribution.

### Betting Terminal (Value Bets)
- **Data Tables**: Clean, border-less rows with hover states.
- **Heatmaps**: Subtle background tints on EV cells using Success/Danger tones at low chroma.
- **KPI Blocks**: Large, bold metrics with muted labels for quick scanning.

### Tournament Brackets
- **Connection Lines**: Thin, 1px lines using the Border color.
- **Nodes**: Responsive cards showing team flags (placeholder shapes) and advancement %.
- **Interactivity**: Smooth transitions when hovering over a team's path.

## Motion
- **Duration**: 200ms default for state changes.
- **Easing**: `cubic-bezier(0.2, 0, 0, 1)` (Ease-out-expo) for a premium, responsive feel.
- **State**: Motion should only be used to convey state transitions (e.g. tab switching, row expansion).
