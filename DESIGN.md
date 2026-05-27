# Design

## Visual Theme
macOS Light. A bright, airy, and premium aesthetic characterized by large border radii, soft shadows, and translucent vibrancy effects.

## Color Palette
- **Background**: `#f5f5f7` (macOS Base), `#ffffff` (Surface/Card)
- **Primary**: `#34c759` (Apple Green - Success)
- **Secondary**: `#ff3b30` (Apple Red - Risk)
- **Tertiary**: `#ffcc00` (Apple Gold - Value)
- **Text**: `#1d1d1f` (Primary), `#86868b` (Secondary/Muted)
- **Border**: `#e5e5ea`

## Typography
- **UI Stack**: `system-ui` (SF Pro on Apple devices)
- **Monospace**: `ui-monospace` (SF Mono)
- **Scale**: Focus on weight-based hierarchy (Semibold/Bold) rather than size-only or tracking-heavy uppercase.

## Layout
- Max-width: `1280px`
- Component Spacing: Generous padding (24px-32px) and open whitespace.

## Components
- **Cards**: Pure white background with 16px border-radius and soft `0 4px 24px rgba(0,0,0,0.04)` shadows.
- **Headers**: Translucent `backdrop-filter: blur(20px)` effects.
- **Tables**: Clean, row-hover highlighted, using system typography.
