# UI Polish Plan

**Date**: 2026-01-12
**Goal**: Make the GeneratePanel V2 feel modern, readable, and pleasant

## Current Layout (Before)

```
+------------------------------------------------------------------+
| [Diagnostics Banner - when visible]                               |
+------------------------------------------------------------------+
| +--------------------------------------------------------------+ |
| | Mode: (o)Artist (o)History (o)Seeds           [gray frame]   | |
| +--------------------------------------------------------------+ |
|                                                                  |
| Cohesion: Balanced    | Length: [30]  | [x]Exclude recent: [14] |
| [====o=======]        |               |    days if played [1]+  |
| Tight Balanced Wide.. |               |                         |
|                       | Artist spacing: [Normal]                |
|                                                                  |
| +--------------------------------------------------------------+ |
| | Artist: [___________________________]                        | |
| | Presence: [Medium] Variety: [===o===] Balanced               | |
| +--------------------------------------------------------------+ |
|                                                                  |
| [Generate] [Regenerate] [New Seeds]              [Cancel]        |
|                                                                  |
| [=============== Progress Bar ===============] Stage label       |
+------------------------------------------------------------------+
|                                                                  |
|                     TRACK TABLE                                  |
|                     (fills space)                                |
|                                                                  |
+------------------------------------------------------------------+
|                    [Export Local] [Export Plex]                  |
+==================================================================+
| LOGS DOCK (tall, visually dominant)                              |
|                                                                  |
+==================================================================+
```

**Problems**:
1. Mode selector in isolated gray box - looks disconnected
2. Global controls scattered horizontally with ugly vertical separators
3. Mode-specific panel floats with too much padding
4. Action buttons on separate row - wastes vertical space
5. Progress bar takes full row
6. Logs dock is too tall by default
7. No visual hierarchy - everything same weight
8. Light gray theme lacks contrast

## Target Layout (After)

```
+------------------------------------------------------------------+
| [Banner - collapsed by default]                                   |
+==================================================================+
| HEADER BAR (compact, toolbar-like)                               |
|                                                                  |
| Mode: [Artist|History|Seeds]  Cohesion:[===o] Length:[30]        |
|       ^^segmented control^^   Recency:[x] 14d 1+ | Spacing:[Norm]|
|                                         [Generate][Regen][New]   |
+------------------------------------------------------------------+
| MODE INPUTS CARD                                                 |
| +--------------------------------------------------------------+ |
| | Artist: [_______________________]  Presence:[Med] Var:[===]  | |
| +--------------------------------------------------------------+ |
| Progress: [===========] Bridging pools...                        |
+------------------------------------------------------------------+
|                                                                  |
|                                                                  |
|                     TRACK TABLE                                  |
|                  (dominates screen)                              |
|                                                                  |
|                                                                  |
+------------------------------------------------------------------+
|                               [Export Local] [Export to Plex]    |
+~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~+
| Logs (collapsible, small default height ~100px)                  |
+~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~+
```

## Layout Changes

### 1. Consolidated Header Bar
- Single horizontal toolbar-like row containing:
  - Mode selector (styled as segmented control, not radio buttons)
  - Cohesion dial (compact inline version)
  - Length dropdown
  - Recency toggle + spinboxes (compact)
  - Artist spacing dropdown
  - Action buttons (Generate primary, others secondary)

### 2. Mode Inputs Card
- Clean card with subtle background
- All mode-specific controls on 1-2 rows max
- Inline progress bar at bottom of card

### 3. Track Table Dominance
- Use QSplitter between controls+table and logs
- Table gets stretch=1, controls get stretch=0
- Minimum height for logs (100px), remembers user resize

### 4. Collapsible Logs
- Move from dock to inline QSplitter section
- Default height: 100-120px
- User can drag to resize

## Files to Modify

| File | Changes |
|------|---------|
| `main_window.py` | Add QSplitter, load theme, restructure central widget |
| `generate_panel.py` | Consolidate header row, inline progress, compact layout |
| `cohesion_dial.py` | Add compact/inline mode option |
| `mode_panels.py` | Tighter layouts, reduce spacing |
| `theme.qss` | NEW - Dark Fusion palette styles |

## Theme / QSS Decisions

### Color Palette (Dark Fusion-inspired)
```
Background:     #2d2d2d (dark gray)
Surface:        #3d3d3d (cards/panels)
Border:         #4d4d4d (subtle)
Text Primary:   #e0e0e0 (light gray)
Text Secondary: #a0a0a0 (muted)
Accent:         #4a9eff (blue - primary actions)
Accent Hover:   #6ab0ff
Success:        #4caf50 (green)
Warning:        #ff9800 (orange)
Error:          #f44336 (red)
```

### Button Hierarchy
- **Primary** (Generate): Accent blue, bold, larger
- **Secondary** (Regenerate, New Seeds): Surface color, subtle border
- **Danger** (Cancel): Muted red, only prominent when enabled
- **Export**: Plex gold (#e5a00d), Local gray

### Typography
- Base font: 10pt (Qt default is often 9pt)
- Labels: Regular weight
- Headers: Semi-bold
- Values: Slightly larger or accent color

## Implementation Order

### Commit 1: Layout Restructure
1. Modify `generate_panel.py`:
   - Consolidate header into single row
   - Move action buttons to header
   - Inline progress bar below mode inputs
2. Modify `main_window.py`:
   - Replace log dock with inline QSplitter
   - Set Fusion style
3. Test functionality unchanged

### Commit 2: Theme + QSS
1. Create `src/playlist_gui/theme.qss`
2. Load in `main_window.py`
3. Apply dark palette via QPalette
4. Style buttons, inputs, cards

### Commit 3: Final Polish
1. Fine-tune spacing/margins
2. Add hover/focus states
3. Ensure log panel remembers size
4. Fix any visual bugs discovered

## Acceptance Criteria

- [ ] Controls read top-to-bottom in clear order
- [ ] Mode inputs grouped in visible card
- [ ] Track table dominates screen (>60% height)
- [ ] Logs accessible but not dominant (~15% height default)
- [ ] Primary button (Generate) clearly distinguished
- [ ] No startup log spam
- [ ] Dark theme with good contrast
