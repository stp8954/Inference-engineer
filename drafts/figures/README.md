# Figures

Source of truth is `gen.py` (writes the SVGs); `shot.py` rasterizes them to 2× PNGs
with headless Chromium. Regenerate with `python3 gen.py && python3 shot.py`.

**Upload the PNGs to Substack, not the SVGs** — Substack's image pipeline doesn't
reliably handle SVG. Keep the SVGs for editing and for the GitHub repo.

Palette: dataviz reference light-mode slots 1 (`#2a78d6`) and 2 (`#eb6834`) on
surface `#fcfcfb`; validated (adjacent CVD ΔE 24.7, normal-vision 33.6, both ≥3:1
on surface). Figures carry their own light surface so they read consistently in
Substack's light and dark modes.

## The anchor (Week 3 onward)

`fig4-anchor` — "Where the bytes go." The series' recurring visual; see
[`../../planning/visual-strategy.md`](../../planning/visual-strategy.md) for the rationale.
`anchor.py` exposes `build(highlight=i)` to dim all but one segment, so each post reuses the
same figure with its own attack point lit.

## Week 1

| File | Where it goes | Job |
|---|---|---|
| `fig1-the-loop` | after "The loop" section opener | The mental model: autoregression as a cycle, with the return arrow as the heaviest line on the page. |
| `fig2-time-budget` | inside "The napkin math", right after the 4.8 ms / 16 µs lower bounds | **The screenshot moment.** Peak resource bounds drawn to scale, explicitly not a measured execution timeline. |
| `fig3-bandwidth-ceilings` | in the "one formula travels" bullet list | Shows the idealized weights-only ceiling across exact hardware configurations; sets up quantization as a bandwidth story without promising realized speedup. |

Fig 2 is the one to lead with on X — it is the whole post in one image.

## Beyond static figures

See [`../../planning/visual-strategy.md`](../../planning/visual-strategy.md) for when to reach for an interactive widget or a ManimCE animation instead, and which concepts justify the cost.
