# Palette (validated light-mode slots 1 & 2 + text roles)
SURF="#fcfcfb"; INK="#0b0b0b"; INK2="#52514e"; MUTED="#8a8985"
BLUE="#2a78d6"; ORANGE="#eb6834"; RULE="#e3e2de"
FONT="-apple-system, BlinkMacSystemFont, 'Segoe UI', Inter, Helvetica, Arial, sans-serif"
MONO="ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"

def head(w,h):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
            f'font-family="{FONT}">\n<rect width="{w}" height="{h}" fill="{SURF}"/>\n')

# ─────────────────────────────────────────────────────────────
# FIG 1 — the autoregressive loop
# ─────────────────────────────────────────────────────────────
W,H=1000,420
s=head(W,H)
s+=f'<text x="40" y="46" font-size="21" font-weight="600" fill="{INK}">The loop</text>'
s+=f'<text x="40" y="72" font-size="15" fill="{INK2}">Everything we call an "inference engine" is orchestration wrapped around this cycle.</text>'

boxes=[(40,"tokens so far","the whole sequence,\nevery time"),
       (280,"forward pass","32–80 transformer\nlayers"),
       (520,"logits","one score per token\nin the vocabulary"),
       (760,"sample","temperature,\ntop-p → one token")]
BY,BW,BH=120,200,96
for i,(x,title,sub) in enumerate(boxes):
    fill = "#eef4fc" if i==1 else "#ffffff"
    stroke = BLUE if i==1 else RULE
    sw = 2 if i==1 else 1.5
    s+=f'<rect x="{x}" y="{BY}" width="{BW}" height="{BH}" rx="10" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'
    s+=f'<text x="{x+BW/2}" y="{BY+37}" font-size="16" font-weight="600" text-anchor="middle" fill="{INK}">{title}</text>'
    for j,line in enumerate(sub.split("\n")):
        s+=f'<text x="{x+BW/2}" y="{BY+59+j*17}" font-size="12.5" text-anchor="middle" fill="{INK2}">{line}</text>'
    if i<3:
        ax=x+BW+8; s+=f'<path d="M {ax} {BY+BH/2} L {ax+24} {BY+BH/2}" stroke="{MUTED}" stroke-width="2" marker-end="url(#a1)"/>'

# return arc — deliberately the heaviest line on the page
s+=(f'<path d="M 860 {BY+BH} L 860 300 Q 860 320 840 320 L 160 320 Q 140 320 140 300 L 140 {BY+BH}" '
    f'fill="none" stroke="{ORANGE}" stroke-width="3.5" marker-end="url(#a2)"/>')
s+=f'<rect x="392" y="304" width="216" height="30" rx="6" fill="{SURF}"/>'
s+=f'<text x="500" y="325" font-size="14.5" font-weight="600" text-anchor="middle" fill="{ORANGE}">append it, and do it all again</text>'

s+=f'<text x="40" y="378" font-size="13.5" fill="{INK2}">No plan. No lookahead. No revision. A 2,000-word essay is 2,000+ passes through this loop,</text>'
s+=f'<text x="40" y="398" font-size="13.5" fill="{INK2}">each one conditioned only on what has already been written.</text>'

s+=(f'<defs><marker id="a1" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
    f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{MUTED}"/></marker>'
    f'<marker id="a2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">'
    f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{ORANGE}"/></marker></defs>')
open("fig1-the-loop.svg","w").write(s+"</svg>")

# ─────────────────────────────────────────────────────────────
# FIG 2 — the decode-step time budget (the screenshot moment)
# ─────────────────────────────────────────────────────────────
W,H=1000,398
s=head(W,H)
s+=f'<text x="40" y="46" font-size="21" font-weight="600" fill="{INK}">One decode step, drawn to scale</text>'
s+=f'<text x="40" y="72" font-size="15" fill="{INK2}">8B model, BF16, batch size 1, H100 SXM. The bar is one token.</text>'

X0,BY,BW,BH=40,120,920,66
s+=f'<rect x="{X0}" y="{BY}" width="{BW}" height="{BH}" rx="5" fill="{BLUE}"/>'
# compute slice: 16 us / 4816 us = 0.33% -> 3.05px. Draw at true scale.
cw=BW*(16/4816)
s+=f'<rect x="{X0+BW-cw}" y="{BY}" width="{cw:.2f}" height="{BH}" fill="{ORANGE}"/>'
s+=f'<text x="{X0+18}" y="{BY+30}" font-size="16" font-weight="600" fill="#ffffff">moving 16 GB of weights from memory</text>'
s+=f'<text x="{X0+18}" y="{BY+52}" font-size="15" fill="#dbe8f8" font-family="{MONO}">4,800 µs  ·  99.7% of the step</text>'

# leader from the hairline to a callout
LX=X0+BW-cw+1.5
s+=f'<path d="M {LX} {BY+BH+6} L {LX} {BY+BH+38} L {LX-150} {BY+BH+38}" fill="none" stroke="{ORANGE}" stroke-width="2"/>'
s+=f'<circle cx="{LX}" cy="{BY+BH+6}" r="3" fill="{ORANGE}"/>'
s+=f'<text x="{LX-158}" y="{BY+BH+43}" font-size="15" font-weight="600" text-anchor="end" fill="{ORANGE}">doing the actual math: 16 µs</text>'
s+=f'<text x="{LX-158}" y="{BY+BH+63}" font-size="13.5" text-anchor="end" fill="{INK2}">16 GFLOPs on a chip rated for 989 TFLOPS</text>'

# the punchline block
PY=272
s+=f'<rect x="{X0}" y="{PY}" width="{BW}" height="1.5" fill="{RULE}"/>'
s+=f'<text x="{X0}" y="{PY+40}" font-size="19" font-weight="600" fill="{INK}">The orange sliver is not a rendering artifact. That is the compute.</text>'
s+=f'<text x="{X0}" y="{PY+68}" font-size="15" fill="{INK2}">Generating a token is a memory-movement problem wearing a matrix-multiplication costume.</text>'
s+=f'<text x="{X0}" y="{PY+90}" font-size="15" fill="{INK2}">Every optimization in this series is a scheme to get more work out of that same 4.8 milliseconds.</text>'
open("fig2-time-budget.svg","w").write(s+"</svg>")

# ─────────────────────────────────────────────────────────────
# FIG 3 — bandwidth ceilings across hardware
# ─────────────────────────────────────────────────────────────
ROWS_MARK=1
rows=[("MacBook Air, M-series","~100 GB/s",6.3),
      ("MacBook Pro, M3 Ultra","~819 GB/s",51.2),
      ("RTX 4090","~1,008 GB/s",63.0),
      ("H100 SXM","3,350 GB/s",209.4)]
W,H=1000,510
s=head(W,H)
s+=f'<text x="40" y="46" font-size="21" font-weight="600" fill="{INK}">One formula, four machines</text>'
s+=f'<text x="40" y="72" font-size="15" fill="{INK2}">Predicted decode ceiling for an 8B model at BF16 (16 GB of weights), batch size 1 — memory bandwidth ÷ weight bytes.</text>'

X0,TOP,ROWH,LW=40,118,68,250
maxv=max(r[2] for r in rows); PW=W-X0-LW-150
for i,(name,bw,val) in enumerate(rows):
    y=TOP+i*ROWH
    s+=f'<text x="{X0}" y="{y+26}" font-size="15" font-weight="600" fill="{INK}">{name}</text>'
    s+=f'<text x="{X0}" y="{y+46}" font-size="13" fill="{INK2}" font-family="{MONO}">{bw}</text>'
    bw_px=max(3,PW*(val/maxv))
    s+=f'<rect x="{X0+LW}" y="{y+8}" width="{bw_px:.1f}" height="34" rx="4" fill="{BLUE}"/>'
    s+=f'<text x="{X0+LW+bw_px+12}" y="{y+32}" font-size="16" font-weight="600" fill="{INK}">{val:.0f} tok/s</text>'

FY=TOP+len(rows)*ROWH+16
s+=f'<rect x="{X0}" y="{FY}" width="{W-2*X0}" height="1.5" fill="{RULE}"/>'
s+=f'<text x="{X0}" y="{FY+36}" font-size="15" fill="{INK2}">These are ceilings, not benchmarks — no real engine beats them, and good ones get close.</text>'
s+=f'<text x="{X0}" y="{FY+58}" font-size="15" fill="{INK2}">Quantize the same model to 4-bit and every number roughly quadruples, because you moved fewer bytes,</text>'
s+=f'<text x="{X0}" y="{FY+80}" font-size="15" fill="{INK2}">not because you found more compute.</text>'
open("fig3-bandwidth-ceilings.svg","w").write(s+"</svg>")
print("svgs written")
