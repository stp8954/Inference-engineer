SURF="#fcfcfb"; INK="#0b0b0b"; INK2="#52514e"; MUTED="#9a9992"; RULE="#e3e2de"
BLUE="#2a78d6"; ORANGE="#eb6834"; AQUA="#1baf7a"; DIM="#c9d7ea"
FONT="-apple-system, BlinkMacSystemFont, 'Segoe UI', Inter, Helvetica, Arial, sans-serif"
MONO="ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"

# 8B, BF16, 4K ctx, batch 1.  KV = 4*N*H_kv*D*L = 4*4096*8*128*32 = 0.54 GB
SEGS=[("model weights",16.0,"projections + FFN, re-read every token"),
      ("KV cache",0.54,"grows with context x batch"),
      ("activations",0.05,"negligible")]

# attack points: (label, week, which segment index it shrinks, x-offset-frac within seg)
ATTACKS=[("quantization","W10-11",0,0.30),
         ("GQA / MLA","W13",1,0.5),
         ("sliding window, SSM","W13,19",1,0.5),
         ("FlashAttention","W12",None,None),
         ("PagedAttention","W6",None,None),
         ("continuous batching","W5",None,None),
         ("speculative decoding","W16",None,None),
         ("disaggregation","W17",None,None)]

def build(highlight=None, fname="fig4-anchor.svg"):
    W,H=1000,646
    s=(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
       f'font-family="{FONT}"><rect width="{W}" height="{H}" fill="{SURF}"/>')
    s+=f'<text x="40" y="46" font-size="22" font-weight="600" fill="{INK}">Where the bytes go</text>'
    s+=f'<text x="40" y="72" font-size="15" fill="{INK2}">One decode step: 8B model, BF16, 4K context, batch size 1. Everything the GPU must move to produce one token.</text>'

    X0,BY,BW,BH=40,108,920,54
    total=sum(v for _,v,_ in SEGS); x=X0
    for i,(name,val,note) in enumerate(SEGS):
        w=BW*(val/total)
        active = (highlight is None) or (highlight==i)
        fill = [BLUE,AQUA,MUTED][i] if active else DIM
        s+=f'<rect x="{x:.1f}" y="{BY}" width="{max(w-2,2):.1f}" height="{BH}" rx="4" fill="{fill}"/>'
        if w>150:
            s+=f'<text x="{x+16:.1f}" y="{BY+24}" font-size="15" font-weight="600" fill="#ffffff">{name}</text>'
            s+=f'<text x="{x+16:.1f}" y="{BY+43}" font-size="14" fill="#dbe8f8" font-family="{MONO}">{val:.0f} GB</text>'
        x+=w
    # callouts for the thin segments
    kvx=X0+BW*(SEGS[0][1]/total)
    s+=f'<circle cx="{kvx+14:.1f}" cy="{BY+BH+3}" r="3" fill="{AQUA}"/>'
    s+=f'<path d="M {kvx+14:.1f} {BY+BH+3} L {kvx+14:.1f} {BY+BH+28} L {kvx-30:.1f} {BY+BH+28}" fill="none" stroke="{AQUA}" stroke-width="1.8"/>'
    s+=f'<text x="{kvx-38:.1f}" y="{BY+BH+33}" font-size="14" font-weight="600" text-anchor="end" fill="{AQUA}">KV cache: 0.54 GB at 4K context — but 4.3 GB at 32K</text>' 

    s+=f'<text x="{X0}" y="{BY+BH+66}" font-size="14.5" fill="{INK2}">Total: <tspan font-family="{MONO}">16.6 GB</tspan> moved per token, at ~3.35 TB/s → a <tspan font-weight="600" fill="{INK}">4.9 ms floor</tspan> before any arithmetic happens.</text>'

    # the counterintuitive note
    NY=248
    s+=f'<rect x="{X0}" y="{NY}" width="{BW}" height="62" rx="8" fill="#fdf1ec" stroke="{ORANGE}" stroke-width="1.5"/>'
    s+=f'<text x="{X0+18}" y="{NY+26}" font-size="15" font-weight="600" fill="{INK}">Note what is <tspan font-style="italic">not</tspan> the bottleneck: attention.</text>'
    s+=f'<text x="{X0+18}" y="{NY+47}" font-size="14" fill="{INK2}">Attention is compute-bound at decode. The starvation is weight reloading — which is why so many optimizations target the blue bar.</text>'

    # attack points
    AY=346
    s+=f'<text x="{X0}" y="{AY}" font-size="16" font-weight="600" fill="{INK}">Every optimization in this series attacks one of these</text>'
    rows=[("Shrink the bytes","quantization (W10–11) · GQA and MLA (W13) · sliding window and SSMs (W13, W19)",BLUE),
          ("Move fewer of them twice","FlashAttention keeps attention intermediates out of memory entirely (W12)",AQUA),
          ("Pack them better","PagedAttention — same bytes, far less waste, so more users fit (W6)",AQUA),
          ("Amortize the load","continuous batching across users (W5) · speculative decoding across tokens (W16)",ORANGE),
          ("Split the workload","disaggregated prefill/decode — the two regimes get different GPUs (W17)",ORANGE)]
    for i,(head,body,col) in enumerate(rows):
        y=AY+32+i*42
        s+=f'<rect x="{X0}" y="{y-14}" width="5" height="30" rx="2.5" fill="{col}"/>'
        s+=f'<text x="{X0+18}" y="{y}" font-size="14.5" font-weight="600" fill="{INK}">{head}</text>'
        s+=f'<text x="{X0+18}" y="{y+18}" font-size="13.5" fill="{INK2}">{body}</text>'

    FY=AY+32+len(rows)*42+8
    s+=f'<rect x="{X0}" y="{FY}" width="{BW}" height="1.5" fill="{RULE}"/>'
    s+=f'<text x="{X0}" y="{FY+30}" font-size="14.5" fill="{INK2}">Keep this picture. Every week from here is one of these five moves, and we will come back to this bar to say which.</text>'
    return s+"</svg>"

open("fig4-anchor.svg","w").write(build())
print("anchor written")
