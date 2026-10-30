"""Render run/metrics.jsonl as a dependency-free SVG of validation loss over training time.
usage: python tools/plot.py [run/metrics.jsonl] [docs/training.svg]
"""
import json, sys

src = sys.argv[1] if len(sys.argv) > 1 else "run/metrics.jsonl"
dst = sys.argv[2] if len(sys.argv) > 2 else "docs/training.svg"
rows = [json.loads(l) for l in open(src)]
series = [("chat", "#2563eb"), ("stories", "#16a34a"), ("code", "#d97706"), ("codeqa", "#9333ea")]
W, H, L, R, T, B = 760, 380, 60, 130, 20, 50
xs = [r["t"] / 3600 for r in rows]
vals = [r[k] for r in rows for k, _ in series]
x0, x1, y0, y1 = 0, max(xs) or 1, min(vals) - 0.1, min(max(vals), 6.0)
px = lambda x: L + (x - x0) / (x1 - x0) * (W - L - R)
py = lambda y: T + (1 - (min(y, y1) - y0) / (y1 - y0)) * (H - T - B)
o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="system-ui,sans-serif" font-size="12">',
     f'<rect width="{W}" height="{H}" fill="white"/>']
for i in range(6):
    y = y0 + (y1 - y0) * i / 5
    o.append(f'<line x1="{L}" x2="{W-R}" y1="{py(y):.1f}" y2="{py(y):.1f}" stroke="#e5e7eb"/>'
             f'<text x="{L-8}" y="{py(y)+4:.1f}" text-anchor="end" fill="#374151">{y:.1f}</text>')
for i in range(int(x1) + 1):
    o.append(f'<text x="{px(i):.1f}" y="{H-B+18}" text-anchor="middle" fill="#374151">{i}h</text>')
o.append(f'<text x="{(L+W-R)/2}" y="{H-8}" text-anchor="middle" fill="#374151">training time</text>')
o.append(f'<text x="14" y="{H/2}" transform="rotate(-90 14 {H/2})" text-anchor="middle" fill="#374151">validation loss (nats/token)</text>')
for n, (k, c) in enumerate(series):
    pts = " ".join(f"{px(r['t']/3600):.1f},{py(r[k]):.1f}" for r in rows)
    o.append(f'<polyline fill="none" stroke="{c}" stroke-width="2" points="{pts}"/>')
    o.append(f'<rect x="{W-R+14}" y="{T+10+n*22}" width="12" height="12" fill="{c}"/>'
             f'<text x="{W-R+32}" y="{T+20+n*22}" fill="#111827">{k}</text>')
o.append("</svg>")
open(dst, "w").write("\n".join(o))
print("wrote", dst)
