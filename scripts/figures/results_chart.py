#!/usr/bin/env python3
"""README results chart: every held-out trial's official score, grouped by target.

  results_chart.py benchmarks/heldout_results.json docs/assets/heldout
writes docs/assets/heldout-light.svg and docs/assets/heldout-dark.svg.
"""
import argparse
import json
from html import escape
from pathlib import Path

from svg_chart import THEMES, svg, text

GROUPS = (('sfp_0', 'SFP port 0'), ('sfp_1', 'SFP port 1'), ('sc_', 'SC port'))
OUTCOMES = (('full_insertion', 'good', 'full insertion', 'full'), ('partial_insertion', 'warning', 'partial', 'partial'),
            ('proximity', 'serious', 'proximity only', 'proximity'), ('no_insertion', 'critical', 'no insertion', 'none'))


def bar(x, y, w, h, fill, title):
    """A bar anchored to the baseline with a 4 px rounded data end."""
    r = min(4, h/2, w/2)
    path = (f'M{x:.1f},{y+h:.1f} V{y+r:.1f} Q{x:.1f},{y:.1f} {x+r:.1f},{y:.1f} H{x+w-r:.1f} '
            f'Q{x+w:.1f},{y:.1f} {x+w:.1f},{y+r:.1f} V{y+h:.1f} Z')
    return f'<path d="{path}" fill="{fill}"><title>{escape(title)}</title></path>'


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('results', type=Path)
    ap.add_argument('output', type=Path, help='prefix; -light.svg and -dark.svg are appended')
    a = ap.parse_args()
    data = json.loads(a.results.read_text())
    trials = data['trials']
    status = {key: (tone, label, short) for key, tone, label, short in OUTCOMES}
    groups = [(name, [t for t in trials if t['trial'].rsplit('repeat_0_', 1)[1].startswith(key)])
              for key, name in GROUPS]
    assert sum(len(g) for _, g in groups) == len(trials)
    full = sum(t['outcome'] == 'full_insertion' for t in trials)

    W, left, right, top = 920, 52, 16, 114
    plot_h, group_gap, bar_gap = 220, 56, 12
    plot_w = W-left-right
    group_w = (plot_w-group_gap*(len(groups)-1))/len(groups)
    base = top+plot_h
    y = lambda v: base-plot_h*v/100
    for theme, c in THEMES.items():
        body = [text(left, 30, f'Held-out set, run once: {data["official_total"]:.1f} / {100*len(trials)}', 17,
                     c['ink'], 600),
                text(left, 50, f'{len(trials)} scenes with seeds fixed before the run, never used for tuning. '
                     f'{full} of {len(trials)} full insertions. Each trial is worth up to 100.', 12.5, c['secondary']),
                text(left, 68, 'Run on an earlier development revision, before the last round of changes; '
                     'the final code has no held-out run.', 12.5, c['secondary'])]
        for v in (0, 25, 50, 75, 100):
            body.append(f'<line x1="{left}" x2="{W-right}" y1="{y(v):.1f}" y2="{y(v):.1f}" '
                        f'stroke="{c["grid"] if v else c["muted"]}" stroke-width="1"/>')
            body.append(text(left-8, y(v)+4, str(v), 11.5, c['muted'], 400, 'end'))
        body.append(f'<line x1="{left}" x2="{W-right}" y1="{y(75):.1f}" y2="{y(75):.1f}" stroke="{c["secondary"]}" '
                    f'stroke-width="1.2" stroke-dasharray="4 4"/>')
        for g, (name, members) in enumerate(groups):
            gx = left+g*(group_w+group_gap)
            bw = (group_w-bar_gap*(len(members)+1))/len(members)
            for i, t in enumerate(members):
                tone, label, short = status[t['outcome']]
                bx = gx+bar_gap+i*(bw+bar_gap)
                h = max(plot_h*t['total']/100, 3)
                body.append(bar(bx, base-h, bw, h, c[tone], f'{name}: {t["total"]:.1f}, {label}'))
                if t['outcome'] != 'full_insertion':
                    halo = f'stroke="{c["surface"]}" stroke-width="4" paint-order="stroke"'
                    body.append(text(bx+bw/2, base-h-22, short, 11, c['secondary'], 500, 'middle', halo))
                    body.append(text(bx+bw/2, base-h-8, f'{t["total"]:.1f}', 11.5, c['ink'], 600, 'middle', halo))
            subtotal = sum(t['total'] for t in members)
            n_full = sum(t['outcome'] == 'full_insertion' for t in members)
            body.append(text(gx+group_w/2, base+22, name, 13, c['ink'], 600, 'middle'))
            body.append(text(gx+group_w/2, base+40, f'{subtotal:.1f} / {100*len(members)}, {n_full} of '
                             f'{len(members)} full', 12, c['secondary'], 400, 'middle'))
        lx, ly = left, 94
        for key, tone, label, _ in OUTCOMES:
            body.append(f'<rect x="{lx}" y="{ly-10}" width="12" height="12" rx="3" fill="{c[tone]}"/>')
            body.append(text(lx+18, ly, label, 12, c['secondary']))
            lx += 18+len(label)*6.6+22
        body.append(f'<line x1="{lx}" x2="{lx+22}" y1="{ly-4}" y2="{ly-4}" stroke="{c["secondary"]}" '
                    f'stroke-width="1.2" stroke-dasharray="4 4"/>')
        body.append(text(lx+30, ly, '75 points for a full insertion', 12, c['secondary']))
        out = Path(f'{a.output}-{theme}.svg')
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(svg(W, base+56, body, c['surface']))
        print(out, f'{out.stat().st_size/1e3:.0f} kB')


if __name__ == '__main__':
    main()
