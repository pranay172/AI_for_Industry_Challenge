#!/usr/bin/env python3
"""README insertion figure: one trial from start to seated, from its evaluator bag.

Two panels on one time axis (simulated seconds since the task started):
lateral offset of the plug tip from the port axis (log scale) and depth along
the port axis from the entrance face. Phase bands come from the
policy's own log. The plug and port poses are ground truth read after the run
(insertion_trace.py); the policy never sees them.

  insertion_figure.py <trace.csv> <run>/compose.log --trial 1 docs/assets/insertion
writes docs/assets/insertion-light.svg and docs/assets/insertion-dark.svg.
"""
import argparse
import math
import re
from pathlib import Path

import numpy as np

from svg_chart import THEMES, polyline, svg, text

STAMP = re.compile(r'\[(\d{10}\.\d+)\] \[(aic_model|aic_engine)\]: (.*)')


def phases(log, trial):
    """Wall-clock phase boundaries for the nth insert_cable call, from the policy and engine logs."""
    start, marks, n = None, {}, 0
    for line in log.read_text(errors='replace').splitlines():
        m = STAMP.search(line)
        if not m:
            continue
        t, node, s = float(m.group(1)), m.group(2), re.sub(r'\x1b\[[0-9;]*m', '', m.group(3))
        if node == 'aic_model' and s.startswith('Policy.insert_cable()'):
            n += 1
            if n == trial:
                start = t
            elif n > trial:
                break
            continue
        if start is None:
            continue
        p = re.search(r'phase=(\w+)', s)
        if p and p.group(1) not in marks:
            marks[p.group(1)] = t
        for key, pattern in (('search', r'\[face_search\] start n=1'), ('entered', r'\[face_search\] entered'),
                             ('seated', r'\[face_search\] seating stopped')):
            if re.match(pattern, s) and key not in marks:
                marks[key] = t
        if node == 'aic_engine' and re.match(r'Task \[\S+\] succeeded', s):
            marks['end'] = t
            break
    return start, marks


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('trace', type=Path)
    ap.add_argument('log', type=Path)
    ap.add_argument('output', type=Path, help='prefix; -light.svg and -dark.svg are appended')
    ap.add_argument('--trial', type=int, default=1)
    ap.add_argument('--title', default='One SFP insertion, start to seated')
    a = ap.parse_args()
    data = np.genfromtxt(a.trace, delimiter=',', names=True)
    start, marks = phases(a.log, a.trial)
    to_sim = lambda t: float(np.interp(t, data['wall_s'], data['sim_s']))
    t0 = to_sim(start)
    keep = (data['wall_s'] >= start-0.5) & (data['wall_s'] <= marks['end']+0.5)
    t = np.array([to_sim(w)-t0 for w in data['wall_s'][keep]])
    depth, lateral = data['depth_mm'][keep], data['lateral_mm'][keep]
    band_edges = [('Register board,\nframe the rail', 0.0, to_sim(marks['find_target'])-t0),
                  ('Detect and align', to_sim(marks['find_target'])-t0, to_sim(marks['pre_insert'])-t0),
                  ('Pre-insert', to_sim(marks['pre_insert'])-t0, to_sim(marks['search'])-t0),
                  ('Search', to_sim(marks['search'])-t0, to_sim(marks['entered'])-t0),
                  ('Seat and hold', to_sim(marks['entered'])-t0, to_sim(marks['end'])-t0)]
    t_end = band_edges[-1][2]

    W, left, right = 920, 76, 24
    plot_w = W-left-right
    x = lambda s: left+plot_w*s/t_end
    for theme, c in THEMES.items():
        body = []
        body.append(text(left, 30, a.title, 17, c['ink'], 600))
        body.append(text(left, 50, 'Simulated time since the task started. Plug and port poses are ground truth '
                         'read from the evaluator bag after the run.', 12.5, c['secondary']))
        top = 92
        panels = [('Off the port axis (mm, log scale)', 170),
                  ('Depth along the port axis (mm; negative is above the face)', 190)]
        bottom = top+sum(h for _, h in panels)+30*(len(panels)-1)
        # phase bands span all panels
        for i, (label, s0, s1) in enumerate(band_edges):
            fill = c['band'] if i % 2 == 0 else c['band2']
            body.append(f'<rect x="{x(s0):.1f}" y="{top-26}" width="{x(s1)-x(s0):.1f}" height="{bottom-top+26}" '
                        f'fill="{fill}"/>')
            lines = label.split('\n')
            for j, line in enumerate(lines):
                body.append(text((x(s0)+x(s1))/2, top-14+13*j-6.5*(len(lines)-1), line, 11.5, c['secondary'], 500,
                                 'middle'))
        y0 = top
        for k, (title, h) in enumerate(panels):
            body.append(text(left, y0+14, title, 12.5, c['ink'], 600))
            py0, py1 = y0+22, y0+h
            if k == 0:
                lo, hi = math.log10(0.3), math.log10(300)
                ymap = lambda v: py1-(py1-py0)*(math.log10(max(v, 0.3))-lo)/(hi-lo)
                ticks = [(1, '1'), (10, '10'), (100, '100')]
                series = [(t, lateral, c['series'][0])]
            else:
                step = 50 if float(depth.max()-depth.min()) < 150 else 100
                lo, hi = step*math.floor(float(depth.min())/step), step*math.ceil(float(depth.max())/step+0.2)
                ymap = lambda v: py1-(py1-py0)*(v-lo)/(hi-lo)
                ticks = [(v, f'{v}'.replace('-', '−')) for v in range(int(lo), int(hi)+1, step)]
                series = [(t, depth, c['series'][1])]
            for v, lab in ticks:
                body.append(f'<line x1="{left}" x2="{W-right}" y1="{ymap(v):.1f}" y2="{ymap(v):.1f}" '
                            f'stroke="{c["grid"]}" stroke-width="1"/>')
                body.append(text(left-8, ymap(v)+4, lab, 11.5, c['muted'], 400, 'end'))
            if k == 1:
                body.append(f'<line x1="{left}" x2="{W-right}" y1="{ymap(0):.1f}" y2="{ymap(0):.1f}" '
                            f'stroke="{c["secondary"]}" stroke-width="1.2" stroke-dasharray="4 4"/>')
                body.append(text(W-right-4, ymap(0)-6, 'port entrance face', 11.5, c['secondary'], 400, 'end'))
                seated = float(depth[-1])
                body.append(text(x(t_end)-6, ymap(seated)-8, f'seated, {seated:.1f} mm in', 11.5, c['ink'], 600, 'end'))
            if k == 0:
                body.append(f'<line x1="{left}" x2="{W-right}" y1="{ymap(1):.1f}" y2="{ymap(1):.1f}" '
                            f'stroke="{c["secondary"]}" stroke-width="1.2" stroke-dasharray="4 4"/>')
                final = float(lateral[-1])
                body.append(text(x(t_end)-6, ymap(final)-8, f'{final:.1f} mm off axis', 11.5, c['ink'], 600, 'end'))
            for ts, vs, colour in series:
                body.append(polyline([(x(s), ymap(v)) for s, v in zip(ts, vs)], colour, 2))
            y0 = py1+30
        for s in range(0, int(t_end)+1, 5):
            body.append(f'<line x1="{x(s):.1f}" x2="{x(s):.1f}" y1="{bottom}" y2="{bottom+5}" stroke="{c["muted"]}"/>')
            body.append(text(x(s), bottom+19, f'{s}', 11.5, c['muted'], 400, 'middle'))
        body.append(text(left+plot_w/2, bottom+38, 'seconds', 12, c['secondary'], 400, 'middle'))
        out = Path(f'{a.output}-{theme}.svg')
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(svg(W, bottom+50, body, c['surface']))
        print(out, f'{out.stat().st_size/1e3:.0f} kB')
    print({k: round(to_sim(v)-t0, 2) for k, v in marks.items()})


if __name__ == '__main__':
    main()
