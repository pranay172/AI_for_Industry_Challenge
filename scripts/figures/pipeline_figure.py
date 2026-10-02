#!/usr/bin/env python3
"""README pipeline figure: the stages of one trial, perception then contact.

  pipeline_figure.py docs/assets/pipeline
writes docs/assets/pipeline-light.svg and docs/assets/pipeline-dark.svg.
"""
import argparse
from pathlib import Path

from svg_chart import THEMES, svg, text

ROWS = [
    ('Perception', 'Wrist cameras get the plug to within about a millimetre of the port',
     '3 wrist cameras', 'and arm kinematics', [
         ('Measure the grasp', ['Plug keypoints give', 'the plug pose in', 'the gripper'], True),
         ('Register the board', ['Fit the board marker', 'across the cameras'], False),
         ('Frame the rail', ['A bounded motion', 'brings the rail', 'into view'], False),
         ('Detect and decode', ['Port-face heatmaps,', 'fit to card or port', 'templates'], True),
         ('Fuse and lock', ['Cameras must agree,', 'rail checks pass,', 'then lock'], False)]),
    ('Contact', 'Compliant motion and contact finish the job',
     'Arm kinematics', 'and wrist force-torque', [
         ('Align above the port', ['Compliant approach,', 'plug centred on the axis'], False),
         ('Land on the face', ['Descend until the plug', 'rests on the port face'], False),
         ('Spiral and yaw search', ['Find the opening', 'by contact'], False),
         ('Seat and hold', ['Push past detents, hold', 'until the port registers'], False)]),
]


def arrow(x1, y1, x2, y2, colour):
    return (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2-6:.1f}" y2="{y2:.1f}" stroke="{colour}" stroke-width="1.6"/>'
            f'<path d="M{x2:.1f},{y2:.1f} l-7,-4 v8 z" fill="{colour}"/>')


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('output', type=Path, help='prefix; -light.svg and -dark.svg are appended')
    a = ap.parse_args()

    W, pad, gap = 920, 14, 12
    inputs_x, inputs_w = 16, 132
    group_x = inputs_x+inputs_w+28
    group_w = W-16-group_x
    heights, row_gap, top = (186, 132), 54, 16
    for theme, c in THEMES.items():
        accent, learned = c['series'][0], c['series'][1]
        body = []
        rows_y = []
        for r, (name, blurb, sensor, sensor2, cards) in enumerate(ROWS):
            group_h = heights[r]
            gy = top+sum(heights[:r])+r*row_gap
            rows_y.append(gy)
            body.append(f'<rect x="{group_x}" y="{gy}" width="{group_w}" height="{group_h}" rx="10" '
                        f'fill="{c["band"]}" stroke="{c["card_edge"]}"/>')
            body.append(text(group_x+pad, gy+24, name, 14, c['ink'], 700))
            body.append(text(group_x+pad+8+len(name)*8.6, gy+24, blurb, 12.5, c['secondary']))
            # sensor card, with an arrow into the group
            sy = gy+group_h/2-30
            body.append(f'<rect x="{inputs_x}" y="{sy:.1f}" width="{inputs_w}" height="60" rx="8" '
                        f'fill="{c["card"]}" stroke="{c["card_edge"]}"/>')
            body.append(text(inputs_x+inputs_w/2, sy+26, sensor, 12.5, c['ink'], 600, 'middle'))
            body.append(text(inputs_x+inputs_w/2, sy+44, sensor2, 11.5, c['secondary'], 400, 'middle'))
            body.append(arrow(inputs_x+inputs_w, sy+30, group_x, sy+30, c['muted']))
            n = len(cards)
            cw = (group_w-2*pad-(n-1)*gap)/n
            cy, ch = gy+40, group_h-40-pad
            for i, (title, lines, is_learned) in enumerate(cards):
                cx = group_x+pad+i*(cw+gap)
                body.append(f'<rect x="{cx:.1f}" y="{cy}" width="{cw:.1f}" height="{ch}" rx="8" fill="{c["surface"]}" '
                            f'stroke="{c["card_edge"]}"/>')
                body.append(f'<rect x="{cx:.1f}" y="{cy}" width="4" height="{ch}" rx="2" '
                            f'fill="{learned if is_learned else accent}"/>')
                body.append(text(cx+14, cy+24, title, 12.5, c['ink'], 600))
                for j, line in enumerate(lines):
                    body.append(text(cx+14, cy+44+16*j, line, 11.5, c['secondary']))
                if is_learned:
                    label = 'learned model'
                    body.append(f'<rect x="{cx+14:.1f}" y="{cy+ch-28}" width="{len(label)*6.0+14:.1f}" height="19" '
                                f'rx="9.5" fill="none" stroke="{learned}"/>')
                    body.append(text(cx+21, cy+ch-14.5, label, 11, c['ink'], 500))
                if i < n-1:
                    body.append(arrow(cx+cw+1, cy+ch/2, cx+cw+gap-1, cy+ch/2, c['muted']))
        # hand-off from the last perception stage to the first contact stage
        n0, n1 = len(ROWS[0][4]), len(ROWS[1][4])
        cw0 = (group_w-2*pad-(n0-1)*gap)/n0
        cw1 = (group_w-2*pad-(n1-1)*gap)/n1
        x_from = group_x+pad+(n0-1)*(cw0+gap)+cw0/2
        x_to = group_x+pad+cw1/2
        y_from = rows_y[0]+heights[0]
        y_mid = y_from+row_gap/2
        y_to = rows_y[1]
        body.append(f'<path d="M{x_from:.1f},{y_from} V{y_mid:.1f} H{x_to:.1f} V{y_to-6}" fill="none" '
                    f'stroke="{c["muted"]}" stroke-width="1.6"/>')
        body.append(f'<path d="M{x_to:.1f},{y_to} l-4,-7 h8 z" fill="{c["muted"]}"/>')
        label = 'hand-off: port pose, and the plug pose in the gripper'
        lw = len(label)*6.5+20
        lx = (x_from+x_to)/2
        body.append(f'<rect x="{lx-lw/2:.1f}" y="{y_mid-11:.1f}" width="{lw:.1f}" height="22" rx="11" '
                    f'fill="{c["surface"]}" stroke="{c["card_edge"]}"/>')
        body.append(text(lx, y_mid+4, label, 11.5, c['secondary'], 500, 'middle'))
        # legend
        ly = rows_y[1]+heights[1]+26
        for k, (colour, label) in enumerate(((accent, 'geometry and control'), (learned, 'learned model'))):
            lx = group_x+k*190
            body.append(f'<rect x="{lx}" y="{ly-11}" width="4" height="14" rx="2" fill="{colour}"/>')
            body.append(text(lx+12, ly, label, 11.5, c['secondary']))
        out = Path(f'{a.output}-{theme}.svg')
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(svg(W, ly+14, body, c['surface']))
        print(out, f'{out.stat().st_size/1e3:.0f} kB')


if __name__ == '__main__':
    main()
