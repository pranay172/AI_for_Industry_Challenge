#!/usr/bin/env python3
"""README perception figure: what the policy sees and what it extracts.

Replays the shipped detectors on captured wrist-camera exposures (benchmark runs
prepared with capture on) and draws, per camera:
- the board marker pixels the board registration used;
- the requested rail's public envelope, projected from the registered board;
- the port-face landmarks the detector and template decoder returned.

Nothing here reads ground truth; it is the runtime perception path, offline.

  perception_figure.py <captures dir> docs/assets/perception.png \\
      --sfp <capture json> --sc <capture json>
"""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'aic_model'), str(ROOT/'scripts')]
from replay_perception import register  # noqa: E402
from aic_model.board_registration import infer_module, module_pixels  # noqa: E402

FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
BOLD = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
RAIL = (42, 120, 214)       # categorical slot 1 (blue)
LANDMARK = (235, 104, 52)   # slot 2 (orange)
MARKER = (27, 175, 122)     # slot 3 (aqua)
INK, MUTED, SURFACE = (11, 11, 11), (82, 81, 78), (252, 252, 251)
PANEL = 360


def episode_board(captures, target):
    """Register the board from the episode's captures up to the target exposure, as the policy does."""
    records = sorted(((p, json.loads(p.read_text())) for p in captures.glob('*.json')),
                     key=lambda r: r[1]['capture_time_sim'])
    board = None
    for path, meta in records:
        if meta['episode_id'] != target['episode_id'] or meta['capture_time_sim'] > target['capture_time_sim']:
            continue
        with np.load(captures/meta['files']['images_npz']) as archive:
            images = {k: archive[k] for k in archive.files}
        board, names, geometry = register(meta, images, board)
        if path.name == target['_name']:
            return board, images, geometry
    raise ValueError('target capture not found')


def panel(image, board, geometry, prediction, module, scale):
    over = image.copy()
    if geometry is not None:
        hull = module_pixels(*geometry, board, module)
        if hull is not None:
            hull = cv2.convexHull(hull.astype(np.float32)).astype(np.int32)
            fill = over.copy()
            cv2.fillPoly(fill, [hull], RAIL)
            over = cv2.addWeighted(fill, .22, over, .78, 0)
            cv2.polylines(over, [hull], True, RAIL, 4, cv2.LINE_AA)
    if prediction is not None:
        for u, v in np.asarray(prediction['points_px']):
            cv2.circle(over, (int(round(u)), int(round(v))), 11, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(over, (int(round(u)), int(round(v))), 8, LANDMARK, -1, cv2.LINE_AA)
    small = cv2.resize(over, (int(image.shape[1]*scale), int(image.shape[0]*scale)), interpolation=cv2.INTER_AREA)
    return Image.fromarray(small)


def marker_overlay(image, marker_mask_color=MARKER):
    """Outline the board marker the registration fits (the magenta marker, found by colour)."""
    color = image.astype(np.float32)
    r, g, b = color[:, :, 0], color[:, :, 1], color[:, :, 2]
    mask = ((r > 45) & (b > 45) & (r > 1.7*g) & (b > 1.7*g) & (np.abs(r-b) < 100)).astype(np.uint8)*255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = image.copy()
    cv2.drawContours(out, [c for c in contours if cv2.contourArea(c) > 40], -1, marker_mask_color, 4, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('captures', type=Path)
    ap.add_argument('output', type=Path)
    ap.add_argument('--sfp', required=True, help='capture json name for the SFP row')
    ap.add_argument('--sc', required=True, help='capture json name for the SC row')
    a = ap.parse_args()

    from aic_model.sfp_face_decoder import load_sfp_face_heatmap
    from aic_model.sc_heatmap_detector import load_sc_port_heatmap
    detectors = {'sfp': load_sfp_face_heatmap(ROOT/'aic_model/models/sfp_port_detector.pt'),
                 'sc': load_sc_port_heatmap(ROOT/'aic_model/models/sc_port_detector.pt')}
    font, bold, small = ImageFont.truetype(FONT, 17), ImageFont.truetype(BOLD, 19), ImageFont.truetype(FONT, 15)
    rows = []
    for mode, name in (('sfp', a.sfp), ('sc', a.sc)):
        meta = json.loads((a.captures/name).read_text()); meta['_name'] = name
        board, images, geometry = episode_board(a.captures, meta)
        module = meta['task']['target_module_name']
        cells = []
        for camera in ('left', 'center', 'right'):
            image = marker_overlay(images[camera+'_image'])
            geo = geometry.get(camera)
            pred = infer_module(detectors[mode], images[camera+'_image'], geo, board, module) if geo else None
            scale = PANEL/image.shape[1]
            cell = panel(image, board, geo, pred, module, scale)
            d = ImageDraw.Draw(cell, 'RGBA')
            tag = f'{camera} camera'
            d.rounded_rectangle((8, 8, 16+d.textlength(tag, font=small), 32), 5, fill=(20, 24, 30, 190))
            d.text((12, 11), tag, font=small, fill=(255, 255, 255))
            if pred is None:
                note = 'detection rejected'
                d.rounded_rectangle((8, cell.height-34, 16+d.textlength(note, font=small), cell.height-10), 5,
                                    fill=(20, 24, 30, 190))
                d.text((12, cell.height-31), note, font=small, fill=(255, 255, 255))
            cells.append(cell)
        port = 'SFP port on a NIC card' if mode == 'sfp' else 'SC port on its rail'
        rows.append((f'{port}  ({module})', cells))

    gap, margin, head = 10, 24, 34
    width = 2*margin + 3*PANEL + 2*gap
    cell_h = rows[0][1][0].height
    legend_h = 44
    height = margin + len(rows)*(head+cell_h) + (len(rows)-1)*gap*2 + legend_h + margin
    sheet = Image.new('RGB', (width, height), SURFACE)
    d = ImageDraw.Draw(sheet)
    y = margin
    for title, cells in rows:
        d.text((margin, y), title, font=bold, fill=INK)
        y += head
        for i, cell in enumerate(cells):
            sheet.paste(cell, (margin + i*(PANEL+gap), y))
        y += cell_h + 2*gap
    x = margin
    for colour, label in ((MARKER, 'board marker (registration)'), (RAIL, 'requested rail, from public geometry'),
                          (LANDMARK, 'detected port-face landmarks')):
        d.rounded_rectangle((x, y+6, x+22, y+22), 4, fill=colour)
        d.text((x+30, y+4), label, font=font, fill=MUTED)
        x += 30 + d.textlength(label, font=font) + 28
    a.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(a.output, optimize=True)
    print(a.output, sheet.size, f'{a.output.stat().st_size/1e3:.0f} kB')


if __name__ == '__main__':
    main()
