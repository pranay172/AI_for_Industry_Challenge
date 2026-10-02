#!/usr/bin/env python3
"""Build the demo GIF from a recorded benchmark run.

Frames come from a view-only Gazebo GUI attached to the evaluator during the
run (record_run.sh). Each trial is cut from the evaluator's InsertCable goal to
its result, using the engine's log timestamps (the same wall clock as the
frames), sped up and labelled with the official outcome and score.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
BOLD = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
TITLE_BAR_PX = 48
STAMP = re.compile(r'\[(\d{10}\.\d+)\] \[aic_engine\]: (.*)')


def trial_windows(log):
    """Wall-clock (goal sent, result) per trial, from the evaluator engine log."""
    windows, start = [], None
    for line in log.read_text(errors='replace').splitlines():
        m = STAMP.search(line)
        if not m:
            continue
        t, text = float(m.group(1)), re.sub(r'\x1b\[[0-9;]*m', '', m.group(2))
        if text.startswith('Sending InsertCable goal'):
            start = t
        elif start is not None and re.match(r'Task \[\S+\] (succeeded|failed|aborted|canceled|timed out)', text):
            windows.append((start, t)); start = None
    return windows


def label_font(size, bold=False):
    return ImageFont.truetype(BOLD if bold else FONT, size)


def overlay(img, left, right, speed, ok):
    draw = ImageDraw.Draw(img, 'RGBA')
    w, _ = img.size
    f, fb = label_font(17), label_font(17, True)
    draw.rounded_rectangle((10, 10, 20+draw.textlength(left, font=fb), 40), 6, fill=(20, 24, 30, 200))
    draw.text((15, 14), left, font=fb, fill=(255, 255, 255))
    if right:
        colour = (30, 140, 70, 220) if ok else (190, 110, 20, 220)
        tw = draw.textlength(right, font=fb)
        draw.rounded_rectangle((w-20-tw, 10, w-10, 40), 6, fill=colour)
        draw.text((w-15-tw, 14), right, font=fb, fill=(255, 255, 255))
    s = f'{speed:g}x speed'
    tw = draw.textlength(s, font=f)
    draw.rounded_rectangle((w-20-tw, img.size[1]-38, w-10, img.size[1]-10), 6, fill=(20, 24, 30, 160))
    draw.text((w-15-tw, img.size[1]-35), s, font=f, fill=(255, 255, 255))
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('run', type=Path)
    ap.add_argument('output', type=Path)
    ap.add_argument('--speed', type=float, default=4.)
    ap.add_argument('--fps', type=float, default=10.)
    ap.add_argument('--width', type=int, default=720)
    ap.add_argument('--hold', type=float, default=1.2, help='seconds the result frame is held')
    ap.add_argument('--crop', type=int, nargs=4, metavar=('LEFT', 'TOP', 'RIGHT', 'BOTTOM'),
                    help='crop box in GUI pixels below the title bar')
    a = ap.parse_args()

    frames_dir = a.run/'demo_frames'
    index = np.loadtxt(frames_dir/'frames.csv', delimiter=',')
    times = index[:, 1]
    windows = trial_windows(a.run/'compose.log')
    summary = json.loads((a.run/'summary.json').read_text())
    config = yaml.safe_load((a.run/'config.yaml').read_text())
    assert len(windows) == len(summary['trials']), (windows, len(summary['trials']))

    out, durations = [], []
    step = a.speed/a.fps
    for (start, end), result in zip(windows, summary['trials']):
        task = next(iter(config['trials'][result['trial']]['tasks'].values()))
        port = 'SFP port' if task['plug_type'] == 'sfp' else 'SC port'
        left = f"{result['trial'].replace('_', ' ').capitalize()}: {port}"
        ok = result['outcome'] == 'full_insertion'
        verdict = f"{result['outcome'].replace('_', ' ')}, {result['total']:.1f}"
        picks = [int(np.argmin(np.abs(times-t))) for t in np.arange(start, end+step, step)]
        for k, i in enumerate(picks):
            img = Image.open(frames_dir/f'{int(index[i, 0]):06d}.jpg').convert('RGB')
            img = img.crop((0, TITLE_BAR_PX, img.width, img.height))
            if a.crop:
                img = img.crop(tuple(a.crop))
            img = img.resize((a.width, round(img.height*a.width/img.width)), Image.LANCZOS)
            last = k == len(picks)-1
            out.append(overlay(img, left, verdict if last else '', a.speed, ok))
            durations.append(int(1000*(a.hold if last else 1/a.fps)))

    palette = Image.new('RGB', (out[0].width, out[0].height*4))
    for j, i in enumerate(np.linspace(0, len(out)-1, 4).astype(int)):
        palette.paste(out[i], (0, out[0].height*j))
    palette = palette.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    gif = [img.quantize(palette=palette, dither=Image.Dither.NONE) for img in out]
    gif[0].save(a.output, save_all=True, append_images=gif[1:], duration=durations, loop=0, optimize=True)
    print(a.output, f'{a.output.stat().st_size/1e6:.1f} MB', len(gif), 'frames',
          f'{sum(durations)/1000:.1f} s', 'windows', [(round(e-s, 1)) for s, e in windows])


if __name__ == '__main__':
    main()
