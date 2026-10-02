#!/usr/bin/env python3
"""README banner: one frame of the recorded demo run, with the title set beside it.

Lays the page out in HTML and renders it with headless Chrome (google-chrome on PATH):

  banner.py benchmark_runs/<run>/demo_frames/001020.jpg docs/assets/banner.jpg
"""
import argparse
import base64
import io
import subprocess
import tempfile
from pathlib import Path

from PIL import Image

TITLE_BAR_PX = 48   # Gazebo's window title bar at the top of each captured frame
W, H, SCALE = 1280, 400, 1.5

PAGE = """<!doctype html><meta charset="utf-8"><style>
html, body {{ margin: 0; width: {W}px; height: {H}px; overflow: hidden; background: #0b111c; }}
.shot {{ position: absolute; right: 0; top: 0; height: {H}px; }}
.fade {{ position: absolute; inset: 0;
  background: linear-gradient(90deg, #0b111c 0%, #0b111c 41%, rgba(11,17,28,.88) 50%, rgba(11,17,28,.45) 62%,
    rgba(11,17,28,0) 76%); }}
.text {{ position: absolute; left: 64px; top: 0; height: {H}px; width: 640px; display: flex; flex-direction: column;
  justify-content: center; font-family: Lato, 'Helvetica Neue', Arial, sans-serif; color: #fff; }}
.over {{ font-size: 15px; letter-spacing: .14em; text-transform: uppercase; color: #8fb8ee; font-weight: 700; }}
h1 {{ font-size: 56px; line-height: 1.06; margin: 16px 0 18px; font-weight: 900; letter-spacing: -.01em; }}
p {{ font-size: 19px; line-height: 1.45; margin: 0; color: #c9d1dc; max-width: 560px; }}
</style>
<img class="shot" src="data:image/jpeg;base64,{shot}">
<div class="fade"></div>
<div class="text">
  <div class="over">A solution for the AI for Industry Challenge</div>
  <h1>Vision-guided<br>cable insertion</h1>
  <p>A UR5e inserts SFP and SC fiber plugs into randomized ports, guided by its wrist cameras and
  force&ndash;torque sensor.</p>
</div>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('frame', type=Path)
    ap.add_argument('output', type=Path)
    ap.add_argument('--left', type=int, default=0, help='frame pixels to drop on the left')
    a = ap.parse_args()
    frame = Image.open(a.frame).convert('RGB')
    frame = frame.crop((a.left, TITLE_BAR_PX, frame.width, frame.height))
    buf = io.BytesIO()
    frame.save(buf, 'JPEG', quality=95)
    page = PAGE.format(W=W, H=H, shot=base64.b64encode(buf.getvalue()).decode())
    with tempfile.TemporaryDirectory() as tmp:
        html, png = Path(tmp)/'banner.html', Path(tmp)/'banner.png'
        html.write_text(page)
        subprocess.run(['google-chrome', '--headless=new', '--disable-gpu', '--no-sandbox', '--hide-scrollbars',
                        f'--window-size={W},{H}', f'--force-device-scale-factor={SCALE}', f'--screenshot={png}',
                        html.as_uri()], check=True, capture_output=True)
        image = Image.open(png).convert('RGB')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(a.output, 'JPEG', quality=88, optimize=True, progressive=True)
    print(a.output, image.size, f'{a.output.stat().st_size/1e3:.0f} kB')


if __name__ == '__main__':
    main()
