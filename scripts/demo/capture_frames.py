#!/usr/bin/env python3
"""Read the Xvfb framebuffer (-fbdir XWD file) at a fixed rate and save JPEG frames.

Runs inside the evaluator container next to a view-only Gazebo GUI client.
Stops when <out>/STOP exists or after --max-sec. frames.csv maps frame index to wall time.
"""
import argparse, os, struct, time
from pathlib import Path
import numpy as np, cv2

ap = argparse.ArgumentParser()
ap.add_argument('--fb', default='/tmp/fb/Xvfb_screen0')
ap.add_argument('--out', required=True)
ap.add_argument('--fps', type=float, default=8.)
ap.add_argument('--max-sec', type=float, default=1500.)
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
while not os.path.exists(a.fb):
    time.sleep(.2)
log = open(out/'frames.csv', 'a')
start = time.time(); i = 0; period = 1/a.fps
while time.time()-start < a.max_sec and not (out/'STOP').exists():
    t0 = time.time()
    data = open(a.fb, 'rb').read()
    h = struct.unpack('>25I', data[:100])
    header_size, width, height, bpp, bpl, ncolors = h[0], h[4], h[5], h[11], h[12], h[19]
    off = header_size+ncolors*12
    img = np.frombuffer(data, np.uint8, count=bpl*height, offset=off).reshape(height, bpl)[:, :width*4].reshape(height, width, 4)
    cv2.imwrite(str(out/f'{i:06d}.jpg'), img[:, :, :3], [cv2.IMWRITE_JPEG_QUALITY, 88])
    log.write(f'{i},{t0:.3f}\n'); log.flush(); i += 1
    time.sleep(max(0., period-(time.time()-t0)))
