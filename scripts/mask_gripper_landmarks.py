#!/usr/bin/env python3
"""Mark landmarks behind the gripper as not visible (training labels only).

The wrist cameras see the gripper at a fixed place in every image.
scripts/data/gripper_masks.npz holds, per camera, the pixels that are dark
(max channel < 40) in over 90% of 120 random wide-train captures. Those masks
are dilated by 8 px here. Rows whose visibility was reviewed by eye
(label_category reviewed_training) are left unchanged. Reads
all-{mode}-{split}.jsonl from the dataset directory (build_wide_faces_dataset.py),
writes all-{mode}-{split}-masked.jsonl next to it and prints counts.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

MASKS = Path(__file__).resolve().parent/'data/gripper_masks.npz'


def load_masks(path=MASKS, dilation_px=8):
    kernel = np.ones((2*dilation_px+1, 2*dilation_px+1), np.uint8)
    return {c: cv2.dilate(m.astype(np.uint8), kernel) > 0 for c, m in np.load(path).items()}


def hidden(masks, camera, point):
    m = masks[camera]; u, v = int(round(point[0])), int(round(point[1]))
    return 0 <= v < m.shape[0] and 0 <= u < m.shape[1] and bool(m[v, u])


def mask_file(masks, mode, source, target):
    changed = rows = total = 0; out = []
    for line in open(source):
        row = json.loads(line); rows += 1
        if row.get('label_category') != 'reviewed_training':
            cam = row['camera'] if 'camera' in row else row['image_key'].split('_')[0]
            projections = ([row['sc_port']['face_corners'], row['sc_port']['face_center']] if mode == 'sc'
                           else [row['landmarks']])
            for proj in projections:
                for i, (p, vis) in enumerate(zip(proj['points_px'], proj['visible'])):
                    total += bool(vis)
                    if vis and hidden(masks, cam, p):
                        proj['visible'][i] = False; changed += 1
                if 'all_visible' in proj:
                    proj['all_visible'] = bool(all(proj['visible'])); proj['any_visible'] = bool(any(proj['visible']))
            row.setdefault('visibility', {})['gripper_mask'] = 'gripper-masks.npz, 8 px dilation'
        out.append(json.dumps(row, allow_nan=False)+'\n')
    target.write_text(''.join(out))
    print(mode, source.stem, 'rows', rows, 'visible landmarks', total, 'hidden by gripper', changed)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('dataset_dir', type=Path)
    ap.add_argument('--masks', type=Path, default=MASKS)
    a = ap.parse_args()
    masks = load_masks(a.masks)
    for mode in ('sc', 'sfp'):
        for split in ('train', 'val'):
            mask_file(masks, mode, a.dataset_dir/f'all-{mode}-{split}.jsonl', a.dataset_dir/f'all-{mode}-{split}-masked.jsonl')
