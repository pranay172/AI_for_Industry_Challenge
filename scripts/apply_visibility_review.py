#!/usr/bin/env python3
"""Apply explicit per-image landmark visibility review to a training manifest.

Only reviewed images are emitted. Review masks can remove frustum-visible
landmarks, never make off-image points visible or change geometry/scene identity.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def projections(row):
    if 'card' in row:
        return [row['card'], *[row['sfp_ports'][n]['face_center']
                              for n in ('sfp_port_0', 'sfp_port_1')]]
    return [row['sc_port']['face_corners'], row['sc_port']['face_center']]


def apply_review(labels, review, output):
    if output.exists():
        raise ValueError('Output already exists')
    record = json.loads(review.read_text())
    if record['labels_sha256'] != digest(labels):
        raise ValueError('Labels changed since review')
    rows = [json.loads(line) for line in labels.read_text().splitlines() if line.strip()]
    indexed = {}
    for row in rows:
        key = (row['sample_id'], row['image_key'])
        if key in indexed:
            raise ValueError('Duplicate sample/image identity')
        indexed[key] = row
    selected = []; seen = set()
    for item in record['images']:
        key = (item['sample_id'], item['image_key'])
        if key in seen or key not in indexed:
            raise ValueError('Duplicate or unknown review image')
        seen.add(key)
        row = deepcopy(indexed[key])
        capture = Path(row['npz_path']).expanduser()
        capture = (labels.parent/capture).resolve() if not capture.is_absolute() else capture
        if item.get('capture_sha256') != digest(capture):
            raise ValueError('Capture changed since review')
        mask = item['visible']
        original = [v for p in projections(row) for v in p['visible']]
        if len(mask) != len(original) or any(type(v) is not bool for v in mask):
            raise ValueError('Review requires one boolean per landmark')
        if any(v and not old for v, old in zip(mask, original)):
            raise ValueError('Review cannot enable an off-image landmark')
        if not item.get('reason', '').strip():
            raise ValueError('Review requires a reason')
        offset = 0
        for projection in projections(row):
            count = len(projection['visible'])
            projection['visible'] = mask[offset:offset+count]
            offset += count
        # All-masked images are explicit rejections, not empty training targets.
        if not any(mask):
            continue
        row['visibility']['occlusion'] = 'image_reviewed_landmark_mask'
        row['label_category'] = 'reviewed_training'
        row['visibility_review'] = {'review_sha256': digest(review),
                                    'source_labels_sha256': record['labels_sha256'],
                                    'reason': item['reason'], 'capture_sha256': item['capture_sha256']}
        # Keep relative paths valid when the output moves to another directory.
        path = Path(row['npz_path']).expanduser()
        row['npz_path'] = str((labels.parent/path).resolve() if not path.is_absolute() else path)
        selected.append(row)
    if not selected:
        raise ValueError('Review contains no usable images')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        for row in selected:
            stream.write(json.dumps(row, allow_nan=False)+'\n')
    return {'input_rows': len(rows), 'reviewed_rows': len(seen), 'output_rows': len(selected)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('labels', 'review', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(apply_review(args.labels, args.review, args.output), indent=2))
