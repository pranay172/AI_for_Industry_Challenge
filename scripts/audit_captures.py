#!/usr/bin/env python3
"""Validate captured RGB/metadata pairs and report collection coverage, without inference."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from replay_perception import capture_inputs


def audit(directory):
    counts = Counter()
    files, scenes, episodes, targets = {}, set(), set(), set()
    exposure_ids = set()
    metadata_paths = sorted(directory.glob('*.json'))
    referenced = set()
    for path in metadata_paths:
        metadata = json.loads(path.read_text())
        npz = path.parent/metadata['files']['images_npz']
        referenced.add(npz.resolve())
        for item in (path,npz):
            files[item.name] = hashlib.sha256(item.read_bytes()).hexdigest()
        with np.load(npz, allow_pickle=False) as archive:
            images = {k:archive[k] for k in archive.files}
        if any(rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[2]!=3 for rgb in images.values()):
            raise ValueError(f'Invalid RGB data: {npz}')
        names,geometry = capture_inputs(metadata, images)
        counts['captures'] += 1
        counts['images'] += len(images)
        counts['fresh_synchronized_images'] += len(names)
        counts['images_with_legal_geometry'] += len(geometry)
        scenes.add(metadata.get('scene_id',''))
        episodes.add(metadata.get('episode_id',''))
        task = metadata['task']
        targets.add((task['plug_type'],task['target_module_name'],task['port_name']))
        for camera in names:
            stamp = metadata['camera_exposures'][camera]
            exposure_ids.add((metadata.get('episode_id',''),camera,stamp['sec'],stamp['nanosec']))
            gt = metadata.get('ground_truth',{}).get(camera,{}) or {}
            valid_gt = gt.get('xyz_camera_port_origin') is not None and gt.get('R_cam_from_port') is not None
            counts['images_with_gt_transform'] += int(valid_gt)
            counts['target_in_frustum_images'] += int(valid_gt and gt.get('visible',False))
    orphan = {p.resolve() for p in directory.glob('*.npz')}-referenced
    if orphan:
        raise ValueError(f'{len(orphan)} orphan NPZ files in {directory}')
    return {'counts':dict(counts), 'unique_exposures':len(exposure_ids),
            'scene_ids':sorted(scenes), 'episode_ids':sorted(episodes), 'targets':sorted(targets),
            'files_sha256':files, 'occlusion_verified':False,
            'scope':'stationary frames are repeated views, not independent scenes'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('captures',type=Path,nargs='+')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result={str(p.resolve()):audit(p) for p in args.captures}
    with args.output.open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
        stream.write('\n')
    print(json.dumps({p:r['counts'] for p,r in result.items()},indent=2))
