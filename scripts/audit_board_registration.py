#!/usr/bin/env python3
"""Measure uncached RGB board acquisition on saved, exposure-matched captures.

Port ground truth is consulted only after registration, to check rail assignment.
It is not a board-pose accuracy measurement or a model/insertion benchmark.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'aic_model'))
import numpy as np
from aic_model.board_registration import register_views, matches_module
from replay_perception import capture_inputs, source_fingerprint


def module_origin(task, gt, geometry):
    """Post-hoc GT reference using the same public asset transforms as runtime."""
    _, rotation, translation = geometry
    port_origin=rotation@np.array(gt['xyz_camera_port_origin'])+translation
    if task['plug_type'].lower()=='sc':
        return port_origin
    from aic_model.sfp_geometry import SFP_PORT_POSES_CARD, sfp_port_rotation_card, rpy_to_matrix
    port=task['port_name']
    R_base_port=rotation@np.array(gt['R_cam_from_port']).reshape(3,3)
    R_base_card=R_base_port@sfp_port_rotation_card(port).T
    card_origin=port_origin-R_base_card@np.array(SFP_PORT_POSES_CARD[port]['translation'])
    R_base_mount=R_base_card@rpy_to_matrix(-1.57,0.,0.).T
    return card_origin-R_base_mount@np.array([-.002,-.01785,.0899])


def audit(directory):
    rows, hashes = [], {}
    for path in sorted(directory.glob('*.json')):
        metadata = json.loads(path.read_text())
        archive_path = path.parent/metadata['files']['images_npz']
        for file in (path, archive_path):
            hashes[file.name] = hashlib.sha256(file.read_bytes()).hexdigest()
        with np.load(archive_path, allow_pickle=False) as archive:
            images = {key:archive[key] for key in archive.files}
        names, geometry = capture_inputs(metadata, images)
        start = time.perf_counter()
        views = {name:(images[name+'_image'], *geometry[name])
                 for name in names if name in geometry}
        board = register_views(views)
        row = {'sample_id':metadata['sample_id'], 'registered':board is not None,
               'available_cameras':list(views),
               'seconds':time.perf_counter()-start, 'module_rail_checks':[]}
        if board is not None:
            row.update(rotation=board.rotation.tolist(),translation=board.translation.tolist())
            task=metadata['task']
            for name in names:
                gt=metadata.get('ground_truth',{}).get(name,{}) or {}
                if name not in geometry or gt.get('xyz_camera_port_origin') is None:
                    continue
                point=module_origin(task, gt, geometry[name])
                family, count = ('sc_port', 2) if task['plug_type'].lower()=='sc' else ('nic_card_mount', 5)
                assigned=[f'{family}_{i}' for i in range(count)
                          if matches_module(board,point,f'{family}_{i}')]
                row['module_rail_checks'].append({'camera':name,'target':task['target_module_name'],
                                                  'assigned':assigned,
                                                  'correct':assigned==[task['target_module_name']]})
        rows.append(row)
    checks=[c for row in rows for c in row['module_rail_checks']]
    return {'captures':len(rows), 'registered':sum(row['registered'] for row in rows),
            'module_rail_checks':len(checks), 'module_rail_checks_correct':sum(c['correct'] for c in checks),
            'latency_p95_seconds':float(np.percentile([row['seconds'] for row in rows],95)) if rows else None,
            'rows':rows, 'capture_sha256':hashes}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('captures', type=Path, nargs='+')
    parser.add_argument('--output', type=Path, required=True)
    args=parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output path')
    source = source_fingerprint([Path(__file__).resolve()])
    result={'directories':{str(path.resolve()):audit(path) for path in args.captures},
            'scope':'uncached per-capture acquisition; development scenes; GT used only for post-hoc module rail checks'}
    if source_fingerprint([Path(__file__).resolve()]) != source:
        raise RuntimeError('Source changed during board audit; rerun with stable source')
    result['source_sha256']=source
    with args.output.open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
        stream.write('\n')
    print(json.dumps({path:{k:v for k,v in report.items() if k not in ('rows','capture_sha256')}
                      for path,report in result['directories'].items()},indent=2))
