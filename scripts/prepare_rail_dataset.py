#!/usr/bin/env python3
"""Attach runtime-selected rail windows to labels without moving original images.

Registration uses all preceding exposures, never label boxes or ground truth.
Unavailable windows are excluded with explicit counts in the provenance report.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'aic_model'))
from aic_model.board_registration import module_crop, module_pixels
from aic_model.dataset import resolve_capture
from replay_perception import register, digest, source_fingerprint


def board_from_ground_truth(metadata, config):
    """PRIVILEGED, training-only: board pose from the target port GT and configured placement.

    For views whose start pose never shows the board marker. The live policy
    registers the board after a bounded search instead; never use this at runtime.
    """
    from aic_model.board_registration import BoardPose
    from screen_scenes import target_port_board
    # The capture's own task names the target; the config supplies its placement.
    trial = {**next(iter(config['trials'].values())), 'tasks': {'task': metadata['task']}}
    # NIC card or SC port, through the same public geometry the decoders use.
    R_board_port, t_board_port, _ = target_port_board(trial)
    for camera, gt in sorted(metadata.get('ground_truth', {}).items()):
        geometry = metadata.get('camera_geometry', {}).get(camera)
        if not gt.get('R_cam_from_port') or gt.get('xyz_camera_port_origin') is None or geometry is None:
            continue
        R_base_cam = np.asarray(geometry['R_base_from_camera'], dtype=float)
        R_base_port = R_base_cam@np.asarray(gt['R_cam_from_port'], dtype=float).reshape(3, 3)
        t_base_port = R_base_cam@np.asarray(gt['xyz_camera_port_origin'])+np.asarray(geometry['t_base_from_camera'])
        rotation = R_base_port@R_board_port.T
        return BoardPose(rotation, t_base_port-rotation@t_board_port, 0.)
    return None


def prepare(labels, output, board_from_gt=False):
    report_path=output.with_suffix('.provenance.json')
    if output.exists() or report_path.exists():
        raise ValueError('Output already exists')
    source=source_fingerprint([Path(__file__).resolve()])
    input_hash=digest(labels)
    rows=[json.loads(line) for line in labels.read_text().splitlines() if line.strip()]
    indexed={}
    for row in rows:
        path=resolve_capture(row.get('npz_path',''),labels)
        key=(path,row['image_key'])
        if key in indexed:
            raise ValueError(f'Duplicate label row: {key}')
        indexed[key]=row
    records=[]
    for directory in sorted({Path(key[0]).parent for key in indexed}):
        for path in directory.glob('*.json'):
            metadata=json.loads(path.read_text())
            if 'files' in metadata and 'images_npz' in metadata['files']:
                records.append((path,metadata))
    records.sort(key=lambda item:(str(item[0].parent),item[1].get('episode_id',''),item[1]['capture_time_sim']))
    boards={};counts=Counter();prepared=[];seen=set();hashes={}
    for path,metadata in records:
        archive=(path.parent/metadata['files']['images_npz']).resolve()
        hashes[str(path)]=digest(path);hashes[str(archive)]=digest(archive)
        with np.load(archive,allow_pickle=False) as data:
            images={key:data[key] for key in data.files}
        episode=(str(path.parent),metadata.get('episode_id') or str(path))
        board,names,geometry=register(metadata,images,boards.get(episode))
        boards[episode]=board
        if board_from_gt:
            import yaml
            board=board_from_ground_truth(metadata,yaml.safe_load((path.parent.parent/'config.yaml').read_text()))
        for camera in ('center','left','right'):
            key=(str(archive),camera+'_image')
            if key not in indexed:
                continue
            seen.add(key);row=indexed[key]
            if row.get('task',{}).get('target_module_name') != metadata['task']['target_module_name']:
                raise ValueError(f'Label/capture target mismatch: {key}')
            if board is None:
                counts['board_unavailable']+=1
                continue
            if camera not in names or camera not in geometry:
                counts['camera_unavailable']+=1
                continue
            crop=module_crop(images[key[1]],*geometry[camera],board,metadata['task']['target_module_name'])
            if crop is None:
                counts['crop_unavailable']+=1
                continue
            image,offset=crop
            low=offset.astype(int)
            box=[int(low[0]),int(low[1]),int(low[0]+image.shape[1]),int(low[1]+image.shape[0])]
            prepared.append({**row,'npz_path':str(archive),'runtime_crop':{
                'preprocessing':'rail_crop_v1','source':'privileged_gt_board' if board_from_gt else 'rgb_board_registration','box_xyxy':box,
                'rail_hull_norm':((module_pixels(*geometry[camera],board,metadata['task']['target_module_name'])-offset)/[image.shape[1],image.shape[0]]).tolist(),
                'metadata_sha256':hashes[str(path)],'capture_sha256':hashes[str(archive)]}})
            counts['prepared']+=1
    if set(indexed)!=seen:
        raise ValueError(f'{len(set(indexed)-seen)} labels lack matching capture metadata')
    if not prepared:
        raise ValueError(f'No runtime crops available: {dict(counts)}')
    if source_fingerprint([Path(__file__).resolve()])!=source or digest(labels)!=input_hash:
        raise RuntimeError('Inputs changed during preparation')
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as stream:
        for row in prepared:
            stream.write(json.dumps(row,allow_nan=False)+'\n')
    report={'input_labels_sha256':input_hash,'output_labels_sha256':digest(output),
            'source_sha256':source,'capture_sha256':hashes,'counts':dict(counts),
            'scene_ids':sorted({row.get('scene_id','') for row in prepared}),
            'board_source':'privileged_gt_board' if board_from_gt else 'rgb_board_registration',
            'limitations':'frustum labels do not verify occlusion; repeated views are not independent scenes'}
    with report_path.open('x') as stream:
        json.dump(report,stream,indent=2);stream.write('\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--board-from-gt',action='store_true',
                        help='PRIVILEGED training-only crops for views without a visible board marker')
    args=parser.parse_args()
    print(json.dumps(prepare(args.labels,args.output,args.board_from_gt)['counts'],indent=2))
