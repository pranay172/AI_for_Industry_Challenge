#!/usr/bin/env python3
"""Privileged training labels for both SFP port faces of the requested NIC card.

Each capture records the target port pose at its image exposure. The public
card geometry turns that into the card pose and so both ports' entrance faces.
Visibility is geometric only (in frame, in front of the camera, face turned
toward it); occlusion by the gripper or cable is not modeled, and rows say so.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'aic_model'))
from aic_model.sfp_card_template import PORTS, card_landmarks_card
from aic_model.sfp_geometry import SFP_PORT_POSES_CARD, rpy_to_matrix
from replay_perception import digest, source_fingerprint

CORNERS = ('tl', 'tr', 'br', 'bl')
LANDMARK_NAMES = tuple(f'{port}_{name}' for port in PORTS for name in (*CORNERS, 'center'))


def card_in_camera(gt, port_name):
    """Card rotation/origin in the camera frame from the target port's GT pose."""
    R_cam_port = np.asarray(gt['R_cam_from_port'], dtype=float).reshape(3, 3)
    t_cam_port = np.asarray(gt['xyz_camera_port_origin'], dtype=float)
    pose = SFP_PORT_POSES_CARD[port_name]
    R_card_port = rpy_to_matrix(*pose['rpy'])
    R_cam_card = R_cam_port@R_card_port.T
    return R_cam_card, t_cam_port-R_cam_card@np.asarray(pose['translation'])


def label_view(gt, port_name, width, height):
    K = np.asarray(gt['intrinsics_k'], dtype=float).reshape(3, 3)
    R_cam_card, t_cam_card = card_in_camera(gt, port_name)
    points = np.einsum('ij,pkj->pki', R_cam_card, card_landmarks_card())+t_cam_card
    # Consistency with the separately recorded GT face center of the target.
    target = points[PORTS.index(port_name), 4]
    if np.linalg.norm(target-np.asarray(gt['xyz_camera'])) > 1e-6:
        raise ValueError('Card geometry disagrees with recorded GT face center')
    pixels = np.einsum('ij,pkj->pki', K, points)
    pixels = pixels[..., :2]/pixels[..., 2:]
    visible = []
    for index, port in enumerate(PORTS):
        axis = R_cam_card@rpy_to_matrix(*SFP_PORT_POSES_CARD[port]['rpy'])[:, 2]
        facing = float(axis@points[index, 4]) > 0.
        for point, pixel in zip(points[index], pixels[index]):
            visible.append(bool(facing and point[2] > .01 and 0 <= pixel[0] < width and 0 <= pixel[1] < height))
    pixels = pixels.reshape(-1, 2)
    return pixels, visible


def label_run(run):
    rows = []
    for scene in sorted(run.glob('scene-*')):
        for path in sorted((scene/'captures').glob('*.json')):
            metadata = json.loads(path.read_text())
            task = metadata['task']
            if task.get('plug_type') != 'sfp':
                continue
            archive = path.parent/metadata['files']['images_npz']
            for camera, gt in sorted(metadata.get('ground_truth', {}).items()):
                if not gt.get('R_cam_from_port') or gt.get('xyz_camera_port_origin') is None:
                    continue
                width, height = gt['image_size']
                pixels, visible = label_view(gt, task['port_name'], width, height)
                rows.append({
                    'sample_id': f"{path.stem}_{camera}", 'episode_id': metadata.get('episode_id', ''),
                    'scene_id': metadata.get('scene_id', ''), 'npz_path': str(archive.resolve()),
                    'image_key': f'{camera}_image', 'camera': camera, 'phase': metadata.get('phase', ''),
                    'task': {k: task.get(k) for k in ('id', 'plug_type', 'port_type', 'port_name', 'target_module_name')},
                    'image_size': [width, height], 'label_category': 'gt_projection_unreviewed',
                    'landmarks': {'names': list(LANDMARK_NAMES), 'points_px': pixels.tolist(),
                                  'points_norm': (pixels/[width, height]).tolist(), 'visible': visible,
                                  'visibility_rule': 'in frame, positive depth, face toward camera; occlusion not modeled'}})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists')
    rows = [row for run in args.runs for row in label_run(run)]
    if not rows:
        raise SystemExit('No SFP captures with ground truth')
    args.output.write_text(''.join(json.dumps(row, allow_nan=False)+'\n' for row in rows))
    args.output.with_suffix('.provenance.json').write_text(json.dumps({
        'runs': [str(run.resolve()) for run in args.runs], 'rows': len(rows),
        'labels_sha256': digest(args.output), 'source_sha256': source_fingerprint([Path(__file__).resolve()]),
        'visible_landmarks': int(sum(sum(r['landmarks']['visible']) for r in rows))}, indent=2)+'\n')
    print(f'{len(rows)} rows -> {args.output}')


if __name__ == '__main__':
    main()
