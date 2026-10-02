#!/usr/bin/env python3
"""Keep only scenes whose target port face is visible from the start pose.

Qualification guarantees the target port is within camera view at the start.
Randomly moving the target to another rail can break that, so evaluation scenes
are screened here: the evaluator-side configuration is projected through the
start camera model calibrated from a stationary collection run. This reads
scene configuration and GT and must never be imported by a policy.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'aic_model'))
from aic_model.sfp_card_template import port_pose_board
from aic_model.sfp_geometry import SFP_FACE_Z_PORT_M, rpy_to_matrix

SC_FACE_Z_PORT_M = -.01564


def board_in_world(trial):
    pose = trial['scene']['task_board']['pose']
    return rpy_to_matrix(pose['roll'], pose['pitch'], pose['yaw']), np.array([pose['x'], pose['y'], pose['z']])


def face_points_port(module, face_z):
    """Entrance-face corners and center in the port frame (label conventions)."""
    hw, hh = (.0125/2, .0085/2) if module.startswith('nic_card_mount_') else (.005, .0125)
    return np.array([[-hw, hh, face_z], [hw, hh, face_z], [hw, -hh, face_z], [-hw, -hh, face_z], [0., 0., face_z]])


def target_port_board(trial):
    """(R, t) of the target port link and its entrance-face offset, in the board frame."""
    task = next(iter(trial['tasks'].values()))
    board = trial['scene']['task_board']
    module = task['target_module_name']
    if module.startswith('nic_card_mount_'):
        rail = board[f'nic_rail_{module[-1]}']['entity_pose']
        R, t = port_pose_board(module, rail['translation'], rail['yaw'], task['port_name'])
        return R, t, SFP_FACE_Z_PORT_M
    index = int(module[-1])
    rail = board[f'sc_rail_{index}']['entity_pose']
    # Same chain as sc_face_decoder: sc_port_i model at (-0.075 + t, 0.0295 + 0.041 i, 0.0165)
    # with rpy (1.57, 0, 1.57), then the port link at rpy (1.5708, pi, 0), offset (0, -0.002, 0).
    R_model = rpy_to_matrix(1.57, 0., 1.57)
    origin = np.array([-.075+rail['translation'], .0295+.041*index, .0165])
    return R_model@rpy_to_matrix(1.5708, 3.14159, 0.), origin+R_model@[0., -.002, 0.], SC_FACE_Z_PORT_M


# A calibration capture must be taken before the arm leaves its start pose.
START_POSE_TOLERANCE_M = .001


def start_capture(scene):
    """Earliest capture with ground truth, if the TCP is still at the episode's first pose.

    Capture names carry an unpadded millisecond stamp, so name order is not time
    order; runs that move (marker search, framing) must be read by time.
    """
    captures = sorted((json.loads(path.read_text()) for path in (scene/'captures').glob('*.json')),
                      key=lambda metadata: metadata['capture_time_sim'])
    if not captures:
        return None
    position = lambda metadata: np.array([metadata['controller']['tcp_pose']['position'][a] for a in 'xyz'])
    for metadata in captures:
        if np.linalg.norm(position(metadata)-position(captures[0])) > START_POSE_TOLERANCE_M:
            return None
        if any(g.get('R_cam_from_port') for g in metadata['ground_truth'].values()):
            return metadata
    return None


def calibrate(run):
    """Start camera model and the fixed base<-world transform from start-pose captures."""
    for scene in sorted(Path(run).glob('scene-*')):
        config = yaml.safe_load((scene/'config.yaml').read_text())
        trial = next(iter(config['trials'].values()))
        metadata = start_capture(scene)
        if metadata is None:
            continue
        camera, gt = next((c, g) for c, g in sorted(metadata['ground_truth'].items()) if g.get('R_cam_from_port'))
        geometry = metadata['camera_geometry'][camera]
        R_base_cam, t_base_cam = np.array(geometry['R_base_from_camera']), np.array(geometry['t_base_from_camera'])
        R_base_port = R_base_cam@np.array(gt['R_cam_from_port']).reshape(3, 3)
        t_base_port = R_base_cam@np.array(gt['xyz_camera_port_origin'])+t_base_cam
        R_board_port, t_board_port, _ = target_port_board(trial)
        R_world_board, t_world_board = board_in_world(trial)
        R_world_port = R_world_board@R_board_port
        R_base_world = R_base_port@R_world_port.T
        t_base_world = t_base_port-R_base_world@(R_world_board@t_board_port+t_world_board)
        cameras = {name: {'K': np.reshape(metadata['ground_truth'][name]['intrinsics_k'], (3, 3)).tolist(),
                          'image_size': metadata['ground_truth'][name]['image_size'],
                          'R_base_from_camera': g['R_base_from_camera'], 't_base_from_camera': g['t_base_from_camera']}
                   for name, g in metadata['camera_geometry'].items() if name in metadata['ground_truth']}
        return {'R_base_world': R_base_world.tolist(), 't_base_world': t_base_world.tolist(), 'cameras': cameras,
                'source': str(scene)}
    raise ValueError('No calibration capture with ground truth')


def visible_cameras(trial, calibration):
    R_board_port, t_board_port, face_z = target_port_board(trial)
    R_world_board, t_world_board = board_in_world(trial)
    R_bw, t_bw = np.array(calibration['R_base_world']), np.array(calibration['t_base_world'])
    R_port = R_bw@R_world_board@R_board_port
    module = next(iter(trial['tasks'].values()))['target_module_name']
    board_points = (R_board_port@face_points_port(module, face_z).T).T+t_board_port
    world_points = (R_world_board@board_points.T).T+t_world_board
    points = (R_bw@world_points.T).T+t_bw
    face = points[4]
    visible = []
    for name, camera in calibration['cameras'].items():
        R, t, K = np.array(camera['R_base_from_camera']), np.array(camera['t_base_from_camera']), np.array(camera['K'])
        outward = -R_port[:, 2]                     # the entrance faces away from the port's insertion axis
        if float(outward@(t-face)) <= 0.:
            continue
        camera_points = (R.T@(points-t).T).T
        if np.any(camera_points[:, 2] <= .01):
            continue
        pixels = (K@camera_points.T).T
        pixels = pixels[:, :2]/pixels[:, 2:]
        width, height = camera['image_size']
        # The whole entrance face (four corners and center) must lie inside the image.
        if np.all((pixels >= 0) & (pixels < [width, height])):
            visible.append(name)
    return visible


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calibration-run', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='Screened config; a .screen.json report is written beside it')
    parser.add_argument('--min-cameras', type=int, default=1)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists')
    calibration = calibrate(args.calibration_run)
    config = yaml.safe_load(args.config.read_text())
    report = {name: visible_cameras(trial, calibration) for name, trial in config['trials'].items()}
    kept = {name: trial for name, trial in config['trials'].items() if len(report[name]) >= args.min_cameras}
    args.output.write_text(yaml.safe_dump({**config, 'trials': kept}, sort_keys=False))
    args.output.with_suffix('.screen.json').write_text(json.dumps(
        {'calibration': calibration, 'min_cameras': args.min_cameras, 'visible_cameras': report,
         'kept': list(kept)}, indent=1)+'\n')
    print(f'kept {len(kept)}/{len(report)}')


if __name__ == '__main__':
    main()
