#!/usr/bin/env python3
"""Diagnose SC face-template decisions against known rail placement (development only).

Runs recorded views through the live estimator (via the replay adapter) and
compares the decoder's best placement and its distinct competitor with the
scene's configured rail translation. Scene configuration and GT are read only
for this post-hoc analysis; they never enter perception.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'aic_model'))
from replay_policy_pose import ReplayPolicy, evaluate_observation, observation
from aic_model.board_registration import module_crop
from aic_model.sc_face_decoder import DISTINCT_TRANSLATION_M, face_templates
from aic_model.sc_heatmap_detector import load_sc_port_heatmap
from types import SimpleNamespace as NS

RAIL_ORIGIN_M = -.075   # template translation = this + scene rail translation


def pixels_per_distinct_shift(policy, parsed, camera, module):
    """Heatmap pixels spanned by the decoder's minimum distinct translation."""
    from aic_model import policy_perception as perception
    geometry = perception.camera_projection_matrix(policy, parsed.camera_info_map.get(camera), parsed,
                                                   parsed.image_header_map[camera])
    if geometry is None or policy._board_pose is None:
        return None
    crop = module_crop(parsed.image_map[camera], *geometry, policy._board_pose, module)
    if crop is None:
        return None
    rgb, offset = crop
    templates = face_templates(policy._board_pose, geometry, module, offset, rgb.shape,
                               policy._sc_port_detector.heatmap_size)
    if templates is None:
        return None
    zero = np.isclose(templates['yaw'], 0.)
    x, points = templates['translation'][zero], templates['points'][zero]
    if len(x) < 2:
        return None
    step = float(np.median(np.linalg.norm(np.diff(points[:, 4], axis=0), axis=1)/np.diff(x)))
    return step*DISTINCT_TRANSLATION_M


def effective_template_yaw(board, metadata):
    """Template yaw (deg) and residual that best explain GT port rotation in the registered board frame.

    SC mounts have no public orientation randomization, so a nonzero yaw here
    is board-registration error the decoder's yaw search must absorb.
    """
    from aic_model.sfp_geometry import rpy_to_matrix
    for camera, gt in metadata.get('ground_truth', {}).items():
        geometry = metadata.get('camera_geometry', {}).get(camera)
        if board is None or not gt.get('R_cam_from_port') or geometry is None:
            continue
        R_base_port = np.asarray(geometry['R_base_from_camera'])@np.asarray(gt['R_cam_from_port']).reshape(3, 3)
        relative = board.rotation.T@R_base_port
        link = rpy_to_matrix(1.5708, 3.14159, 0.)
        yaws = np.radians(np.arange(-30., 30.001, .05))
        errors = [np.linalg.norm(rpy_to_matrix(1.57, 0., 1.57+y)@link-relative) for y in yaws]
        best = int(np.argmin(errors))
        return float(np.degrees(yaws[best])), float(errors[best])
    return None, None


def diagnose(run, detector):
    rows = []
    for scene in sorted(run.glob('scene-*')):
        config = yaml.safe_load((scene/'config.yaml').read_text())
        trial = next(iter(config['trials'].values()))
        task = next(iter(trial['tasks'].values()))
        if task['plug_type'] != 'sc':
            continue
        module = task['target_module_name']
        rail = trial['scene']['task_board'][f'sc_rail_{module[-1]}']['entity_pose']['translation']
        other = trial['scene']['task_board'][f'sc_rail_{1-int(module[-1])}']['entity_present']
        policy = ReplayPolicy(detector)
        state = NS(port_pos_smoothed=None, phase='find_target')
        for path in sorted((scene/'captures').glob('*.json')):
            metadata = json.loads(path.read_text())
            with np.load(path.parent/metadata['files']['images_npz']) as data:
                images = {key: data[key] for key in data.files}
            metadata['phase'] = 'find_target'
            result = evaluate_observation(policy, state, metadata, images)
            estimator = result['estimator'] or {}
            legal = {k: metadata.get(k, {}) for k in ('camera_exposures', 'camera_geometry', 'controller', 'capture_time_sim')}
            parsed = observation(policy, legal, images)
            truth = RAIL_ORIGIN_M+rail
            yaw_deg, yaw_residual = effective_template_yaw(policy._board_pose, metadata)
            for camera, status in (estimator.get('cameras') or {}).items():
                decoder = (estimator.get('decoder') or {}).get(camera, {})
                best, competitor = decoder.get('best'), decoder.get('competitor')
                rows.append({
                    'scene': scene.name, 'module': module, 'rail_translation': rail, 'other_present': other,
                    'camera': camera, 'status': status, 'pose_status': result['status'],
                    'best_error_mm': None if best is None else (best['translation']-truth)*1000,
                    'best_yaw_deg': None if best is None else float(np.degrees(best['yaw'])),
                    'competitor_error_mm': None if competitor is None else (competitor['translation']-truth)*1000,
                    'competitor_yaw_deg': None if competitor is None else float(np.degrees(competitor['yaw'])),
                    'margin': decoder.get('margin'), 'valid_placements': decoder.get('valid_placements'),
                    'effective_yaw_deg': yaw_deg, 'yaw_fit_residual': yaw_residual,
                    'distinct_shift_px': pixels_per_distinct_shift(policy, parsed, camera, module)})
    return rows


def summarize(rows):
    by = defaultdict(list)
    for row in rows:
        by[(row['module'], round(row['rail_translation'], 3), row['camera'])].append(row)
    table = []
    for (module, rail, camera), items in sorted(by.items()):
        best = [abs(r['best_error_mm']) for r in items if r['best_error_mm'] is not None]
        comp = [r['competitor_error_mm'] for r in items if r['competitor_error_mm'] is not None]
        table.append({'module': module, 'rail': rail, 'camera': camera, 'views': len(items),
                      'status': dict(Counter(r['status'] for r in items)),
                      'best_abs_error_mm_median': float(np.median(best)) if best else None,
                      'competitor_error_mm_median': float(np.median(comp)) if comp else None,
                      'margin_median': float(np.median([r['margin'] for r in items if r['margin'] is not None]))
                      if any(r['margin'] is not None for r in items) else None,
                      'distinct_shift_px_median': float(np.median([r['distinct_shift_px'] for r in items
                                                                   if r['distinct_shift_px'] is not None]))
                      if any(r['distinct_shift_px'] is not None for r in items) else None})
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists')
    rows = diagnose(args.run, load_sc_port_heatmap(args.checkpoint))
    result = {'rows': rows, 'table': summarize(rows)}
    args.output.write_text(json.dumps(result, indent=1, allow_nan=False)+'\n')
    for row in result['table']:
        print(json.dumps(row))


if __name__ == '__main__':
    main()
