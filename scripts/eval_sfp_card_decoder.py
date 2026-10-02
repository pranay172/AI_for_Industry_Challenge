#!/usr/bin/env python3
"""Post-hoc SFP card-decoder accuracy on labeled views (development only).

Uses each row's prepared crop and board source (runtime registration or the
privileged GT-derived board recorded by prepare_rail_dataset). This isolates the
network and card decoder from board registration; it is not a runtime replay.
Configured card placement and labels are read only for scoring.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'aic_model')]
from prepare_rail_dataset import board_from_ground_truth
from replay_perception import register
from aic_model.board_registration import module_crop, module_pixels, module_support_mask
from aic_model.sfp_face_decoder import card_templates, load_sfp_face_heatmap


def evaluate(rows, detector):
    results, boards = [], {}
    for row in rows:
        archive = Path(row['npz_path'])
        metadata = json.loads(archive.with_suffix('.json').read_text())
        config = yaml.safe_load((archive.parent.parent/'config.yaml').read_text())
        with np.load(archive) as data:
            images = {key: data[key] for key in data.files}
        camera = row['camera']
        episode = str(archive.parent)
        if row['runtime_crop']['source'] == 'privileged_gt_board':
            board = board_from_ground_truth(metadata, config)
            _, _, geometry = register(metadata, images, board)
        else:
            board, _, geometry = register(metadata, images, boards.get(episode))
            boards[episode] = board
        module = row['task']['target_module_name']
        result = {'sample_id': row['sample_id'], 'camera': camera, 'module': module, 'port': row['task']['port_name']}
        results.append(result)
        if board is None or camera not in geometry:
            result['status'] = 'board_or_geometry_unavailable'
            continue
        crop = module_crop(images[row['image_key']], *geometry[camera], board, module)
        if crop is None:
            result['status'] = 'crop_unavailable'
            continue
        rgb, offset = crop
        support = module_support_mask(geometry[camera], board, module, offset, rgb.shape,
                                      (detector.heatmap_size, detector.heatmap_size))
        templates = card_templates(board, geometry[camera], module, offset, rgb.shape, detector.heatmap_size)
        hull = (module_pixels(*geometry[camera], board, module)-offset)/[rgb.shape[1], rgb.shape[0]]
        prediction = detector.infer(rgb, support_mask=support, face_templates=templates, rail_hull=hull) \
            if support is not None and templates is not None else None
        diagnostics = getattr(detector, 'last_decoder_diagnostics', None) or {}
        if prediction is None:
            result['status'] = getattr(detector, 'last_rejection_reason', None) or 'no_prediction'
            continue
        rail = config and next(iter(config['trials'].values()))['scene']['task_board'][f'nic_rail_{module[-1]}']['entity_pose']
        target = 5*int(row['task']['port_name'][-1])+4
        truth = np.asarray(row['landmarks']['points_px'])[target]
        predicted = np.asarray(prediction['points_px'])[target]+offset
        result.update(status='decoded', translation_error_mm=(prediction['rail_translation']-rail['translation'])*1000,
                      yaw_error_deg=float(np.degrees(prediction['rail_yaw']-rail['yaw'])),
                      target_center_error_px=float(np.linalg.norm(predicted-truth)),
                      target_visible=bool(row['landmarks']['visible'][target]), margin=diagnostics.get('margin'))
    return results


def summarize(results):
    decoded = [r for r in results if r['status'] == 'decoded']
    pitch = [r for r in decoded if abs(r['translation_error_mm']) > 11.6]   # half a port pitch
    summary = {'views': len(results), 'status': dict(Counter(r['status'] for r in results)),
               'port_pitch_confusions': len(pitch)}
    for key in ('translation_error_mm', 'yaw_error_deg', 'target_center_error_px'):
        values = np.abs([r[key] for r in decoded])
        if len(values):
            summary[key] = {'p50': float(np.median(values)), 'p95': float(np.percentile(values, 95)), 'max': float(values.max())}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels', type=Path, required=True, help='Prepared rail labels (rows with runtime_crop)')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists')
    rows = [json.loads(line) for line in args.labels.read_text().splitlines() if line.strip()]
    results = evaluate(rows, load_sfp_face_heatmap(args.checkpoint))
    output = {'summary': summarize(results), 'results': results}
    args.output.write_text(json.dumps(output, indent=1, allow_nan=False)+'\n')
    print(json.dumps(output['summary'], indent=1))


if __name__ == '__main__':
    main()
