#!/usr/bin/env python3
"""Replay detector inputs and RGB board acquisition from recorded exposures.

Labels are used only after inference, for original-image pixel metrics. Missing
registration/crops remain in the denominator. This measures input/detector
coverage, not closed-loop insertion or the complete multiview pose pipeline.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'aic_model'), str(ROOT/'aic_model/tools')]
import numpy as np
from aic_model.board_registration import register_views, infer_module
from aic_model.camera_timing import synchronized_exposures


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_fingerprint(extra=()):
    files = [Path(__file__).resolve(), *extra,
             ROOT/'aic_model/tools/train_sc_port_detector.py',
             *sorted((ROOT/'aic_model/aic_model').glob('*.py'))]
    return {str(p.relative_to(ROOT)):digest(p) for p in files}


def capture_inputs(metadata, images):
    exposures = metadata.get('camera_exposures', {})
    stamps = {name: int(exposures[name]['sec'])*1_000_000_000+int(exposures[name]['nanosec'])
              for name in ('center','left','right')
              if name in exposures and name+'_image' in images}
    names = synchronized_exposures(stamps, round(metadata['capture_time_sim']*1e9))
    geometry = {}
    for name in names:
        item = metadata.get('camera_geometry', {}).get(name)
        if item is None:
            continue
        values = (np.asarray(item['K'], dtype=float).reshape(3,3),
                  np.asarray(item['R_base_from_camera'], dtype=float).reshape(3,3),
                  np.asarray(item['t_base_from_camera'], dtype=float).reshape(3))
        if all(np.isfinite(v).all() for v in values):
            geometry[name] = values
    return names, geometry


def register(metadata, images, cached):
    names, geometry = capture_inputs(metadata, images)
    if cached is None:
        cached = register_views({name:(images[name+'_image'], *geometry[name])
                                 for name in names if name in geometry})
    return cached, names, geometry


def summarize(rows):
    failures = Counter(row['status'] for row in rows)
    errors = [e for row in rows for e in row.get('errors_px', []) if e is not None]
    visible = sum(row['visible_landmarks'] for row in rows)
    hits = sum(e <= 10 for e in errors)
    return {'rows': len(rows), 'status_counts': dict(failures),
            'prediction_coverage': failures['predicted']/len(rows) if rows else None,
            'visible_gt_landmarks': visible, 'measured_landmarks': len(errors),
            'mean_error_px_when_predicted': float(np.mean(errors)) if errors else None,
            'p95_error_px_when_predicted': float(np.percentile(errors,95)) if errors else None,
            'within_10px_including_missed_predictions': hits/visible if visible else None}


def replay(captures, detector, mode, labels=None):
    label_rows = {}
    if labels:
        from train_sc_port_detector import SPECS, load_manifest
        for sample in load_manifest(labels, SPECS['sfp_faces' if mode=='sfp' else 'sc']):
            key = (str(Path(sample.npz_path).resolve()), sample.image_key)
            if key in label_rows:
                raise ValueError(f'Duplicate label row: {key}')
            label_rows[key] = sample
    metadata_files = sorted({p.resolve() for directory in captures for p in directory.glob('*.json')})
    records = [(p, json.loads(p.read_text())) for p in metadata_files]
    records = [(p,m) for p,m in records if m.get('task',{}).get('plug_type','').lower()==mode]
    # Each episode's complete history is processed, including unlabelled frames
    # that may establish board registration before an evaluated frame.
    records.sort(key=lambda item:(str(item[0].parent), item[1].get('episode_id',''),
                                   item[1]['capture_time_sim'], item[1]['sample_id']))
    boards, rows, inventory, seen_labels = {}, [], {}, set()
    for path, metadata in records:
        npz = path.parent/metadata['files']['images_npz']
        inventory[str(path)] = digest(path)
        inventory[str(npz)] = digest(npz)
        with np.load(npz, allow_pickle=False) as archive:
            images = {key: archive[key] for key in archive.files}
        if any(image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 for image in images.values()):
            raise ValueError(f'Invalid RGB capture: {npz}')
        episode = (str(path.parent), metadata.get('episode_id') or str(path))
        board, names, geometry = register(metadata, images, boards.get(episode))
        boards[episode] = board
        for camera in ('center','left','right'):
            key = (str(npz.resolve()), camera+'_image')
            sample = label_rows.get(key)
            if labels and sample is None:
                continue
            if sample is not None:
                seen_labels.add(key)
                if sample.target_module_name != metadata['task']['target_module_name']:
                    raise ValueError(f'Label task differs from captured task: {key}')
            visible = np.asarray(sample.visible, dtype=bool) if sample else np.array([], dtype=bool)
            row = {'capture': str(npz), 'camera': camera, 'scene_id': metadata.get('scene_id',''),
                   'episode_id': metadata.get('episode_id',''), 'phase':metadata.get('phase',''),
                   'board_registered':board is not None, 'visible_landmarks':int(visible.sum())}
            rgb = images.get(camera+'_image')
            if rgb is None:
                row['status'] = 'missing_image'
            elif camera not in names:
                row['status'] = 'missing_or_stale_exposure'
            elif detector.preprocessing in {'target_crop_v1', 'rail_crop_v1', 'rail_conditioned_v1'} and camera not in geometry:
                row['status'] = 'missing_geometry'
            elif detector.preprocessing in {'target_crop_v1', 'rail_crop_v1', 'rail_conditioned_v1'} and board is None:
                row['status'] = 'board_not_registered'
            else:
                prediction = infer_module(detector, rgb, geometry.get(camera), board,
                                          metadata['task']['target_module_name'])
                if prediction is None:
                    row['status'] = 'no_usable_prediction'
                    row['rejection_reason'] = getattr(detector, 'last_rejection_reason', None) or 'crop_or_geometry_unavailable'
                else:
                    points = np.asarray(prediction['points_px'])
                    row.update(status='predicted', points_px=points.tolist(),
                               confidence=np.asarray(prediction['confidence']).tolist())
                    if sample:
                        gt = np.asarray(sample.points_norm)*[rgb.shape[1],rgb.shape[0]]
                        if points.shape != gt.shape or not np.isfinite(points).all() or not np.isfinite(gt[visible]).all():
                            raise ValueError(f'Invalid landmark values or shape: {key}')
                        errors = np.linalg.norm(points-gt, axis=1)
                        row['errors_px'] = [float(e) if v else None for e,v in zip(errors,visible)]
            rows.append(row)
    if labels and set(label_rows) != seen_labels:
        raise ValueError(f'{len(set(label_rows)-seen_labels)} label rows lack matching capture metadata')
    if not rows:
        raise ValueError('No matching capture rows')
    return {'summary':summarize(rows), 'by_scene':{scene:summarize([r for r in rows if r['scene_id']==scene])
             for scene in sorted({r['scene_id'] for r in rows})}, 'rows':rows, 'capture_sha256':inventory}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--captures', type=Path, nargs='+', required=True)
    parser.add_argument('--mode', choices=['sfp','sc'], required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--labels', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output exists; use a new experiment path')
    source = source_fingerprint()
    if args.mode=='sfp':
        from aic_model.sfp_face_decoder import load_sfp_face_heatmap as load
    else:
        from aic_model.sc_heatmap_detector import load_sc_port_heatmap as load
    detector = load(args.checkpoint)
    result = replay(args.captures, detector, args.mode, args.labels)
    if source_fingerprint() != source:
        raise RuntimeError('Source changed during replay; rerun with stable source')
    result.update(checkpoint_sha256=digest(args.checkpoint), preprocessing=detector.preprocessing,
                  decoder=detector.decoder, coordinate_space='original_image_pixels',
                  labels_sha256=digest(args.labels) if args.labels else None,
                  independence='development diagnostic; no held-out generalization claim',
                  visibility='GT frustum labels do not establish absence of occlusion',
                  source_sha256=source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(result['summary'], indent=2))


if __name__ == '__main__':
    main()
