#!/usr/bin/env python3
"""Replay SC/SFP pose acceptance through the live estimator and position filter.

Only recorded RGB, exposure camera geometry, task identity and TCP position enter
perception. Ground truth is used after acceptance, for error reporting only.
Sparse capture history cannot reproduce control-driven phase transitions.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'aic_model'))
from rclpy.time import Time
from std_msgs.msg import Header
from tf2_ros import TransformException
from aic_model.policy import Policy
from aic_model import policy_perception as perception
from aic_model.policy_geometry import matrix_to_quaternion
from replay_perception import digest, source_fingerprint


class ExposureTransforms:
    """Exact-exposure, camera-only TF adapter: no latest or hidden-object lookup."""
    def __init__(self, exposures, geometry):
        self.transforms = {}
        for name, exposure in exposures.items():
            if name not in geometry:
                continue
            item = geometry[name]
            R = np.asarray(item['R_base_from_camera'], dtype=float).reshape(3,3)
            t = np.asarray(item['t_base_from_camera'], dtype=float).reshape(3)
            if not np.isfinite(R).all() or not np.isfinite(t).all():
                continue
            stamp = exposure['sec']*1_000_000_000+exposure['nanosec']
            self.transforms[(exposure['frame_id'], stamp)] = NS(transform=NS(
                rotation=matrix_to_quaternion(R), translation=NS(x=t[0],y=t[1],z=t[2])))

    def lookup_transform(self, target, source, stamp):
        if target != 'base_link' or (source, stamp.nanoseconds) not in self.transforms:
            raise TransformException('Only recorded exposure camera transforms are available')
        return self.transforms[(source, stamp.nanoseconds)]


class ReplayPolicy(Policy):
    """Use live policy constants and functions without starting a ROS controller."""
    def __init__(self, detector, mode='sc'):
        self._sc_port_detector = detector if mode == 'sc' else None
        self._sfp_detector = detector if mode == 'sfp' else None
        self.mode = mode
        self._board_pose = None
        self.warnings = []
        self.now_ns = 0

    def _warn_throttled(self, key, message):
        self.warnings.append({'key':key, 'message':message})

    def get_logger(self):
        return NS(info=lambda _:None)

    def time_now(self):
        return Time(nanoseconds=self.now_ns)


def observation(policy, metadata, images):
    exposures = metadata.get('camera_exposures', {})
    geometry = metadata.get('camera_geometry', {})
    headers = {}; infos = {}
    for name, e in exposures.items():
        header = Header(frame_id=e['frame_id'], stamp=Time(
            nanoseconds=e['sec']*1_000_000_000+e['nanosec']).to_msg())
        headers[name] = header
        if name in geometry:
            infos[name] = NS(header=header, k=np.asarray(geometry[name]['K']).reshape(-1).tolist())
    tcp = metadata.get('controller', {}).get('tcp_pose', {}).get('position')
    if tcp is None or not np.isfinite([tcp[axis] for axis in ('x','y','z')]).all():
        raise ValueError('Pose replay requires a finite recorded TCP position')
    policy._parent_node = NS(_tf_buffer=ExposureTransforms(exposures, geometry))
    policy.now_ns = round(metadata['capture_time_sim']*1e9)
    return NS(image_map={name:images[name+'_image'] for name in ('center','left','right')
                         if name+'_image' in images}, image_header_map=headers,
              camera_info_map=infos, tcp_pose=NS(position=NS(**tcp)))


def evaluate_observation(policy, state, metadata, images):
    # Deliberate allowlist. Capture target/GT/plug-tip records never enter policy.
    legal = {key:metadata.get(key, {}) for key in
             ('camera_exposures','camera_geometry','controller','capture_time_sim')}
    parsed = observation(policy, legal, images)
    task = NS(target_module_name=metadata['task']['target_module_name'],
              port_name=metadata['task']['port_name'])
    state.phase = metadata['phase']
    policy.warnings = []
    policy._last_sc_pose_diagnostics = policy._last_sfp_pose_diagnostics = None
    mode = getattr(policy, 'mode', 'sc')
    target = perception.estimate_target(policy, parsed, mode, state.phase, task=task)
    raw = target.port_pos_base_link.copy() if target.port_pos_base_link is not None else None
    status = perception.filter_target_position(policy, target, parsed, state)
    diagnostics = getattr(policy, f'_last_{mode}_pose_diagnostics')
    if status == 'no_visible_pose':
        status = diagnostics['stage'] if diagnostics else target.rejection_reason
    return {'status':status, 'visible':target.visible,
            'raw_face_position':raw.tolist() if raw is not None else None,
            'used_face_position':target.port_pos_base_link.tolist() if target.visible else None,
            'used_rotation':target.port_rot_base_link.tolist() if target.visible and target.port_rot_base_link is not None else None,
            'source_cameras':target.source_camera, 'confidence':target.confidence,
            'estimator':diagnostics, 'rejection_reason':target.rejection_reason,
            'warnings':policy.warnings}


def rotation_error_deg(truth, estimate):
    """Estimate relative to truth in the true port frame: yaw about the port axis and axis tilt."""
    relative = np.asarray(truth).T@np.asarray(estimate)
    return {'yaw_error_deg':float(np.degrees(np.arctan2(relative[1,0], relative[0,0]))),
            'tilt_error_deg':float(np.degrees(np.arccos(np.clip(relative[2,2], -1., 1.))))}


def summarize(rows):
    result = {'captures':len(rows), 'status_counts':dict(Counter(r['status'] for r in rows)),
              'visible_poses':sum(r['visible'] for r in rows)}
    for kind in ('raw','used'):
        errors = [r[kind+'_center_error_mm'] for r in rows if r.get(kind+'_center_error_mm') is not None]
        result[kind+'_center_error_mm'] = {'count':len(errors), 'median':float(np.median(errors)) if errors else None,
            'max':max(errors) if errors else None, 'p95':float(np.percentile(errors,95)) if errors else None}
    def pose_source(row):
        return str(row.get('rejection_reason') or '').rpartition('pose=')[2] or 'unknown'
    for source in sorted({pose_source(r) for r in rows if r.get('yaw_error_deg') is not None}):
        yaw = np.abs([r['yaw_error_deg'] for r in rows if r.get('yaw_error_deg') is not None and pose_source(r) == source])
        result.setdefault('abs_yaw_error_deg', {})[source] = {'count':len(yaw), 'median':float(np.median(yaw)),
            'p90':float(np.percentile(yaw, 90)), 'max':float(yaw.max())}
    return result


def replay(captures, detector, mode='sc'):
    records = []
    for path in sorted({p.resolve() for directory in captures for p in directory.glob('*.json')}):
        metadata = json.loads(path.read_text())
        if metadata.get('task', {}).get('plug_type') == mode:
            records.append((path, metadata))
    records.sort(key=lambda pair:(str(pair[0].parent),pair[1].get('episode_id',''),pair[1]['capture_time_sim']))
    policies = {}; states = {}; rows = []; hashes = {}
    for path, m in records:
        archive = path.parent/m['files']['images_npz']
        hashes[str(path)] = digest(path); hashes[str(archive)] = digest(archive)
        with np.load(archive, allow_pickle=False) as data:
            images = {key:data[key] for key in data.files}
        if any(im.dtype != np.uint8 or im.ndim != 3 or im.shape[2] != 3 for im in images.values()):
            raise ValueError('Expected uint8 RGB captures')
        episode = (str(path.parent), m.get('episode_id') or str(path))
        if episode not in policies:
            policies[episode] = ReplayPolicy(detector, mode)
            states[episode] = NS(port_pos_smoothed=None, phase=m['phase'])
        row = evaluate_observation(policies[episode], states[episode], m, images)
        row.update(capture=str(archive), scene_id=m.get('scene_id',''), phase=m['phase'])
        # Post-hoc scoring only. Never use GT to choose views, poses or acceptance.
        if row['visible']:
            for camera in ('center','left','right'):
                gt = m.get('ground_truth', {}).get(camera, {}).get('xyz_camera')
                geom = m.get('camera_geometry', {}).get(camera)
                if gt is None or geom is None:
                    continue
                truth = np.asarray(geom['R_base_from_camera'])@np.asarray(gt)+geom['t_base_from_camera']
                if not np.isfinite(truth).all():
                    continue
                for kind in ('raw','used'):
                    row[kind+'_center_error_mm'] = float(np.linalg.norm(np.asarray(row[kind+'_face_position'])-truth)*1000)
                rotation = m['ground_truth'][camera].get('R_cam_from_port')
                if rotation is not None and row['used_rotation'] is not None:
                    row.update(rotation_error_deg(np.asarray(geom['R_base_from_camera'])@np.reshape(rotation, (3, 3)),
                                                  np.asarray(row['used_rotation'])))
                break
        rows.append(row)
    if not rows:
        raise ValueError(f'No {mode.upper()} captures')
    return {'summary':summarize(rows), 'by_scene':{scene:summarize([r for r in rows if r['scene_id']==scene])
            for scene in sorted({r['scene_id'] for r in rows})}, 'rows':rows, 'capture_sha256':hashes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--captures', type=Path, nargs='+', required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('sc', 'sfp'), default='sc')
    args = parser.parse_args()
    if args.output.exists():parser.error('Output already exists')
    source = source_fingerprint([Path(__file__).resolve()])
    if args.mode == 'sc':
        from aic_model.sc_heatmap_detector import load_sc_port_heatmap as load
    else:
        from aic_model.sfp_face_decoder import load_sfp_face_heatmap as load   # same loader as the policy
    result = replay(args.captures, load(args.checkpoint), args.mode)
    if source != source_fingerprint([Path(__file__).resolve()]):
        raise RuntimeError('Source changed during replay')
    result.update(checkpoint_sha256=digest(args.checkpoint), source_sha256=source,
        policy_constants={key:getattr(Policy,key) for key in ('SC_PORT_HEATMAP_MIN_CAMERAS',
            'SC_PORT_HEATMAP_MIN_CONFIDENCE','SC_PORT_HEATMAP_MAX_GEOMETRY_RESIDUAL_M',
            'SC_PORT_SPATIAL_LOCK_SEARCH_M','SC_PORT_SPATIAL_LOCK_CLOSE_M','PORT_POS_EMA_ALPHA',
            'PLAUSIBLE_PORT_DISTANCE_MIN_M','PLAUSIBLE_PORT_DISTANCE_MAX_M','SFP_CAGE_HEATMAP_MIN_CAMERAS',
            'SFP_CAGE_HEATMAP_MIN_CONFIDENCE')}, mode=args.mode,
        limitations='Recorded phases, sparse history and no prior insertion-axis state; estimator and position filtering only, not motion/contact/insertion success. Consumed development data, not fresh held-out evaluation.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(result['summary'],indent=2))


if __name__ == '__main__':main()
