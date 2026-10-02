#!/usr/bin/env python3
"""Acquisition-only validation of the live policy, one isolated scene per run.

The model container runs `aic_model.ValidateAcquisition`: the unmodified live
loop registers the board, frames the rail, estimates, filters and locks, then
holds position while perception keeps running. No insertion is attempted, so
official scores are meaningless and are not reported. Ground truth is enabled
only so the policy can record the port face after the episode for error
reporting. This is not a qualification run.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import yaml

from benchmark import prepare, sha256, write_json, image_info, compose_config
from collect_initial_views import build_verified_image, isolated_compose, scene_id, simulator_health

CHECKPOINT_MOUNT = '/checkpoints'
# Cycles that end in board search, return, framing motion or a rail-view hold
# `continue` before the lock rule runs, so they are not lock evidence.
NON_LOCK_STATUSES = {'acquiring board reference', 'returning to start view', 'framing_requested_rail',
                     'waiting_for_camera_geometry', 'rail_view_distance_limit'}


def validation_compose(run, evaluator, model, scene, gpu, checkpoints, capture=False):
    config = compose_config(run, evaluator, model, gpu)
    ev, policy = config['services']['eval'], config['services']['model']
    # Ground truth is recorded by the policy only after each episode ends.
    ev['command'] = [arg.replace('ground_truth:=false', 'ground_truth:=true') for arg in ev['command']]
    for service in (ev, policy):
        service['environment']['AIC_ENABLE_ACL'] = 'false'
    policy['command'] = ['--ros-args', '-p', 'policy:=aic_model.ValidateAcquisition',
                         '-p', 'use_sim_time:=true']
    environment = policy['environment']
    environment.update(AIC_ACQUISITION_VALIDATION='1', AIC_ACQUISITION_TRACE_DIR='/traces',
                       AIC_ACQUISITION_SCENE_ID=scene)
    policy['volumes'] = [{'type': 'bind', 'source': str(run/'traces'), 'target': '/traces'}]
    if capture:
        # Diagnostic image capture: slows the loop, so its timing is not evaluation-like.
        environment.update(AIC_CAPTURE_SCENE_ID=scene, AIC_ACQUISITION_DIAGNOSTIC_CAPTURE='1')
        policy['volumes'].append({'type': 'bind', 'source': str(run/'captures'), 'target': '/captures'})
    else:
        del environment['AIC_CAPTURE_DIR']
    if checkpoints:
        policy['volumes'].append({'type': 'bind', 'source': str(run.parent/'checkpoints'),
                                  'target': CHECKPOINT_MOUNT, 'read_only': True})
        for variable, name in checkpoints.items():
            environment[variable] = f'{CHECKPOINT_MOUNT}/{name}'
    return config


# ── Trace analysis ─────────────────────────────────────────────────────────
def _stats(values):
    values = [float(v) for v in values if v is not None]
    if not values:
        return {'count': 0}
    return {'count': len(values), 'median': float(np.median(values)),
            'p95': float(np.percentile(values, 95)), 'max': float(max(values))}


def _spread_mm(points):
    """Maximum distance from the componentwise median, in millimetres."""
    if not points:
        return None
    points = np.asarray(points, dtype=float)
    return float(np.max(np.linalg.norm(points-np.median(points, axis=0), axis=1))*1000)


def _reason(row):
    if row['visible']:
        return row['filter']
    if not row['board_registered']:
        return 'board_unregistered'
    if row['filter'] == 'implausible_tcp_distance':
        return row['filter']
    return row['estimator_stage'] or row['rejection'] or 'not_visible'


def summarize_trace(trace):
    rows, events = trace['rows'], trace['events']
    truth = trace['ground_truth']['face_position']
    truth = None if truth is None else np.asarray(truth, dtype=float)

    def error_mm(point):
        return None if truth is None or point is None else float(np.linalg.norm(np.asarray(point)-truth)*1000)

    def components_mm(point):
        # base_link Z is vertical, the SC/SFP insertion axis; XY is lateral.
        if truth is None or point is None:
            return None
        delta = (np.asarray(point)-truth)*1000
        return {'lateral': float(np.linalg.norm(delta[:2])), 'vertical': float(delta[2])}

    locks = [e for e in events if e['event'] == 'lock']
    losses = [e for e in events if e['event'] == 'lock_lost']
    first = lambda predicate: next((r['t'] for r in rows if predicate(r)), None)
    result = {
        'outcome': trace['outcome'], 'scene_id': trace['scene_id'], 'task': trace['task'],
        'aborts': [f['message'] for f in trace['feedback'] if f['message'].startswith('abort:')],
        'cycles': len(rows), 'duration_s': rows[-1]['t'] if rows else 0.,
        'time_to_board_s': first(lambda r: r['board_registered']),
        'time_to_first_visible_s': first(lambda r: r['visible']),
        'time_to_first_lock_s': locks[0]['t'] if locks else None,
        'lock_count': len(locks), 'lock_losses': len(losses),
        'statuses': dict(Counter(r['status'] for r in rows if r['status'])),
        'reasons_before_lock': dict(Counter(_reason(r) for r in rows[:locks[0]['row'] if locks else None])),
        'sim_cycle_s': _stats(np.diff([r['t'] for r in rows])),
        'wall_cycle_s': _stats(np.diff([r['wall'] for r in rows])),
        'max_rail_distance_bound_m': max((r['rail_distance_bound'] for r in rows
                                          if r['rail_distance_bound'] is not None), default=None),
        'max_used_tcp_distance_m': max((r['used_tcp_distance'] for r in rows
                                        if r['used_tcp_distance'] is not None), default=None),
        'ground_truth_available': truth is not None,
    }
    if locks:
        index = locks[0]['row']
        # Cycles that produced the first lock: the last lock-window search
        # cycles before the phase change (the consecutive rule's window is
        # REQUIRED_LOCK_COUNT). Rows are traced before phase dispatch, so the
        # first counted cycle may still read 'initialize'.
        constants = trace['constants']
        size = constants.get('LOCK_WINDOW_CYCLES') or constants['REQUIRED_LOCK_COUNT']
        window = [r for r in rows[:index] if r['phase'] in {'initialize', 'find_target'}
                  and r['status'] not in NON_LOCK_STATUSES][-size:]
        result['lock_window'] = {
            'cycles': len(window), 'accepted': sum(r['visible'] for r in window),
            'distinct_center_exposures': len({r['exposures_ns'].get('center') for r in window}),
            'duration_s': window[-1]['t']-window[0]['t'] if window else None,
            'raw_spread_mm': _spread_mm([r['raw_position'] for r in window if r['raw_position'] and r['visible']]),
            'cameras': dict(Counter(r['source_cameras'] for r in window if r['visible'])),
            'used_error_at_lock_mm': error_mm(rows[index-1]['used_position'] if index else None),
        }
        end = next((e['row'] for e in losses if e['row'] >= index), len(rows))
        hold = rows[index:end+1]
        raw = [r['raw_position'] for r in hold if r['raw_position'] and r['filter'] != 'implausible_tcp_distance']
        used = [r['used_position'] for r in hold if r['used_position']]
        result['first_hold'] = {
            'cycles': len(hold), 'duration_s': hold[-1]['t']-hold[0]['t'] if hold else None,
            'visible_fraction': sum(r['visible'] for r in hold)/len(hold) if hold else None,
            'distinct_center_exposures': len({r['exposures_ns'].get('center') for r in hold}),
            'reasons': dict(Counter(_reason(r) for r in hold)),
            'raw_spread_mm': _spread_mm(raw), 'used_spread_mm': _spread_mm(used),
            'raw_error_mm': _stats(error_mm(p) for p in raw),
            'used_error_mm': _stats(error_mm(p) for p in used),
            'final_used_error_mm': error_mm(used[-1]) if used else None,
            'final_used_error_components_mm': components_mm(used[-1]) if used else None,
        }
    return result


def aggregate(summaries):
    acquired = [s for s in summaries if s['outcome'] == 'acquired']
    return {
        'episodes': len(summaries), 'outcomes': dict(Counter(s['outcome'] for s in summaries)),
        'aborts': dict(Counter(a for s in summaries for a in s['aborts'])),
        'time_to_first_lock_s': _stats(s['time_to_first_lock_s'] for s in summaries),
        'lock_losses': sum(s['lock_losses'] for s in summaries),
        'used_error_at_lock_mm': _stats(s.get('lock_window', {}).get('used_error_at_lock_mm') for s in summaries),
        'final_used_error_mm': _stats(s.get('first_hold', {}).get('final_used_error_mm') for s in acquired),
        'final_used_lateral_error_mm': _stats((s['first_hold'].get('final_used_error_components_mm') or {}).get('lateral')
                                              for s in acquired),
        'final_used_vertical_error_mm': _stats((s['first_hold'].get('final_used_error_components_mm') or {}).get('vertical')
                                               for s in acquired),
        'hold_raw_spread_mm': _stats(s.get('first_hold', {}).get('raw_spread_mm') for s in summaries),
        'max_used_tcp_distance_m': max((s['max_used_tcp_distance_m'] for s in summaries
                                        if s['max_used_tcp_distance_m'] is not None), default=None),
    }


def load_traces(run):
    traces = [json.loads(p.read_text()) for p in sorted((run/'traces').glob('*.json'))]
    return [(trace, summarize_trace(trace)) for trace in traces]


# ── Execution ──────────────────────────────────────────────────────────────
def validate(config, destination, evaluator, checkpoints, gpu=False, timeout=900, capture=False):
    destination = Path(destination).resolve()
    prepare(config, destination)
    manifest_path = destination/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest.update(purpose='acquisition_validation', ground_truth=True, acl_enabled=False, capture_enabled=capture,
                    scores_are_benchmark_results=False, insertion_attempted=False, scene_runs=[],
                    limitations='Acquisition and stationary hold only; no approach, contact or insertion.')
    mounted = {}
    if checkpoints:
        (destination/'checkpoints').mkdir()
        manifest['checkpoints'] = {}
        for variable, path in checkpoints.items():
            path = Path(path).resolve()
            name = f'{variable.lower()}-{path.name}'
            shutil.copy2(path, destination/'checkpoints'/name)
            mounted[variable] = name
            manifest['checkpoints'][variable] = {'origin': str(path), 'name': name,
                                                 'sha256': sha256(destination/'checkpoints'/name)}
    write_json(manifest_path, manifest)
    try:
        manifest['evaluator'] = image_info(evaluator)
        manifest['model'] = build_verified_image(destination, manifest, 'aic-collection-model:', timeout)
        manifest['installed_policy_verified'] = True
        scene_config = yaml.safe_load((destination/'config.yaml').read_text())
        manifest['status'] = 'running'
        for index, (name, trial) in enumerate(scene_config['trials'].items()):
            run = destination/f'scene-{index:04d}'
            run.mkdir()
            for directory in ('traces', 'results') + (('captures',) if capture else ()):
                (run/directory).mkdir(mode=0o777)
                (run/directory).chmod(0o777)
            (run/'config.yaml').write_text(yaml.safe_dump({**scene_config, 'trials': {name: trial}},
                                                          sort_keys=False))
            record = {'trial': name, 'scene_id': scene_id(trial), 'directory': run.name,
                      'config_sha256': sha256(run/'config.yaml'), 'status': 'running'}
            manifest['scene_runs'].append(record)
            write_json(manifest_path, manifest)
            (run/'compose.yaml').write_text(yaml.safe_dump(validation_compose(
                run, manifest['evaluator']['id'], manifest['model']['id'], record['scene_id'],
                gpu, mounted, capture)))
            timed_out = False
            try:
                with isolated_compose(run, 'aic-acquire-', timeout):
                    pass
            except subprocess.TimeoutExpired:
                # In one validation run the simulator died during the
                # post-trial reset and the engine waited forever. Record it and
                # go on; the scene and run stay marked, never 'traced'/'completed'.
                timed_out = True
            finally:
                record.update(simulator_health(run/'compose.log'))
                summaries = [summary for _, summary in load_traces(run)]
                record['episodes'] = summaries
                record['harness_timeout'] = timed_out
                record['status'] = ('timed_out' if timed_out else 'traced') if summaries else 'no_trace'
                write_json(manifest_path, manifest)
        manifest['summary'] = aggregate([s for r in manifest['scene_runs'] for s in r['episodes']])
        manifest['status'] = 'completed' if all(r['status'] == 'traced' for r in manifest['scene_runs']) else 'incomplete'
    except (Exception, KeyboardInterrupt) as exc:
        manifest.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        manifest['finished_at'] = datetime.now(timezone.utc).isoformat()
        write_json(manifest_path, manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--eval-image')
    parser.add_argument('--sc-checkpoint', type=Path, help='Mounted read-only in place of the default SC model')
    parser.add_argument('--sfp-checkpoint', type=Path, help='Mounted read-only in place of the default SFP model')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--timeout', type=int, default=900)
    parser.add_argument('--summarize', type=Path, help='Re-summarize an existing run directory')
    parser.add_argument('--capture', action='store_true',
                        help='Also save images with ground-truth labels for offline replay (slows the loop)')
    args = parser.parse_args()
    if args.summarize:
        summaries = [s for run in sorted(args.summarize.glob('scene-*')) for _, s in load_traces(run)]
        print(json.dumps({'episodes': summaries, 'summary': aggregate(summaries)}, indent=2))
        return 0
    if not (args.config and args.output and args.eval_image):
        parser.error('--config, --output and --eval-image are required')
    checkpoints = {variable: path for variable, path in (
        ('AIC_SC_PORT_DETECTOR_PATH', args.sc_checkpoint),
        ('AIC_SFP_DETECTOR_PATH', args.sfp_checkpoint)) if path is not None}
    result = validate(args.config, args.output, args.eval_image, checkpoints, args.gpu, args.timeout, args.capture)
    print(json.dumps({k: result.get(k) for k in ('status', 'summary')}, indent=2))
    return 0 if result['status'] == 'completed' else 1


if __name__ == '__main__':
    sys.exit(main())
