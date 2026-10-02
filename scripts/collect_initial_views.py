#!/usr/bin/env python3
"""Collect privileged training captures, one isolated scene per simulator run.

This is NOT a benchmark. The robot stays at its initial pose unless --rail-views
explicitly selects bounded rail framing. Scene identity is
content-addressed so repeated collection of the same scene cannot cross splits.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import uuid
import yaml

from benchmark import (prepare, sha256, write_json, command, image_info,
                       compose_config, build_verified_image)


def scene_id(trial):
    return 'scene-sha256:' + hashlib.sha256(json.dumps(trial['scene'], sort_keys=True,
                                                     separators=(',', ':')).encode()).hexdigest()


def run_scene_id(trials):
    """A single trial's scene ID, or one covering the ordered scenes of a multi-trial run."""
    if len(trials) == 1:
        return scene_id(trials[0])
    return 'scenes-sha256:' + hashlib.sha256(json.dumps([t['scene'] for t in trials], sort_keys=True,
                                                      separators=(',', ':')).encode()).hexdigest()


def collection_compose(run, evaluator, model, scene, gpu, rail_views=False, keep_bags=True):
    config = compose_config(run, evaluator, model, gpu)
    ev, policy = config['services']['eval'], config['services']['model']
    if not keep_bags:
        # Training labels come from the captures; the evaluator's bags and scoring
        # go to an anonymous volume that teardown removes, never to the host.
        ev['volumes'] = [m if m['target'] != '/results' else {'type': 'volume', 'target': '/results'}
                         for m in ev['volumes']]
    ev['command'] = [arg.replace('ground_truth:=false', 'ground_truth:=true') for arg in ev['command']]
    for service in (ev, policy):
        service['environment']['AIC_ENABLE_ACL'] = 'false'
    policy_module = 'CaptureRailViews' if rail_views else 'CaptureInitialViews'
    policy['command'] = ['--ros-args', '-p', 'policy:=aic_model.'+policy_module,
                         '-p', 'use_sim_time:=true']
    policy['environment'].update(AIC_TRAINING_COLLECTION='1', AIC_CAPTURE_SCENE_ID=scene)
    return config


class isolated_compose:
    """Run run/compose.yaml once; always record container status and tear down."""
    def __init__(self, run, prefix, timeout):
        self.run, self.timeout = Path(run), timeout
        self.compose = ['docker', 'compose', '--project-name', prefix+uuid.uuid4().hex[:12],
                        '-f', str(self.run/'compose.yaml')]

    def __enter__(self):
        try:
            with (self.run/'compose.log').open('w') as log:
                command(self.compose+['up', '--abort-on-container-exit', '--exit-code-from', 'eval',
                                      '--pull', 'never'], stdout=log, stderr=subprocess.STDOUT,
                        timeout=self.timeout)
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *_):
        with (self.run/'container-status.json').open('w') as status:
            subprocess.run(self.compose+['ps', '--all', '--format', 'json'], stdout=status, timeout=30)
        command(self.compose+['down', '--timeout', '10', '--volumes'], timeout=60, stdout=subprocess.DEVNULL)
        return False


def simulator_health(log_path):
    """An outer successful exit does not imply a clean simulator run."""
    text = Path(log_path).read_text(errors='replace')
    return {'simulator_process_failures': [line for line in text.splitlines() if 'process has died' in line],
            'simulator_spawn_retry_count': text.count('Retrying ready_simulator')}


def collect(config, destination, evaluator, gpu=False, timeout=900, rail_views=False, trials_per_run=1,
            keep_bags=True):
    destination = Path(destination).resolve()
    # Reuse the source snapshotter, never the benchmark execution/scoring path.
    prepare(config, destination)
    manifest_path = destination / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest.update(purpose='training_rail_views' if rail_views else 'training_initial_views', ground_truth=True, acl_enabled=False,
                    scores_are_benchmark_results=False, bags_kept=keep_bags, scene_runs=[])
    write_json(manifest_path, manifest)
    try:
        manifest['evaluator'] = image_info(evaluator)
        manifest['model'] = build_verified_image(destination, manifest, 'aic-collection-model:', timeout)
        manifest['installed_policy_verified'] = True
        scene_config = yaml.safe_load((destination/'config.yaml').read_text())
        manifest['status'] = 'collecting'
        items = list(scene_config['trials'].items())
        # Stationary collection ends every trial where it began, at home, so the
        # evaluator's reset race cannot carry a pose into the next trial.
        groups = [dict(items[i:i+trials_per_run]) for i in range(0, len(items), trials_per_run)]
        for index, group in enumerate(groups):
            run = destination/f'scene-{index:04d}'
            run.mkdir()
            for directory in ('captures', 'results') if keep_bags else ('captures',):
                (run/directory).mkdir(mode=0o777)
                (run/directory).chmod(0o777)
            (run/'config.yaml').write_text(yaml.safe_dump({**scene_config, 'trials': group}, sort_keys=False))
            record = {'trial': next(iter(group)) if len(group) == 1 else list(group),
                      'scene_id': run_scene_id(list(group.values())), 'directory': run.name,
                      'config_sha256': sha256(run/'config.yaml'), 'status': 'running'}
            manifest['scene_runs'].append(record)
            write_json(manifest_path, manifest)
            (run/'compose.yaml').write_text(yaml.safe_dump(collection_compose(
                run, manifest['evaluator']['id'], manifest['model']['id'], record['scene_id'], gpu, rail_views,
                keep_bags)))
            with isolated_compose(run, 'aic-collect-', timeout*len(group)):
                metadata = list((run/'captures').glob('*.json'))
                record['captures'] = len(metadata)
                record['captures_with_gt'] = sum(any(g.get('xyz_camera') is not None or
                    g.get('visible', False) for g in json.loads(p.read_text()).get('ground_truth', {}).values())
                    for p in metadata)
                from audit_captures import audit
                capture_audit = audit(run/'captures')
                write_json(run/'capture-audit.json', capture_audit)
                record['capture_counts'] = capture_audit['counts']
                record.update(simulator_health(run/'compose.log'))
                record['status'] = 'captured' if record['captures_with_gt'] else 'no_ground_truth'
            write_json(manifest_path, manifest)
        manifest['status'] = 'completed' if all(r['status']=='captured' for r in manifest['scene_runs']) else 'incomplete'
    except (Exception, KeyboardInterrupt) as exc:
        manifest.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        manifest['finished_at'] = datetime.now(timezone.utc).isoformat()
        write_json(manifest_path, manifest)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--eval-image', required=True)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--timeout', type=int, default=900)
    parser.add_argument('--rail-views', action='store_true', help='Use bounded geometry-only view motion during collection')
    parser.add_argument('--trials-per-run', type=int, default=1,
                        help='Trials per simulator run (stationary collection only); the scene ID covers the run')
    parser.add_argument('--no-bags', action='store_true',
                        help='Discard the evaluator bags and scoring at teardown (training labels come from captures)')
    args = parser.parse_args()
    if args.trials_per_run < 1 or args.trials_per_run > 1 and args.rail_views:
        parser.error('--trials-per-run must be 1 with --rail-views')
    result = collect(args.config, args.output, args.eval_image, args.gpu, args.timeout, args.rail_views,
                     args.trials_per_run, not args.no_bags)
    print(json.dumps({k:result[k] for k in ('status', 'scene_runs')}, indent=2))
    sys.exit(0 if result['status']=='completed' else 1)
