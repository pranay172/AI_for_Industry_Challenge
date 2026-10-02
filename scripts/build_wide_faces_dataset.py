#!/usr/bin/env python3
"""Wide board-pose batch: labels -> rail crops -> split -> merged SC and SFP sets.

Scenes come from generate_scenes.py --board-pose wide (board pose sampled over
every organizer example, NIC cards over the physical rail travel), screened by
screen_scenes.py and captured with collect_initial_views.py --rail-views. The
train/validation split follows a plan file fixed before collection, mapping each
trial name to "train" or "validation".

Labels are privileged ground-truth projections (training only). Crops come from
runtime RGB board registration where it succeeds and from the ground-truth
board otherwise. Visibility is geometric; occlusion by the gripper is applied
afterwards by mask_gripper_landmarks.py. Each mode's new rows are appended to the
previous training set (--previous-sc / --previous-sfp, train then validation).

The shipped detectors were trained on the masked outputs of this script.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def run(*args):
    subprocess.run([sys.executable, *map(str, args)], check=True, cwd=ROOT)


def scene_dirs(runs, mode):
    dirs, trial_of = [], {}
    for run_dir in runs:
        for record in json.loads((run_dir/'manifest.json').read_text())['scene_runs']:
            scene = run_dir/record['directory']
            if (scene/'captures').is_dir() and any((scene/'captures').glob('*.json')):
                plug = json.loads(next((scene/'captures').glob('*.json')).read_text())['task']['plug_type']
                if plug == mode:
                    dirs.append(scene/'captures'); trial_of[str(scene)] = record['trial']
    return dirs, trial_of


def build(mode, runs, previous, plan, out):
    dirs, trial_of = scene_dirs(runs, mode)
    labels = out/f'labels-{mode}.jsonl'
    if mode == 'sc':
        run('aic_model/tools/label_sc_port_dataset.py', *dirs, '--output', labels)
    else:
        run('scripts/label_sfp_faces.py', '--runs', *sorted({d.parent.parent for d in dirs}), '--output', labels)
    run('scripts/prepare_rail_dataset.py', '--labels', labels, '--output', out/f'rail-{mode}.jsonl')
    run('scripts/prepare_rail_dataset.py', '--labels', labels, '--output', out/f'rail-{mode}-gtboard.jsonl', '--board-from-gt')
    key = lambda row: (row['npz_path'], row['image_key'])
    registered = {key(json.loads(line)): line for line in open(out/f'rail-{mode}.jsonl')}
    merged = [registered.get(key(json.loads(line)), line) for line in open(out/f'rail-{mode}-gtboard.jsonl')]
    (out/f'rail-{mode}-merged.jsonl').write_text(''.join(merged))
    print(mode, Counter(json.loads(line)['runtime_crop']['source'] for line in merged))
    parts = {'train': [], 'val': []}
    for line in merged:
        scene = str(Path(json.loads(line)['npz_path']).parent.parent)
        parts['val' if plan[trial_of[scene]] == 'validation' else 'train'].append(line)
    for (name, lines), old in zip(parts.items(), previous):
        (out/f'wide-{mode}-{name}.jsonl').write_text(''.join(lines))
        (out/f'all-{mode}-{name}.jsonl').write_text(old.read_text()+''.join(lines))
        print(mode, name, 'wide', len(lines), 'all', sum(1 for _ in open(out/f'all-{mode}-{name}.jsonl')))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--plan', type=Path, required=True, help='JSON with "partitions": {trial: train|validation}')
    ap.add_argument('--sc-runs', type=Path, nargs='+', required=True, help='collection runs with SC scenes')
    ap.add_argument('--sfp-runs', type=Path, nargs='+', required=True, help='collection runs with SFP scenes')
    ap.add_argument('--previous-sc', type=Path, nargs=2, required=True, metavar=('TRAIN', 'VAL'))
    ap.add_argument('--previous-sfp', type=Path, nargs=2, required=True, metavar=('TRAIN', 'VAL'))
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    plan = json.loads(a.plan.read_text())['partitions']
    build('sc', [p.resolve() for p in a.sc_runs], a.previous_sc, plan, a.output.resolve())
    build('sfp', [p.resolve() for p in a.sfp_runs], a.previous_sfp, plan, a.output.resolve())
