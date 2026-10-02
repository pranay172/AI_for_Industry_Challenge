#!/usr/bin/env python3
"""Privileged training scenes for the plug-in-gripper estimator (train_plug_pose.py).

Each trial is a generate_scenes.py scene whose cable spawns with a perturbed
grasp. The engine spawns the cable at the gripper's world position plus the
config offset, in the config's world orientation, so a TCP-frame grasp change D
is applied as G_h D G_h^-1 to the normal spawn pose (G_h: the home gripper
pose). The realised grasp is labelled from ground truth during collection, not
from D.

The perturbations follow the shifted grasps measured post-hoc in evaluator
bags after the evaluator's reset race: rotation about TCP x of +7 to +12 deg,
y and z within a few degrees (one SC trial: 34 deg about z), and translations
of 1-10 mm. The shipped plug-pose models were trained on

  make_grasp_scenes.py --seed 20264101 --trials 160 --profile v1 --output grasp-train-001.yaml
  make_grasp_scenes.py --seed 20264102 --trials 160 --profile v2 --output grasp-train-002.yaml

collected with collect_initial_views.py. Held-out seeds 20261901-30 are refused.
"""
import argparse, json, math, random, sys
from pathlib import Path
import numpy as np, yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_scenes import generate

# Home gripper orientation in the world frame, from the evaluator's /scoring/tf in a normal trial.
RH = np.array([[-0.9999997359374777, -0.0007261087928974947, 2.984955385831589e-05],
               [-0.0007261095590884917, 0.999999736053058, -2.5665596075451885e-05],
               [-2.9830909964572234e-05, -2.5687263344544332e-05, -0.9999999992251412]])


def rotvec(v):
    v = np.asarray(v, float); a = np.linalg.norm(v)
    if a < 1e-12: return np.eye(3)
    k = v/a; K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3)+math.sin(a)*K+(1-math.cos(a))*K@K


def rpy_matrix(r, p, y):  # SDF pose: extrinsic x-y-z
    return rotvec([0, 0, y])@rotvec([0, p, 0])@rotvec([r, 0, 0])


def matrix_rpy(M):
    p = -math.asin(max(-1., min(1., M[2, 0])))
    return math.atan2(M[2, 1], M[2, 2]), p, math.atan2(M[1, 0], M[0, 0])


def sample_delta(rng, profile='v1'):
    """TCP-frame grasp change (rotation vector in rad, translation in m).

    v1 (grasp-train-001): +-6 mm. v2: every race seen so far pushed the plug
    7-15 mm deeper along TCP z and 4-7 mm in -y, at +7-19 deg about x, so v2
    samples y in [-10, +4] mm, z in [-18, +4] mm and x rotation in [-6, 22] deg."""
    if rng.random() < (.2 if profile == 'v1' else .15):
        return [0., 0., 0.], [0., 0., 0.]
    d = math.radians
    rz = rng.uniform(-40, 40) if rng.random() < .1 else rng.uniform(-12, 12)
    if profile == 'v1':
        rot = [d(rng.uniform(-6, 18)), d(rng.uniform(-8, 8)), d(rz)]
        return rot, [rng.uniform(-.006, .006) for _ in range(3)]
    rot = [d(rng.uniform(-6, 22)), d(rng.uniform(-8, 8)), d(rz)]
    return rot, [rng.uniform(-.005, .005), rng.uniform(-.010, .004), rng.uniform(-.018, .004)]


def perturb(cable, rot, trans):
    DR = rotvec(rot)
    o = np.array([cable['pose']['gripper_offset'][k] for k in 'xyz'])
    Rc = rpy_matrix(*(cable['pose'][k] for k in ('roll', 'pitch', 'yaw')))
    o2 = RH@DR@RH.T@o+RH@np.asarray(trans)
    rpy = matrix_rpy(RH@DR@RH.T@Rc)
    cable['pose']['gripper_offset'] = {k: round(float(v), 6) for k, v in zip('xyz', o2)}
    cable['pose'].update({k: round(float(v), 6) for k, v in zip(('roll', 'pitch', 'yaw'), rpy)})


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--trials', type=int, required=True, help='per plug type')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--profile', choices=('v1', 'v2'), default='v1')
    a = ap.parse_args()
    assert not 20261901 <= a.seed <= 20261930, 'held-out seeds are spent'
    rng = random.Random(a.seed)
    trials, deltas = {}, {}
    rep = 0
    while min(sum(t.endswith(m) for t in trials) for m in ('sfp', 'sc')) < a.trials:
        board = 'wide' if rep % 2 else 'template'
        scenes, _ = generate(a.seed*1000+rep, 1, 10., 'rail', board)
        for name, trial in scenes['trials'].items():
            mode = next(iter(trial['tasks'].values()))['plug_type']
            if sum(t.endswith(mode) for t in trials) >= a.trials:
                continue
            rot, trans = sample_delta(rng, a.profile)
            for cable in trial['scene']['cables'].values():
                if cable.get('attach_cable_to_gripper'):
                    perturb(cable, rot, trans)
            key = f'grasp_{a.seed}_{len(trials):04d}_{mode}'
            trials[key] = trial
            deltas[key] = {'rotvec_rad': rot, 'translation_m': trans, 'board_pose': board, 'profile': a.profile}
        rep += 1
    out = {k: v for k, v in scenes.items() if k != 'trials'}
    out['trials'] = dict(sorted(trials.items()))
    a.output.write_text(yaml.safe_dump(out, sort_keys=False))
    a.output.with_suffix('.deltas.json').write_text(json.dumps(deltas, indent=1))
    print(len(trials), 'trials ->', a.output)
