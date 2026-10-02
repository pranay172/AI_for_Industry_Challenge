#!/usr/bin/env python3
"""Seeded regression scenes within the current engine's five NIC/two SC slots.

This is a bounded stress suite, not the challenge's exhaustive distribution.
Changing the populated target rail does not verify the qualification requirement
that its port is initially visible. These scenes are not qualification claims.
Never read its output from a policy; it is an evaluator input only.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import random
import yaml

ROOT = Path(__file__).resolve().parents[1]
# NIC translation: the documented limits, or the mount's physical travel
# (task_board.urdf.xacro), which the engine's own sample_config.yaml uses.
NIC_TRANSLATION_M = {'spec': (-.0215, .0234), 'rail': (-.048, .036)}
# Board pose. 'template' perturbs each qualification.yaml trial by +-5 mm and
# +-2 deg. 'wide' samples the span of every organizer example (eval_config.yaml
# and sample_config.yaml: x 0.15-0.20, y -0.21-0.05, yaw 3.0 through pi to
# -1.8) plus a margin; the target's initial visibility is left to screen_scenes.
WIDE_BOARD_X_M, WIDE_BOARD_Y_M = (.14, .21), (-.22, .06)
WIDE_BOARD_YAW_RAD = (2.9, 2*math.pi-1.7)


def wrap_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def generate(seed, repetitions=1, nic_yaw_deg=10., nic_translation='spec', board_pose='template'):
    if not math.isfinite(nic_yaw_deg) or not 0 <= nic_yaw_deg <= 10:
        raise ValueError("nic_yaw_deg must be finite and between 0 and 10")
    if nic_translation not in NIC_TRANSLATION_M or board_pose not in ('template', 'wide'):
        raise ValueError("unknown nic_translation or board_pose")
    source = ROOT/'benchmarks/qualification.yaml'
    base = yaml.safe_load(source.read_text())
    templates = list(base['trials'].values())
    rng = random.Random(seed)
    output = deepcopy(base); output['trials'] = {}
    for rep in range(repetitions):
        for index, template in enumerate(templates):
            trial = deepcopy(template)
            task = next(iter(trial['tasks'].values()))
            board = trial['scene']['task_board']
            mode = task['plug_type']
            target_slot = rng.randrange(5 if mode == 'sfp' else 2)
            for family, count, low, high in [('nic',5,*NIC_TRANSLATION_M[nic_translation]),('sc',2,-.06,.055)]:
                for slot in range(count):
                    target = (mode == 'sfp' and family == 'nic' or mode == 'sc' and family == 'sc') and slot == target_slot
                    present = target or rng.random() < .65
                    board[f'{family}_rail_{slot}'] = {
                        'entity_present': present,
                        'entity_name': f'{"nic_card_mount" if family == "nic" else "sc_port"}_{slot}',
                        'entity_pose': {'translation': rng.uniform(low,high), 'roll':0., 'pitch':0.,
                                        'yaw':rng.uniform(-math.radians(nic_yaw_deg),math.radians(nic_yaw_deg)) if family=='nic' else 0.}}
            task['target_module_name'] = f'{"nic_card_mount" if mode == "sfp" else "sc_port"}_{target_slot}'
            task['port_name'] = f'sfp_port_{index % 2}' if mode == 'sfp' else 'sc_port_base'
            if board_pose == 'wide':
                board['pose']['x'] = rng.uniform(*WIDE_BOARD_X_M)
                board['pose']['y'] = rng.uniform(*WIDE_BOARD_Y_M)
                board['pose']['yaw'] = wrap_angle(rng.uniform(*WIDE_BOARD_YAW_RAD))
            else:
                board['pose']['x'] += rng.uniform(-.005,.005)
                board['pose']['y'] += rng.uniform(-.005,.005)
                board['pose']['yaw'] += rng.uniform(-math.radians(2),math.radians(2))
            # Small grasp perturbations expose reliance on a single calibrated offset.
            for cable in trial['scene']['cables'].values():
                for axis in ('x','y','z'):
                    cable['pose']['gripper_offset'][axis] += rng.uniform(-.001,.001)
            output['trials'][f'seed_{seed}_repeat_{rep}_{mode}_{index}'] = trial
    return output, {'seed':seed,'repetitions':repetitions,
                    'qualification_start_conditions_verified':False,
                    'nic_yaw_deg':nic_yaw_deg,
                    'nic_yaw_range_is_official':False,
                    'nic_translation':nic_translation,'board_pose':board_pose,
                    'template_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
                    'scope':'five NIC slots, two SC slots; bounded board and grasp perturbations'}


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--repetitions',type=int,default=1)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--nic-yaw-deg',type=float,default=10.,
                        help='Maximum absolute NIC yaw; 0 keeps mounts aligned (default: 10, stress-test choice)')
    parser.add_argument('--nic-translation',choices=sorted(NIC_TRANSLATION_M),default='spec',
                        help='NIC translation range: documented limits or the physical rail travel')
    parser.add_argument('--board-pose',choices=('template','wide'),default='template',
                        help='Perturb the qualification board poses, or sample the span of all organizer examples')
    args=parser.parse_args()
    if args.repetitions < 1: parser.error('repetitions must be positive')
    if args.output.exists(): parser.error('output already exists')
    scene, provenance=generate(args.seed,args.repetitions,args.nic_yaw_deg,args.nic_translation,args.board_pose)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(yaml.safe_dump(scene,sort_keys=False))
    args.output.with_suffix('.source.json').write_text(json.dumps(provenance,indent=2)+'\n')
