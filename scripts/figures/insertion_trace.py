#!/usr/bin/env python3
"""Export one trial's insertion trace from an evaluator bag as CSV (post-hoc, ground truth).

Runs inside the evaluation image (rosbag2, AIC messages), like posthoc_bag.py:

  docker run --rm --entrypoint bash -v $PWD/benchmark_runs/<run>/results:/r:ro \\
      -v $PWD/scripts:/s:ro -v $PWD/out:/o <eval-image> -lc 'source /ws_aic/install/setup.bash; \\
      python3 /s/figures/insertion_trace.py /r/bag_trial_1_* /o/trace.csv \\
      --module nic_card_mount_2 --port sfp_port_0 --mode sfp'

Columns: wall_s (bag receive time), sim_s (simulation time, from /joint_states
stamps), depth_mm (plug tip along the port axis from the entrance face, positive
inside), lateral_mm (plug tip off the port axis) and force_n (wrist force
magnitude, sampled at the nearest wrench message).
"""
import argparse
import bisect
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from posthoc_bag import FACE_Z_M, reader, transform_matrix  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('bag')
    ap.add_argument('output')
    ap.add_argument('--module', required=True)
    ap.add_argument('--port', default='sc_port_base')
    ap.add_argument('--mode', choices=tuple(FACE_Z_M), default='sfp')
    ap.add_argument('--cable', default='cable_0')
    ap.add_argument('--every', type=float, default=0.05)
    a = ap.parse_args()
    from rclpy.serialization import deserialize_message
    from geometry_msgs.msg import WrenchStamped
    from sensor_msgs.msg import JointState
    from tf2_msgs.msg import TFMessage

    clock, forces = [], []
    for topic, data, stamp, _ in reader(a.bag, ['/joint_states', '/fts_broadcaster/wrench']):
        if topic == '/joint_states':
            s = deserialize_message(data, JointState).header.stamp
            clock.append((stamp, s.sec+s.nanosec*1e-9))
        else:
            f = deserialize_message(data, WrenchStamped).wrench.force
            forces.append((stamp, float(np.linalg.norm([f.x, f.y, f.z]))))
    clock.sort(); forces.sort()
    wall_clock, sim_clock = np.array(clock).T
    force_t = [t for t, _ in forces]

    port_link = f'{a.port}_link' if a.mode == 'sfp' else 'sc_port_base_link'
    chain = ('task_board', f'task_board/{a.module}', f'task_board/{a.module}/{port_link}')
    plug = f'{a.cable}/{a.mode}_tip_link'
    frames, parents, face, last, rows = {}, {}, None, -np.inf, []
    for _, data, stamp, _ in reader(a.bag, ['/scoring/tf']):
        for item in deserialize_message(data, TFMessage).transforms:
            frames[item.child_frame_id] = transform_matrix(item.transform)
            parents[item.child_frame_id] = item.header.frame_id
        if face is None and all(name in frames for name in chain):
            port = frames[chain[0]]@frames[chain[1]]@frames[chain[2]]
            face = port.copy()
            face[:3, 3] = port[:3, 3]+port[:3, :3]@[0., 0., FACE_Z_M[a.mode]]
        if face is None or plug not in frames or stamp-last < a.every:
            continue
        last = stamp
        T, name = np.eye(4), plug
        while name in frames:
            T = frames[name]@T
            if parents.get(name) in (None, 'aic_world', 'world'):
                break
            name = parents[name]
        rel = np.linalg.inv(face)@T
        i = min(bisect.bisect_left(force_t, stamp), len(forces)-1)
        rows.append((stamp, float(np.interp(stamp, wall_clock, sim_clock)), rel[2, 3]*1e3,
                     float(np.hypot(rel[0, 3], rel[1, 3]))*1e3, forces[i][1]))
    with open(a.output, 'w') as out:
        out.write('wall_s,sim_s,depth_mm,lateral_mm,force_n\n')
        for row in rows:
            out.write(','.join(f'{v:.4f}' for v in row)+'\n')
    print(a.output, len(rows), 'rows')


if __name__ == '__main__':
    main()
