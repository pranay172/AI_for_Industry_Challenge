#!/usr/bin/env python3
"""Post-hoc trial analysis from evaluator bags (ground truth; never used by the policy).

Runs inside the evaluation image, which has rosbag2 and the AIC messages:

  docker run --rm --entrypoint bash -v $PWD/benchmark_runs/<run>/results:/r:ro \\
      -v $PWD/scripts:/s:ro <eval-image> -lc 'source /opt/ros/kilted/setup.bash; \\
      source /ws_aic/install/setup.bash; python3 /s/posthoc_bag.py track /r/bag_trial_2_* \\
      --module nic_card_mount_4 --port sfp_port_1 --mode sfp'

`track` prints the GT plug tip in the GT target entrance-face frame (x, y lateral;
z along insertion, positive inside), the tilt between plug and port axes, and the
plug yaw about the port axis. `wrench` and `tcp` print force and controller TCP.
"""
import argparse

import numpy as np

# Entrance-face offsets along the port link z axis (ground_truth.ENTRANCE_Z_M).
FACE_Z_M = {'sfp': -0.0458, 'sc': -0.01564}


def reader(bag, topics):
    import rosbag2_py
    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=bag, storage_id='mcap'), rosbag2_py.ConverterOptions('', ''))
    types = {t.name: t.type for t in r.get_all_topics_and_types()}
    r.set_filter(rosbag2_py.StorageFilter(topics=topics))
    while r.has_next():
        topic, data, stamp = r.read_next()
        yield topic, data, stamp*1e-9, types[topic]


def transform_matrix(transform):
    q, t = transform.rotation, transform.translation
    x, y, z, w = q.x, q.y, q.z, q.w
    T = np.eye(4)
    T[:3, :3] = [[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                 [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                 [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]]
    T[:3, 3] = [t.x, t.y, t.z]
    return T


def track(args):
    from rclpy.serialization import deserialize_message
    from tf2_msgs.msg import TFMessage
    port_link = f'{args.port}_link' if args.mode == 'sfp' else 'sc_port_base_link'
    chain = ('task_board', f'task_board/{args.module}', f'task_board/{args.module}/{port_link}')
    plug = f'{args.cable}/{args.mode}_tip_link'
    frames, parents, face, last = {}, {}, None, -np.inf
    for _, data, stamp, _ in reader(args.bag, ['/scoring/tf']):
        for item in deserialize_message(data, TFMessage).transforms:
            frames[item.child_frame_id] = transform_matrix(item.transform)
            parents[item.child_frame_id] = item.header.frame_id
        if face is None and all(name in frames for name in chain):
            port = frames[chain[0]]@frames[chain[1]]@frames[chain[2]]
            face = port.copy()
            face[:3, 3] = port[:3, 3]+port[:3, :3]@[0., 0., FACE_Z_M[args.mode]]
        if face is None or plug not in frames or stamp-last < args.every:
            continue
        last = stamp
        T, name = np.eye(4), plug
        while name in frames:
            T = frames[name]@T
            if parents.get(name) in (None, 'aic_world', 'world'):
                break
            name = parents[name]
        rel = np.linalg.inv(face)@T
        print(f'{stamp:.2f} plug_in_face_mm x={rel[0, 3]*1e3:+7.2f} y={rel[1, 3]*1e3:+7.2f} '
              f'z={rel[2, 3]*1e3:+8.2f} tilt_deg={np.degrees(np.arccos(np.clip(rel[2, 2], -1, 1))):.2f} '
              f'yaw_deg={np.degrees(np.arctan2(rel[1, 0], rel[0, 0])):+.2f}')


def wrench(args):
    from rclpy.serialization import deserialize_message
    from geometry_msgs.msg import WrenchStamped
    last = -np.inf
    for _, data, stamp, _ in reader(args.bag, ['/fts_broadcaster/wrench']):
        f = deserialize_message(data, WrenchStamped).wrench.force
        if stamp-last >= args.every:
            last = stamp
            print(f'{stamp:.2f} |f|={np.linalg.norm([f.x, f.y, f.z]):6.1f} ({f.x:+.1f},{f.y:+.1f},{f.z:+.1f})')


def tcp(args):
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    last = -np.inf
    for _, data, stamp, kind in reader(args.bag, ['/aic_controller/controller_state']):
        if stamp-last < args.every:
            continue
        last = stamp
        p = deserialize_message(data, get_message(kind)).tcp_pose.position
        print(f'{stamp:.2f} tcp=({p.x:+.4f},{p.y:+.4f},{p.z:+.4f})')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=('track', 'wrench', 'tcp'))
    parser.add_argument('bag')
    parser.add_argument('--module')
    parser.add_argument('--port', default='sc_port_base')
    parser.add_argument('--mode', choices=tuple(FACE_Z_M), default='sfp')
    parser.add_argument('--cable', default='cable_0')
    parser.add_argument('--every', type=float, default=2.)
    args = parser.parse_args()
    if args.command == 'track' and not args.module:
        parser.error('track needs --module')
    {'track': track, 'wrench': wrench, 'tcp': tcp}[args.command](args)


if __name__ == '__main__':
    main()
