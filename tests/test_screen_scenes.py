"""Evaluation screening uses the same public geometry as the decoders."""
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
from screen_scenes import face_points_port, target_port_board, visible_cameras
from aic_model.sc_face_decoder import _board_templates
from aic_model.sfp_card_template import card_landmarks_board


def trial(module, translation=.01, yaw=0., port='sc_port_base'):
    board = {'pose': {'x': 0., 'y': 0., 'z': 0., 'roll': 0., 'pitch': 0., 'yaw': 0.}}
    key = f"{'nic' if module.startswith('nic') else 'sc'}_rail_{module[-1]}"
    board[key] = {'entity_pose': {'translation': translation, 'yaw': yaw}}
    return {'scene': {'task_board': board}, 'tasks': {'t': {'target_module_name': module, 'port_name': port}}}


def face_board(t):
    R, origin, z = target_port_board(t)
    module = next(iter(t['tasks'].values()))['target_module_name']
    return (R@face_points_port(module, z).T).T+origin


def test_sc_face_center_matches_the_decoder_template_chain():
    x, yaw, points = _board_templates('sc_port_1', np.deg2rad(2.5))
    index = int(np.argmin(np.abs(x-(-.075+.02))+np.abs(yaw)))
    np.testing.assert_allclose(face_board(trial('sc_port_1', .02))[4], points[index, 4], atol=1e-9)


def test_sfp_face_center_matches_the_card_template():
    t = trial('nic_card_mount_3', -.004, np.deg2rad(7.), 'sfp_port_1')
    np.testing.assert_allclose(face_board(t)[4], card_landmarks_board('nic_card_mount_3', -.004, np.deg2rad(7.))[1, 4],
                               atol=1e-9)


def test_visibility_requires_the_whole_face_in_frame_and_facing_the_camera():
    t = trial('nic_card_mount_2', 0., 0., 'sfp_port_0')
    face = face_board(t)[4]
    R_port, _, _ = target_port_board(t)
    outward = -R_port[:, 2]
    K = [[1000., 0, 500], [0, 1000., 500], [0, 0, 1]]

    def camera(position):
        z = (face-position)/np.linalg.norm(face-position)
        x = np.cross([0., 0., 1.] if abs(z[2]) < .9 else [1., 0., 0.], z); x /= np.linalg.norm(x)
        return {'K': K, 'image_size': [1000, 1000], 'R_base_from_camera': np.column_stack((x, np.cross(z, x), z)).tolist(),
                't_base_from_camera': position.tolist()}
    identity = {'R_base_world': np.eye(3).tolist(), 't_base_world': [0., 0., 0.]}
    front = {**identity, 'cameras': {'front': camera(face+.3*outward)}}
    behind = {**identity, 'cameras': {'behind': camera(face-.3*outward)}}
    assert visible_cameras(t, front) == ['front'] and visible_cameras(t, behind) == []
    # With a narrow field of view, a card shifted 20 mm along its rail leaves the image.
    moved = trial('nic_card_mount_2', .02, 0., 'sfp_port_0')
    far = {**identity, 'cameras': {'front': {**camera(face+.3*outward), 'K': [[20000., 0, 500], [0, 20000., 500], [0, 0, 1]]}}}
    assert visible_cameras(moved, far) == []


def test_calibration_uses_the_earliest_start_pose_capture_not_name_order(tmp_path):
    import json
    from screen_scenes import start_capture
    captures = tmp_path/'captures'; captures.mkdir()

    def write(name, t, z, gt=True):
        (captures/f'{name}.json').write_text(json.dumps({
            'capture_time_sim': t, 'controller': {'tcp_pose': {'position': {'x': 0., 'y': 0., 'z': z}}},
            'ground_truth': {'center': {'R_cam_from_port': [1.] * 9} if gt else {'visible': False}}}))
    # Unpadded millisecond stamps: '10388' sorts before '2806' by name.
    write('ep_10388_000021', 10.4, .42)
    write('ep_2806_000000', 2.8, .32, gt=False)
    write('ep_3300_000001', 3.3, .32)
    assert start_capture(tmp_path)['capture_time_sim'] == 3.3
    # No ground truth before the arm moves: the scene cannot calibrate the start view.
    (captures/'ep_3300_000001.json').unlink()
    assert start_capture(tmp_path) is None
