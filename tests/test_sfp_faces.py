"""SFP face labels and training semantics use observable port-face geometry."""
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
from label_sfp_faces import LANDMARK_NAMES, label_view
from aic_model.sfp_card_template import PORTS, card_landmarks_board, card_landmarks_card
from aic_model.sfp_geometry import SFP_FACE_Z_PORT_M, SFP_PORT_POSES_CARD
from aic_model.policy import Policy


def synthetic_gt(port, R_cam_port, t_cam_port):
    K = np.array([[1100., 0, 576], [0, 1100., 512], [0, 0, 1]])
    face = R_cam_port@np.array([0., 0., SFP_FACE_Z_PORT_M])+t_cam_port
    return {'R_cam_from_port': R_cam_port.ravel().tolist(), 'xyz_camera_port_origin': t_cam_port.tolist(),
            'xyz_camera': face.tolist(), 'intrinsics_k': K.ravel().tolist(), 'image_size': [1152, 1024]}


@pytest.mark.parametrize('port', PORTS)
def test_labels_place_both_port_faces_from_the_target_port_pose(port):
    R = cv2.Rodrigues(np.array([2.6, .2, -.1]))[0]
    gt = synthetic_gt(port, R, np.array([.01, -.02, .3]))
    pixels, visible = label_view(gt, port, 1152, 1024)
    assert pixels.shape == (10, 2) and len(visible) == 10 and len(LANDMARK_NAMES) == 10
    K = np.asarray(gt['intrinsics_k']).reshape(3, 3)
    center = K@np.asarray(gt['xyz_camera'])
    np.testing.assert_allclose(pixels[5*PORTS.index(port)+4], center[:2]/center[2], atol=1e-6)
    # The other port lies 23.2 mm away along the card's X axis, whichever is the target.
    offset = np.linalg.norm(np.subtract(SFP_PORT_POSES_CARD['sfp_port_0']['translation'],
                                        SFP_PORT_POSES_CARD['sfp_port_1']['translation']))
    assert offset == pytest.approx(.0232)
    faces = card_landmarks_card()
    assert np.linalg.norm(faces[0, 4]-faces[1, 4]) == pytest.approx(offset)
    gt['xyz_camera'] = (np.asarray(gt['xyz_camera'])+.001).tolist()
    with pytest.raises(ValueError, match='disagrees'):
        label_view(gt, port, 1152, 1024)


def test_card_template_translation_moves_ports_along_board_x_and_yaw_rotates_about_normal():
    base = card_landmarks_board('nic_card_mount_2', 0., 0.)
    shifted = card_landmarks_board('nic_card_mount_2', .01, 0.)
    np.testing.assert_allclose(shifted-base, np.broadcast_to([.01, 0., 0.], base.shape), atol=1e-12)
    yawed = card_landmarks_board('nic_card_mount_2', 0., np.deg2rad(10.))
    np.testing.assert_allclose(yawed[..., 2], base[..., 2], atol=1e-12)
    with pytest.raises(ValueError):
        card_landmarks_board('sc_port_0', 0., 0.)


def test_trainer_loads_generic_sfp_rows_and_rejects_mismatched_semantics(tmp_path):
    from train_sc_port_detector import SFP_FACES_SPEC, SC_SPEC, load_manifest
    row = {'npz_path': str(tmp_path/'a.npz'), 'image_key': 'center_image', 'sample_id': 's', 'camera': 'center',
           'task': {'target_module_name': 'nic_card_mount_0', 'port_name': 'sfp_port_0'},
           'landmarks': {'names': list(SFP_FACES_SPEC.names), 'points_norm': [[.5, .5]]*10, 'visible': [True]*10}}
    labels = tmp_path/'labels.jsonl'
    labels.write_text(json.dumps(row)+'\n')
    samples = load_manifest(labels, SFP_FACES_SPEC)
    assert len(samples[0].points_norm) == 10 and all(samples[0].visible)
    with pytest.raises(ValueError, match='differ'):
        load_manifest(labels, SC_SPEC)
    assert SFP_FACES_SPEC.names == LANDMARK_NAMES


def sfp_scene(translation=.010, yaw_deg=5., shift_camera=None):
    """Three downward cameras over NIC mount 2 and a stub network that picks a card placement."""
    from types import SimpleNamespace as NS
    from replay_policy_pose import ReplayPolicy
    from aic_model.board_registration import BoardPose
    from aic_model.sfp_face_decoder import DECODER
    K = np.array([[1100., 0, 576], [0, 1100., 512], [0, 0, 1]])
    R = np.diag([1., -1., -1.])        # camera looks down the board normal
    images, geometry, exposures, cameras = {}, {}, {}, ('center', 'left', 'right')
    for i, (name, dx) in enumerate(zip(cameras, (0., -.06, .06))):
        image = np.zeros((1024, 1152, 3), np.uint8); image[0, 0, 0] = i
        images[name+'_image'] = image
        geometry[name] = {'K': K.tolist(), 'R_base_from_camera': R.tolist(),
                          't_base_from_camera': [-.075+dx, -.1025, .42]}
        exposures[name] = {'frame_id': name+'/optical', 'sec': 1, 'nanosec': 0}

    def infer(image, support_mask=None, face_templates=None, rail_hull=None):
        index = int(image[0, 0, 0]) if image[0, 0, 0] < 3 else None
        t = translation+(.0232 if shift_camera is not None and cameras[index or 0] == shift_camera else 0.)
        choice = int(np.argmin(np.abs(face_templates['translation']-t)
                               + np.abs(face_templates['yaw']-np.deg2rad(yaw_deg))))
        scale = np.array([image.shape[1], image.shape[0]])/96.
        return {'points_px': face_templates['points'][choice]*scale, 'confidence': np.ones(10),
                'image_size': (image.shape[1], image.shape[0]),
                'rail_translation': float(face_templates['translation'][choice]),
                'rail_yaw': float(face_templates['yaw'][choice])}
    detector = NS(decoder=DECODER, preprocessing='rail_crop_v1', heatmap_size=96, infer=infer)
    policy = ReplayPolicy(detector, 'sfp')
    policy._board_pose = BoardPose(np.eye(3), np.zeros(3), 0.)
    metadata = {'camera_geometry': geometry, 'camera_exposures': exposures, 'capture_time_sim': 1.,
                'controller': {'tcp_pose': {'position': {'x': -.075, 'y': -.1025, 'z': .40}}},
                'task': {'target_module_name': 'nic_card_mount_2', 'port_name': 'sfp_port_1'}, 'phase': 'find_target'}
    return policy, metadata, images


def test_sfp_face_estimator_recovers_the_requested_port_face():
    from types import SimpleNamespace as NS
    from replay_policy_pose import evaluate_observation
    from aic_model import board_registration
    policy, metadata, images = sfp_scene()
    # The crop keeps the image marker the stub uses to identify the camera.
    original = board_registration.module_crop
    board_registration.module_crop = lambda image, *a: (image, np.zeros(2))
    try:
        row = evaluate_observation(policy, NS(port_pos_smoothed=None, phase='find_target'), metadata, images)
    finally:
        board_registration.module_crop = original
    truth = card_landmarks_board('nic_card_mount_2', .010, np.deg2rad(5.))[1, 4]
    assert row['status'] == 'accepted' and row['source_cameras'] == 'center+left+right'
    np.testing.assert_allclose(row['raw_face_position'], truth, atol=2e-4)


def test_sfp_face_estimator_drops_a_camera_that_decodes_the_card_one_port_pitch_away():
    from types import SimpleNamespace as NS
    from replay_policy_pose import evaluate_observation
    from aic_model import board_registration
    policy, metadata, images = sfp_scene(shift_camera='left')
    original = board_registration.module_crop
    board_registration.module_crop = lambda image, *a: (image, np.zeros(2))
    try:
        row = evaluate_observation(policy, NS(port_pos_smoothed=None, phase='find_target'), metadata, images)
    finally:
        board_registration.module_crop = original
    assert row['status'] == 'accepted' and row['source_cameras'] == 'center+right'
    assert row['estimator']['cameras']['left'] == 'inconsistent_rail_placement'


def test_backbone_initialization_copies_features_but_never_heads():
    import torch
    from train_sc_port_detector import load_backbone
    from aic_model.landmark_network import LandmarkHeatmapNet
    source = LandmarkHeatmapNet(num_landmarks=5, input_channels=3)
    target = LandmarkHeatmapNet(num_landmarks=10, input_channels=4)
    head_before = target.heatmap_head.weight.detach().clone()
    load_backbone(target, {'state_dict': source.state_dict()}, 4)
    assert torch.equal(target.down3[1].net[0].weight, source.down3[1].net[0].weight)
    assert torch.equal(target.stem.net[0].weight[:, :3], source.stem.net[0].weight)
    assert torch.count_nonzero(target.stem.net[0].weight[:, 3]) == 0
    assert torch.equal(target.heatmap_head.weight, head_before)
    with pytest.raises(ValueError, match='mismatch'):
        load_backbone(target, {'state_dict': {**source.state_dict(), 'extra.weight': torch.zeros(1)}}, 4)


def test_gt_board_reconstruction_inverts_the_card_chain():
    from prepare_rail_dataset import board_from_ground_truth
    from aic_model.sfp_card_template import port_pose_board
    board_R = cv2.Rodrigues(np.array([.02, -.01, 1.3]))[0]; board_t = np.array([.3, -.1, .05])
    R_bp, t_bp = port_pose_board('nic_card_mount_3', .007, np.deg2rad(-6.), 'sfp_port_0')
    R_cam = cv2.Rodrigues(np.array([2.5, .1, .2]))[0]; t_cam = np.array([.25, -.05, .45])
    R_base_port, t_base_port = board_R@R_bp, board_R@t_bp+board_t
    gt = {'R_cam_from_port': (R_cam.T@R_base_port).ravel().tolist(),
          'xyz_camera_port_origin': (R_cam.T@(t_base_port-t_cam)).tolist()}
    metadata = {'task': {'target_module_name': 'nic_card_mount_3', 'port_name': 'sfp_port_0'},
                'ground_truth': {'center': gt},
                'camera_geometry': {'center': {'R_base_from_camera': R_cam.tolist(), 't_base_from_camera': t_cam.tolist()}}}
    config = {'trials': {'t': {'scene': {'task_board': {'nic_rail_3': {'entity_pose': {'translation': .007, 'yaw': np.deg2rad(-6.)}}}}}}}
    board = board_from_ground_truth(metadata, config)
    np.testing.assert_allclose(board.rotation, board_R, atol=1e-9)
    np.testing.assert_allclose(board.translation, board_t, atol=1e-9)


def test_training_accepts_labeled_gt_board_crops_but_no_unknown_source():
    from aic_model.dataset import parse_runtime_crop
    row = {'runtime_crop': {'preprocessing': 'rail_crop_v1', 'source': 'privileged_gt_board', 'box_xyxy': [0, 0, 40, 40],
                            'capture_sha256': 'a'*64}}
    assert parse_runtime_crop(row) == (0, 0, 40, 40)
    row['runtime_crop']['source'] = 'label_box'
    with pytest.raises(ValueError, match='provenance'):
        parse_runtime_crop(row)


def test_consistent_points_picks_the_tightest_largest_agreeing_set():
    from aic_model.sc_face_decoder import consistent_points
    a, b, c = np.zeros(3), np.array([.0015, 0., 0.]), np.array([0., .0012, 0.])
    assert consistent_points({'a': a, 'b': b, 'c': c}, .002) == (['a', 'b', 'c'], None)
    assert consistent_points({'a': a, 'b': b, 'c': np.array([.01, 0., 0.])}, .002) == (['a', 'b'], None)
    assert consistent_points({'a': a, 'b': np.array([.01, 0., 0.])}, .002) == (None, 'inconsistent_rail_placement')


@pytest.mark.parametrize('dt, dyaw, kept', [(-.001, 5., True), (.004, 0., False)])
def test_sfp_gate_compares_the_face_each_view_implies(dt, dyaw, kept):
    from types import SimpleNamespace as NS
    from replay_policy_pose import evaluate_observation
    from aic_model import board_registration
    policy, metadata, images = sfp_scene()
    inner = policy._sfp_detector.infer

    def infer(image, **kwargs):
        prediction = inner(image, **kwargs)
        if image[0, 0, 0] == 1:     # the left camera decodes a perturbed placement
            templates = kwargs['face_templates']
            choice = int(np.argmin(np.abs(templates['translation']-(.010+dt))
                                   + np.abs(templates['yaw']-np.deg2rad(5.+dyaw))))
            scale = np.array([image.shape[1], image.shape[0]])/96.
            prediction.update(points_px=templates['points'][choice]*scale,
                              rail_translation=float(templates['translation'][choice]),
                              rail_yaw=float(templates['yaw'][choice]))
        return prediction
    policy._sfp_detector.infer = infer
    original = board_registration.module_crop
    board_registration.module_crop = lambda image, *a: (image, np.zeros(2))
    try:
        row = evaluate_observation(policy, NS(port_pos_smoothed=None, phase='find_target'), metadata, images)
    finally:
        board_registration.module_crop = original
    assert row['status'] == 'accepted'
    assert ('left' in row['source_cameras']) is kept


def test_face_rail_check_rejects_every_neighbour_placement_and_accepts_every_legal_one():
    from aic_model.sfp_card_template import face_on_requested_card
    # The physical rail travel, including the sample config's +36 mm card.
    for t in (*np.linspace(-.048, .036, 6), .0234):
        for yaw in np.deg2rad(np.linspace(-10., 10., 5)):
            for mount in range(5):
                for index, port in enumerate(PORTS):
                    face = card_landmarks_board(f'nic_card_mount_{mount}', t, yaw)[index, 4]
                    assert face_on_requested_card(face, f'nic_card_mount_{mount}', port)
                    for other in {max(mount-1, 0), min(mount+1, 4)}-{mount}:
                        assert not face_on_requested_card(face, f'nic_card_mount_{other}', port)


@pytest.mark.parametrize('enabled', [False, True])
def test_single_view_board_pose_is_an_explicit_option(enabled):
    from types import SimpleNamespace as NS
    from replay_policy_pose import evaluate_observation
    from aic_model import board_registration
    policy, metadata, images = sfp_scene()
    policy.SFP_SINGLE_VIEW_BOARD_POSE = enabled
    inner = policy._sfp_detector.infer
    policy._sfp_detector.infer = lambda image, **kw: inner(image, **kw) if image[0, 0, 0] == 1 else None
    original = board_registration.module_crop
    board_registration.module_crop = lambda image, *a: (image, np.zeros(2))
    try:
        row = evaluate_observation(policy, NS(port_pos_smoothed=None, phase='find_target'), metadata, images)
    finally:
        board_registration.module_crop = original
    if not enabled:
        assert row['status'] == 'insufficient_cameras'
        return
    assert Policy.SFP_SINGLE_VIEW_BOARD_POSE and Policy.SFP_SINGLE_VIEW_MIN_MARGIN == .15
    truth = card_landmarks_board('nic_card_mount_2', .010, np.deg2rad(5.))[1, 4]
    assert row['status'] == 'accepted' and row['source_cameras'] == 'left'
    assert row['estimator']['pose_source'] == 'single_view_board'
    np.testing.assert_allclose(row['raw_face_position'], truth, atol=2e-4)


def test_single_view_rejects_near_ambiguous_decodes():
    from types import SimpleNamespace as NS
    from replay_policy_pose import evaluate_observation
    from aic_model import board_registration
    policy, metadata, images = sfp_scene()
    policy.SFP_SINGLE_VIEW_BOARD_POSE = True
    detector = policy._sfp_detector
    inner = detector.infer

    def infer(image, **kw):
        if image[0, 0, 0] != 1:
            return None
        detector.last_decoder_diagnostics = {'margin': .104}
        return inner(image, **kw)
    detector.infer = infer
    original = board_registration.module_crop
    board_registration.module_crop = lambda image, *a: (image, np.zeros(2))
    try:
        row = evaluate_observation(policy, NS(port_pos_smoothed=None, phase='find_target'), metadata, images)
    finally:
        board_registration.module_crop = original
    assert row['status'] == 'single_view_margin' and not row['visible']


def test_card_refinement_recovers_an_off_grid_yaw_but_keeps_the_coarse_placement():
    from aic_model.board_registration import BoardPose
    from aic_model.sfp_face_decoder import CORNER_INDICES, YAW_STEP_RAD, card_templates, decode_card
    K = np.array([[1100., 0, 576], [0, 1100., 512], [0, 0, 1]])
    geometry = (K, np.diag([1., -1., -1.]), np.array([-.075, -.1025, .42]))
    board = BoardPose(np.eye(3), np.zeros(3), 0.)
    truth = (.0102, np.deg2rad(3.7))            # between the 0.5 mm / 2.5 degree grid points
    center = K@(geometry[1].T@(card_landmarks_board('nic_card_mount_2', *truth)[1, 4]-geometry[2]))
    offset = center[:2]/center[2]-[144., 128.]
    templates = card_templates(board, geometry, 'nic_card_mount_2', offset, (256, 288, 3), 96)
    uv, _ = templates['project'](np.array([truth[0]]), np.array([truth[1]]))
    rows, cols = np.mgrid[:96, :96]
    heatmaps = np.zeros((10, 96, 96))
    for index in CORNER_INDICES:
        u, v = uv[0, index]
        heatmaps[index] = np.exp(-((cols-u)**2+(rows-v)**2)/(2*1.5**2))
    decoded = decode_card(heatmaps, np.ones((96, 96), bool), templates, diagnostics := {})
    coarse = diagnostics['best']
    assert abs(coarse['yaw']-truth[1]) > np.deg2rad(.5)        # the grid alone cannot represent it
    assert abs(decoded['refined_yaw']-truth[1]) < np.deg2rad(.3)
    assert abs(decoded['refined_yaw']-coarse['yaw']) <= YAW_STEP_RAD
    # Position (points, translation) stays on the grid decision.
    assert decoded['yaw'] == coarse['yaw'] and decoded['translation'] == coarse['translation']


def test_templates_cover_the_physical_rail_beyond_the_documented_limits():
    from aic_model.sfp_card_template import SPEC_TRANSLATION_LIMITS_M, TRANSLATION_LIMITS_M
    from aic_model.sfp_face_decoder import _board_templates
    x, _, _ = _board_templates('nic_card_mount_0')
    assert x.min() <= -.048+1e-9 and x.max() >= .036-1e-9
    assert TRANSLATION_LIMITS_M[0] < SPEC_TRANSLATION_LIMITS_M[0] and TRANSLATION_LIMITS_M[1] > SPEC_TRANSLATION_LIMITS_M[1]


def test_gt_board_reconstruction_also_inverts_the_sc_port_chain():
    from prepare_rail_dataset import board_from_ground_truth
    from screen_scenes import target_port_board
    board_R = cv2.Rodrigues(np.array([.01, .02, -2.2]))[0]; board_t = np.array([.2, .05, .02])
    config = {'trials': {'t': {'scene': {'task_board': {'sc_rail_1': {'entity_pose': {'translation': -.055, 'yaw': 0.}}}}}}}
    task = {'target_module_name': 'sc_port_1', 'port_name': 'sc_port_base'}
    R_bp, t_bp, _ = target_port_board({**config['trials']['t'], 'tasks': {'task': task}})
    R_cam = cv2.Rodrigues(np.array([2.4, -.1, .3]))[0]; t_cam = np.array([.1, .1, .5])
    R_base_port, t_base_port = board_R@R_bp, board_R@t_bp+board_t
    metadata = {'task': task,
                'ground_truth': {'left': {'R_cam_from_port': (R_cam.T@R_base_port).ravel().tolist(),
                                          'xyz_camera_port_origin': (R_cam.T@(t_base_port-t_cam)).tolist()}},
                'camera_geometry': {'left': {'R_base_from_camera': R_cam.tolist(), 't_base_from_camera': t_cam.tolist()}}}
    board = board_from_ground_truth(metadata, config)
    np.testing.assert_allclose(board.rotation, board_R, atol=1e-9)
    np.testing.assert_allclose(board.translation, board_t, atol=1e-9)
