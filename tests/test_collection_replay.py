"""Training/evaluation isolation and replay failure-accounting regressions."""
import json
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
import collect_initial_views as collection
import replay_perception as replay
from aic_model.camera_timing import synchronized_exposures


def test_collection_is_explicitly_privileged_and_keeps_config_out_of_policy(tmp_path):
    from benchmark import compose_config
    training = collection.collection_compose(tmp_path, 'eval', 'model', 'scene', False)
    benchmark = compose_config(tmp_path, 'eval', 'model')
    assert 'ground_truth:=true' in training['services']['eval']['command']
    assert 'ground_truth:=false' in benchmark['services']['eval']['command']
    assert benchmark['services']['model']['environment']['AIC_ENABLE_ACL'] == 'true'
    assert training['services']['model']['environment']['AIC_TRAINING_COLLECTION'] == '1'
    assert [m['target'] for m in training['services']['model']['volumes']] == ['/captures']
    assert training['networks']['default']['internal']
    results = lambda c: [m for m in c['services']['eval']['volumes'] if m['target'] == '/results']
    assert results(training)[0]['type'] == 'bind'
    assert results(collection.collection_compose(tmp_path, 'eval', 'model', 'scene', False, keep_bags=False)) == [
        {'type': 'volume', 'target': '/results'}]
    a = {'scene': {'board': 1}, 'tasks': {'target': 'a'}}
    b = {'scene': {'board': 1}, 'tasks': {'target': 'b'}}
    assert collection.scene_id(a) == collection.scene_id(b)
    assert collection.scene_id(a) != collection.scene_id({'scene': {'board': 2}})


def test_replay_timing_rejects_future_stale_and_unsynchronized_images():
    assert synchronized_exposures({'center':990_000_000,'left':960_000_000,
                                  'right':900_000_000,'old':0,'future':1_100_000_000},
                                  1_000_000_000) == ['center','left']


def test_metric_denominator_includes_missing_predictions():
    rows = [{'status':'board_not_registered','visible_landmarks':5},
            {'status':'predicted','visible_landmarks':5,'errors_px':[1.,2.,3.,4.,20.]}]
    result = replay.summarize(rows)
    assert result['prediction_coverage'] == .5
    assert result['within_10px_including_missed_predictions'] == .4
    assert result['mean_error_px_when_predicted'] == 6.


def write_capture(folder, index, episode):
    npz = folder/f'{index}.npz'
    np.savez(npz, **{c+'_image':np.zeros((48,64,3),np.uint8) for c in ('center','left','right')})
    metadata = {'episode_id':episode,'sample_id':str(index),'capture_time_sim':float(index),
                'files':{'images_npz':npz.name},'task':{'plug_type':'sc','target_module_name':'sc_port_0','port_name':'sc_port_base'},
                'camera_exposures':{c:{'sec':index,'nanosec':0} for c in ('center','left','right')},
                'camera_geometry':{c:{'K':np.eye(3).tolist(),'R_base_from_camera':np.eye(3).tolist(),
                                      't_base_from_camera':[0,0,0]} for c in ('center','left','right')}}
    (folder/f'{index}.json').write_text(json.dumps(metadata))


@pytest.mark.parametrize('preprocessing', ['target_crop_v1', 'rail_crop_v1', 'rail_conditioned_v1'])
def test_replay_caches_only_within_episode_and_never_uses_gt_for_board(tmp_path, monkeypatch, preprocessing):
    for index,episode in [(1,'a'),(2,'a'),(3,'b')]:
        write_capture(tmp_path,index,episode)
    board = object()
    estimates = Mock(side_effect=[board,None])
    monkeypatch.setattr(replay,'register_views',estimates)
    infer = Mock(return_value={'points_px':[[1.,1.]],'confidence':[.9]})
    monkeypatch.setattr(replay,'infer_module',infer)
    result = replay.replay([tmp_path],NS(preprocessing=preprocessing),'sc')
    assert result['summary']['status_counts'] == {'predicted':6,'board_not_registered':3}
    assert estimates.call_count == 2  # Episode a's second frame reuses its RGB estimate.
    assert infer.call_count == 6


def test_collector_requires_explicit_training_mode(monkeypatch):
    from aic_model.CaptureInitialViews import CaptureInitialViews
    monkeypatch.delenv('AIC_TRAINING_COLLECTION',raising=False)
    with pytest.raises(RuntimeError,match='explicit training'):
        CaptureInitialViews(None)


def test_stationary_collector_does_not_issue_motion_and_honors_cancel(monkeypatch):
    import aic_model.CaptureInitialViews as module
    policy = module.CaptureInitialViews.__new__(module.CaptureInitialViews)
    policy.begin_episode = Mock()
    policy._capture_counter = 0
    policy.time_now = Mock(side_effect=[NS(nanoseconds=0),NS(nanoseconds=0),NS(nanoseconds=3_000_000_000)])
    policy._parent_node = NS(check_policy_execution=Mock())
    monkeypatch.setattr(module.time,'sleep',lambda _:None)
    motion = Mock()
    assert not policy.insert_cable(None,lambda:None,motion,Mock())
    motion.assert_not_called()
    policy.time_now = lambda:NS(nanoseconds=0)
    policy._parent_node.check_policy_execution.side_effect = RuntimeError('cancel')
    with pytest.raises(RuntimeError,match='cancel'):
        policy.insert_cable(None,lambda:None,motion,Mock())
    motion.assert_not_called()


def test_capture_audit_detects_incomplete_pairs_and_counts_unique_exposures(tmp_path):
    from audit_captures import audit
    write_capture(tmp_path,1,'a')
    result = audit(tmp_path)
    assert result['counts']['captures'] == 1
    assert result['counts']['fresh_synchronized_images'] == 3
    assert result['unique_exposures'] == 3
    assert result['counts']['images_with_gt_transform'] == 0
    np.savez(tmp_path/'orphan.npz',center_image=np.zeros((2,2,3),np.uint8))
    with pytest.raises(ValueError,match='orphan'):
        audit(tmp_path)


def test_gt_wait_uses_frozen_exposure_and_remains_cancellable(monkeypatch):
    import aic_model.CaptureInitialViews as module
    from rclpy.time import Time
    header = NS(frame_id='camera',stamp=Time(seconds=2).to_msg())
    buffer = NS(can_transform=Mock(side_effect=[False,True]))
    policy = module.CaptureInitialViews.__new__(module.CaptureInitialViews)
    policy._parent_node = NS(_tf_buffer=buffer,check_policy_execution=Mock())
    monkeypatch.setattr(module.time,'monotonic',lambda:0.)
    monkeypatch.setattr(module.time,'sleep',lambda _:None)
    task = NS(target_module_name='sc_port_1',port_name='sc_port_base')
    policy._wait_for_capture_tf(task,NS(image_header_map={'center':header}),30.)
    assert buffer.can_transform.call_count == 2
    args = buffer.can_transform.call_args.args
    assert args[:2] == ('camera','task_board/sc_port_1/sc_port_base_link')
    assert args[2].nanoseconds == 2_000_000_000


def test_collector_queues_exposures_before_waiting_for_slow_gt(monkeypatch):
    import aic_model.CaptureInitialViews as module
    policy = module.CaptureInitialViews.__new__(module.CaptureInitialViews)
    policy.begin_episode = Mock()
    policy._capture_counter = 0
    policy.time_now = Mock(side_effect=[NS(nanoseconds=n) for n in
                                       (0,0,100_000_000,300_000_000,3_000_000_000)])
    policy._parent_node = NS(check_policy_execution=Mock())
    parsed = [object(),object(),object()]
    policy._parse_observation = Mock(side_effect=parsed)
    policy._wait_for_capture_tf = Mock()
    policy._maybe_capture_sample = Mock()
    monkeypatch.setattr(module.time,'sleep',lambda _:None)
    motion = Mock()
    assert not policy.insert_cable(None,lambda:object(),motion,Mock())
    calls = policy._maybe_capture_sample.call_args_list
    assert len(calls) == 2
    assert calls[0].args[1] is parsed[0] and calls[0].args[-1] == 0.
    assert calls[1].args[1] is parsed[2] and calls[1].args[-1] == .3
    assert policy._wait_for_capture_tf.call_count == 2
    motion.assert_not_called()


def test_capture_sample_records_ground_truth_through_the_lazy_import(tmp_path, monkeypatch):
    from aic_model import ground_truth, policy_perception
    from aic_model.policy_capture import CaptureMixin
    monkeypatch.setattr(ground_truth, 'project_port_to_camera', lambda *a: {'port': 1})
    monkeypatch.setattr(ground_truth, 'project_frame_origin_to_camera', lambda *a: {'tip': 1})
    monkeypatch.setattr(policy_perception, 'camera_projection_matrix', lambda *a: None)
    header = NS(frame_id='left_camera', stamp=NS(sec=1, nanosec=2))
    vec = NS(tolist=lambda: [0., 0., 0.])
    xyz = NS(x=0., y=0., z=0., w=1.)
    obs = NS(image_map={'left': np.zeros((2, 2, 3), np.uint8)}, image_header_map={'left': header},
             camera_info_map={'left': object()}, force_vec=vec, torque_vec=vec, force_mag=0.,
             lateral_force_mag=0., tcp_error=vec, speed_mag=0., joint_positions=vec,
             tcp_pose=NS(position=xyz, orientation=xyz),
             tcp_velocity=NS(linear=xyz, angular=xyz))
    state = NS(**{f'last_pre_insert_{k}': 0. for k in (
        'pose_err_m', 'axis_error_rad', 'orientation_error_rad', 'plug_axis_error_rad', 'plug_orientation_error_rad')},
        phase='approach', retry_count=0)
    target = NS(visible=False, confidence=0., presence_prob=0., centering_score=0., landmark_score=0.,
                rejection_reason='', x_error=0., y_error=0., bbox_width_px=0., bbox_height_px=0.,
                z_distance_m=0., detection_source='', source_camera='', port_pos_base_link=None)
    policy = NS(_capture_dir=tmp_path, _last_capture_time=0., CAPTURE_MIN_PERIOD_SEC=0.,
                _capture_episode_id='e', _capture_counter=0,
                _task_metadata=lambda task: {}, _insertion_metadata=lambda state: {})
    task = NS(cable_name='cable_0', plug_name='sfp_tip')
    CaptureMixin._maybe_capture_sample(policy, task, obs, target, state, 10.)
    metadata = json.loads(next(tmp_path.glob('*.json')).read_text())
    assert metadata['ground_truth'] == {'left': {'port': 1}} and metadata['plug_tip'] == {'left': {'tip': 1}}
