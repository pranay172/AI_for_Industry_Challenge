"""Recorded inputs must pass the same estimator and state filter as live control."""
from types import SimpleNamespace as NS
import copy
import numpy as np
import pytest
from replay_policy_pose import ReplayPolicy, ExposureTransforms, evaluate_observation
from aic_model.board_registration import BoardPose
from aic_model.sc_heatmap_detector import sc_landmarks_port
from aic_model.policy_perception import filter_target_position
from aic_model.policy import TargetEstimate
from rclpy.time import Time
from tf2_ros import TransformException


def fixture(points=None, confidence=None, tcp=(0,0,.3)):
    if points is None:
        points=(np.diag([1.,-1.,-1.])@sc_landmarks_port().T).T+[-.05,.0295,.5165]
    if confidence is None:confidence=np.full(5,.9)
    K=np.array([[500.,0,320],[0,500.,240],[0,0,1.]])
    images={};geometry={};exposures={};predictions=[]
    for i,(name,x) in enumerate(zip(('center','left','right'),(0,-.1,.1))):
        image=np.zeros((480,640,3),np.uint8);image[0,0,0]=i;images[name+'_image']=image
        t=np.array([x,0,0]);uv=(K@(points-t).T).T;uv=uv[:,:2]/uv[:,2:]
        predictions.append({'points_px':uv,'confidence':confidence,'image_size':(640,480)})
        geometry[name]={'K':K.tolist(),'R_base_from_camera':np.eye(3).tolist(),'t_base_from_camera':t.tolist()}
        exposures[name]={'frame_id':name+'/optical','sec':1,'nanosec':0}
    detector=NS(preprocessing='full_frame_v1',infer=lambda image:predictions[int(image[0,0,0])])
    policy=ReplayPolicy(detector);policy._board_pose=BoardPose(np.eye(3),np.array([0,0,.5]),0.)
    state=NS(port_pos_smoothed=None,phase='find_target')
    metadata={'camera_geometry':geometry,'camera_exposures':exposures,'capture_time_sim':1.,
              'controller':{'tcp_pose':{'position':dict(zip(('x','y','z'),tcp))}},
              'task':{'target_module_name':'sc_port_0','port_name':'sc_port_base'},'phase':'find_target'}
    return policy,state,metadata,images


def test_actual_estimator_and_filter_accept_correct_pose_without_gt():
    policy,state,m,images=fixture()
    row=evaluate_observation(policy,state,m,images)
    assert row['visible'] and row['status']=='accepted'
    assert row['estimator']['stage']=='accepted'
    assert row['estimator']['geometry_residual_m']<1e-10
    np.testing.assert_allclose(row['raw_face_position'],[-.05,.0295,.53214],atol=1e-9)
    # Poisoning privileged and previous detector results cannot affect acceptance.
    m.update(ground_truth={'center':{'xyz_camera':[999]*3}},target={'visible':False},plug_tip={'bad':True})
    policy,state,_,_=fixture()
    assert evaluate_observation(policy,state,m,images)==row


@pytest.mark.parametrize('case,expected',[
    ('wrong_rail','wrong_module_rail'),('deformed','geometry_residual'),
    ('low_confidence','insufficient_cameras'),('nan_confidence','insufficient_cameras'),
    ('far_tcp','implausible_tcp_distance'),('stale','insufficient_cameras')])
def test_live_gates_are_reported(case,expected):
    points=(np.diag([1.,-1.,-1.])@sc_landmarks_port().T).T+[-.05,.0295,.5165]
    if case=='wrong_rail':points[:,1]+=.041
    if case=='deformed':points[4,0]+=.12
    conf=np.full(5,.9)
    if case=='low_confidence':conf[4]=.1
    if case=='nan_confidence':conf[4]=np.nan
    policy,state,m,images=fixture(points,conf,tcp=(0,0,2) if case=='far_tcp' else (0,0,.3))
    if case=='stale':m['capture_time_sim']=2.
    row=evaluate_observation(policy,state,m,images)
    assert not row['visible'] and row['status']==expected
    assert state.port_pos_smoothed is None


def test_exposure_adapter_refuses_latest_and_hidden_object_transforms():
    _,_,m,_=fixture();buffer=ExposureTransforms(m['camera_exposures'],m['camera_geometry'])
    assert buffer.lookup_transform('base_link','center/optical',Time(seconds=1))
    for target,source,stamp in [('base_link','center/optical',Time()),
        ('base_link','task_board/sc_port_0/sc_port_base_link',Time(seconds=1)),
        ('other','center/optical',Time(seconds=1))]:
        with pytest.raises(TransformException):buffer.lookup_transform(target,source,stamp)


def test_shared_position_filter_preserves_search_ema_and_close_spatial_lock():
    policy,state,m,images=fixture()
    parsed=NS(tcp_pose=NS(position=NS(x=0.,y=0.,z=.3)))
    def target(position):return TargetEstimate(visible=True,confidence=.9,detection_source='sc_heatmap',port_pos_base_link=np.array(position))
    assert filter_target_position(policy,target([0,0,.4]),parsed,state)=='accepted'
    t=target([.04,0,.4])
    assert filter_target_position(policy,t,parsed,state)=='accepted' # Unlocked search.
    np.testing.assert_allclose(t.port_pos_base_link,[.006,0,.4])
    state.phase='coarse_align';t=target([.05,0,.4])
    assert filter_target_position(policy,t,parsed,state)=='spatial_lock_hold'
    np.testing.assert_allclose(t.port_pos_base_link,[.006,0,.4])
    state.phase='insert';t=target([.011,0,.4])
    assert filter_target_position(policy,t,parsed,state)=='spatial_lock_hold'
    t=target([.006,0,.45]) # Close-phase lock uses XY, not insertion depth.
    assert filter_target_position(policy,t,parsed,state)=='accepted'
    np.testing.assert_allclose(t.port_pos_base_link,[.006,0,.4075])


def test_replay_retries_missing_board_and_resets_registration_each_episode(tmp_path, monkeypatch):
    import json
    from unittest.mock import Mock
    from replay_policy_pose import replay
    from aic_model import board_registration
    policy,_,metadata,images=fixture()
    for index,episode in enumerate(('a','a','b')):
        m=copy.deepcopy(metadata)
        m.update(episode_id=episode,sample_id=str(index),scene_id=episode,
                 files={'images_npz':f'{index}.npz'})
        m['task']['plug_type']='sc'
        np.savez(tmp_path/f'{index}.npz',**images)
        (tmp_path/f'{index}.json').write_text(json.dumps(m))
    register=Mock(side_effect=[None,policy._board_pose,None])
    monkeypatch.setattr(board_registration,'register_views',register)
    result=replay([tmp_path],policy._sc_port_detector)
    assert result['summary']['status_counts']=={'board_marker_unregistered':2,'accepted':1}
    assert result['summary']['used_center_error_mm']['count']==0 # No GT required.
    assert register.call_count==3


def test_replay_refuses_to_invent_missing_tcp_position():
    policy,state,m,images=fixture();del m['controller']['tcp_pose']
    with pytest.raises(ValueError,match='TCP position'):
        evaluate_observation(policy,state,m,images)


@pytest.mark.parametrize('translations, status, cameras', [
    ((-.060, -.0605, -.061), 'accepted', 'center+left+right'),
    ((-.060, -.066, -.0605), 'accepted', 'center+right'),
])
def test_v2_decoder_triangulates_only_views_that_agree_on_rail_placement(translations, status, cameras):
    policy, state, m, images = fixture()
    detector = policy._sc_port_detector
    original = detector.infer
    detector.decoder = 'rail_template_face_v2'
    detector.infer = lambda image: {**original(image), 'rail_translation': translations[int(image[0, 0, 0])]}
    row = evaluate_observation(policy, state, m, images)
    assert row['status'] == status and row['source_cameras'] == cameras
    if cameras == 'center+right':
        assert row['estimator']['cameras']['left'] == 'inconsistent_rail_placement'


def test_v2_decoder_rejects_two_views_that_disagree_on_rail_placement():
    policy, state, m, images = fixture()
    detector = policy._sc_port_detector
    original = detector.infer
    detector.decoder = 'rail_template_face_v2'
    detector.infer = lambda image: (None if int(image[0, 0, 0]) == 2 else
                                    {**original(image), 'rail_translation': (-.060, -.066)[int(image[0, 0, 0])]})
    row = evaluate_observation(policy, state, m, images)
    assert row['status'] == 'inconsistent_rail_placement' and not row['visible']


def test_sfp_replay_reports_the_rejecting_stage_per_camera():
    policy, state, m, images = fixture()
    detector = NS(preprocessing='full_frame_v1', infer=lambda image: None)
    sfp = ReplayPolicy(detector, 'sfp')
    sfp._board_pose = policy._board_pose
    m = dict(m, task={'target_module_name': 'nic_card_mount_0', 'port_name': 'sfp_port_0'})
    row = evaluate_observation(sfp, NS(port_pos_smoothed=None, phase='find_target'), m, images)
    assert row['status'] == 'insufficient_cameras' and not row['visible']
    assert set(row['estimator']['cameras'].values()) == {'no_prediction'}
