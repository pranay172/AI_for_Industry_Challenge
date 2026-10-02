"""Bounds, coordinate invariance and aborts for experimental rail framing."""
import sys
from pathlib import Path
from types import SimpleNamespace as NS
import cv2
import numpy as np
import pytest

from aic_model.board_registration import BoardPose, module_bounds
from aic_model.rail_view import plan_rail_view, rail_points


def test_view_motion_is_bounded_withdraws_before_turning_and_is_frame_invariant():
    board=BoardPose(np.eye(3),np.array([.2,.3,.7]),0.)
    R=np.eye(3);camera=np.array([0.,0.,.1]);tcp=np.zeros(3)
    plan=plan_rail_view(board,'nic_card_mount_0',R,camera,R,tcp,max_tcp_distance=2.)
    for elapsed in (-1,0,2,4,8,12,100):
        turn,position=plan.pose_at(elapsed)
        assert np.linalg.norm(position-tcp)<=.100001
        assert np.dot(position-tcp,plan.normal)>=-1e-9
        angle=np.arccos(np.clip((np.trace(turn)-1)/2,-1,1))
        assert angle<=np.deg2rad(25)+1e-8
        if elapsed<=4:
            np.testing.assert_allclose(turn,R,atol=1e-8)
    Q=cv2.Rodrigues(np.array([.2,.3,.7]))[0];offset=np.array([.4,-.3,.2])
    transformed=BoardPose(Q@board.rotation,Q@board.translation+offset,0.)
    other=plan_rail_view(transformed,'nic_card_mount_0',Q@R,Q@camera+offset,Q@R,Q@tcp+offset,max_tcp_distance=2.)
    for elapsed in (0,3,8,100):
        a,b=plan.pose_at(elapsed);c,d=other.pose_at(elapsed)
        np.testing.assert_allclose(c,Q@a,atol=1e-7)
        np.testing.assert_allclose(d,Q@b+offset,atol=1e-7)
    assert module_bounds('nic_card_mount_5') is None
    with pytest.raises(ValueError,match='Unknown module'):
        rail_points(board,'wrong_module')


def test_runtime_step_holds_on_stale_geometry_and_rejects_contact_and_timeout(monkeypatch):
    from geometry_msgs.msg import Pose
    import aic_model.policy_rail_view as adapter
    tcp=Pose();tcp.orientation.w=1.
    parsed=NS(force_mag=0.,tcp_pose=tcp,camera_info_map={},image_header_map={},image_map={})
    policy=NS(_board_pose=BoardPose(np.eye(3),np.array([0.,0.,.8]),0.),_rail_view_start=None,PLAUSIBLE_PORT_DISTANCE_MAX_M=1.)
    task=NS(target_module_name='sc_port_0')
    monkeypatch.setattr(adapter.perception,'synchronized_camera_names',lambda *_:[])
    status,pose=adapter.rail_view_step(policy,parsed,task,1.,6.)
    assert status=='waiting_for_camera_geometry' and pose is not None
    parsed.force_mag=6.
    assert adapter.rail_view_step(policy,parsed,task,1.,6.)==('rail_view_contact',None)
    # A load between the limits holds; only a sustained one aborts.
    status,pose=adapter.rail_view_step(policy,parsed,task,1.,6.,10.)
    assert status=='rail_view_force_hold' and pose is not None
    assert adapter.rail_view_step(policy,parsed,task,1.+adapter.CONTACT_HOLD_SEC-.1,6.,10.)[0]=='rail_view_force_hold'
    assert adapter.rail_view_step(policy,parsed,task,1.+adapter.CONTACT_HOLD_SEC,6.,10.)==('rail_view_contact',None)
    parsed.force_mag=0.
    adapter.rail_view_step(policy,parsed,task,5.,6.,10.)
    parsed.force_mag=6.
    assert adapter.rail_view_step(policy,parsed,task,5.1,6.,10.)[0]=='rail_view_force_hold'
    parsed.force_mag=10.
    assert adapter.rail_view_step(policy,parsed,task,5.2,6.,10.)==('rail_view_contact',None)
    policy._rail_view_suspended=None
    parsed.force_mag=0.;policy._rail_view_start=0.
    assert adapter.rail_view_step(policy,parsed,task,13.,6.)==('rail_view_timeout',None)
    monkeypatch.setattr(adapter.perception,'synchronized_camera_names',lambda *_:['center','left'])
    parsed.image_header_map={'center':None,'left':None}
    parsed.image_map={'center':np.zeros((100,100,3),np.uint8),'left':np.zeros((100,100,3),np.uint8)}
    monkeypatch.setattr(adapter.perception,'camera_projection_matrix',lambda *_:(np.eye(3),np.eye(3),np.zeros(3)))
    monkeypatch.setattr(adapter,'rail_in_view',lambda *_:True)
    assert adapter.rail_view_step(policy,parsed,task,13.,6.)[0]=='framed'


def test_collection_motion_requires_explicit_opt_in(tmp_path):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    from collect_initial_views import collection_compose
    default=collection_compose(tmp_path,'eval','model','scene',False)
    motion=collection_compose(tmp_path,'eval','model','scene',False,rail_views=True)
    assert 'policy:=aic_model.CaptureInitialViews' in default['services']['model']['command']
    assert 'policy:=aic_model.CaptureRailViews' in motion['services']['model']['command']


def test_distance_budget_caps_every_public_placement_throughout_motion():
    from aic_model.rail_view import rail_distance_bound, DISTANCE_MARGIN_M
    board=BoardPose(np.eye(3),np.array([0.,0.,0.]),0.)
    tcp=np.array([0.,0.,.32]);camera=tcp+np.array([0.,0.,.04])
    R=np.diag([1.,-1.,-1.])
    plan=plan_rail_view(board,'sc_port_1',R,camera,R,tcp,max_tcp_distance=.4)
    assert 0 < plan.retreat_m < .1
    for elapsed in np.linspace(0,20,101):
        _,position=plan.pose_at(elapsed)
        assert rail_distance_bound(board,'sc_port_1',position)<=.4-DISTANCE_MARGIN_M+1e-12
    Q=cv2.Rodrigues(np.array([.3,-.2,.4]))[0];shift=np.array([1.,2.,3.])
    rotated=BoardPose(Q,shift,0.)
    other=plan_rail_view(rotated,'sc_port_1',Q@R,Q@camera+shift,Q@R,Q@tcp+shift,max_tcp_distance=.4)
    assert other.retreat_m==pytest.approx(plan.retreat_m)


def test_outside_distance_budget_is_not_reported_as_framed(monkeypatch):
    from geometry_msgs.msg import Pose
    import aic_model.policy_rail_view as adapter
    from aic_model.rail_view import RailDistanceError
    board=BoardPose(np.eye(3),np.zeros(3),0.)
    tcp=Pose();tcp.position.z=.5;tcp.orientation.w=1.
    policy=NS(_board_pose=board,_rail_view_start=None,PLAUSIBLE_PORT_DISTANCE_MAX_M=.4)
    parsed=NS(force_mag=0.,tcp_pose=tcp,camera_info_map={},image_header_map={'center':None,'left':None},
              image_map={n:np.zeros((100,100,3),np.uint8) for n in ('center','left')})
    monkeypatch.setattr(adapter.perception,'synchronized_camera_names',lambda *_:['center','left'])
    monkeypatch.setattr(adapter.perception,'camera_projection_matrix',lambda *_:(np.eye(3),np.eye(3),np.zeros(3)))
    monkeypatch.setattr(adapter,'rail_in_view',lambda *_:True)
    status,pose=adapter.rail_view_step(policy,parsed,NS(target_module_name='sc_port_0'),1.,6.)
    # Hold where the target is guaranteed visible; never move toward the board.
    assert status=='rail_view_distance_limit' and pose is not parsed.tcp_pose and pose==parsed.tcp_pose
    assert getattr(policy,'_rail_view_plan',None) is None
    with pytest.raises(RailDistanceError):
        plan_rail_view(board,'sc_port_0',np.eye(3),np.array([0.,0.,.5]),np.eye(3),np.array([0.,0.,.5]))


def framing_fixture(monkeypatch):
    from geometry_msgs.msg import Pose
    import aic_model.policy_rail_view as adapter
    tcp=Pose();tcp.orientation.w=1.
    parsed=NS(force_mag=0.,tcp_pose=tcp,camera_info_map={},image_header_map={'center':None,'left':None},
              image_map={'center':np.zeros((10,10,3),np.uint8),'left':np.zeros((10,10,3),np.uint8)})
    policy=NS(_board_pose=BoardPose(np.eye(3),np.array([0.,0.,.3]),0.),_rail_view_start=None,
              _rail_view_plan=None,PLAUSIBLE_PORT_DISTANCE_MAX_M=1.)
    monkeypatch.setattr(adapter.perception,'synchronized_camera_names',lambda *_:['center','left'])
    monkeypatch.setattr(adapter.perception,'camera_projection_matrix',
                        lambda *_:(np.eye(3),np.eye(3),np.array([0.,0.,.05])))
    monkeypatch.setattr(adapter,'rail_in_view',lambda *_:False)
    return adapter,policy,parsed,NS(target_module_name='sc_port_0')


def test_suspended_framing_neither_jumps_along_the_plan_nor_times_out(monkeypatch):
    adapter,policy,parsed,task=framing_fixture(monkeypatch)
    assert adapter.rail_view_step(policy,parsed,task,1.,6.)[0]=='framing_requested_rail'
    adapter.rail_view_step(policy,parsed,task,2.,6.)
    adapter.suspend_rail_view(policy,2.5)   # target evidence paused framing
    adapter.suspend_rail_view(policy,9.)    # repeated suspension keeps the first time
    status,pose=adapter.rail_view_step(policy,parsed,task,30.,6.)
    assert status=='framing_requested_rail'
    _,expected=policy._rail_view_plan.pose_at(1.5)
    np.testing.assert_allclose([pose.position.x,pose.position.y,pose.position.z],expected)
    # Active framing time still bounds the episode.
    assert adapter.rail_view_step(policy,parsed,task,30.+adapter.TIMEOUT_SECONDS,6.)==('rail_view_timeout',None)


def test_reacquisition_after_lock_plans_from_the_current_pose(monkeypatch):
    adapter,policy,parsed,task=framing_fixture(monkeypatch)
    adapter.rail_view_step(policy,parsed,task,1.,6.)
    first=policy._rail_view_plan
    adapter.reset_rail_view(policy)          # target locked
    parsed.tcp_pose.position.z=.05            # controller moved toward the port
    status,_=adapter.rail_view_step(policy,parsed,task,60.,6.)
    assert status=='framing_requested_rail' and policy._rail_view_start==60.
    assert policy._rail_view_plan is not first
    np.testing.assert_allclose(policy._rail_view_plan.tcp_translation,[0.,0.,.05])


def test_time_spent_framed_does_not_count_toward_the_framing_timeout(monkeypatch):
    adapter,policy,parsed,task=framing_fixture(monkeypatch)
    adapter.rail_view_step(policy,parsed,task,1.,6.)
    monkeypatch.setattr(adapter,'rail_in_view',lambda *_:True)
    assert adapter.rail_view_step(policy,parsed,task,3.,6.)[0]=='framed'
    monkeypatch.setattr(adapter,'rail_in_view',lambda *_:False)
    status,pose=adapter.rail_view_step(policy,parsed,task,40.,6.)
    assert status=='framing_requested_rail'
    _,expected=policy._rail_view_plan.pose_at(2.)
    np.testing.assert_allclose([pose.position.x,pose.position.y,pose.position.z],expected)


def board_return_fixture():
    from aic_model.policy import Policy
    policy=Policy.__new__(Policy)
    policy.begin_episode()
    policy._board_view_origin='start'; policy._board_view_start=0.
    policy._board_view_pose=lambda remaining:remaining   # the pose is just the search time left
    return policy


def test_board_return_retraces_the_search_then_holds_at_the_start():
    policy=board_return_fixture()
    assert policy._board_return_pose(False,10.)==pytest.approx(10.)
    assert policy._board_return_pose(False,14.)==pytest.approx(6.)
    assert policy._board_return_pose(False,20.) is None
    assert policy._board_view_origin is None and policy._hold_pose=='start'


def test_accepted_target_pauses_the_return_and_stops_its_clock():
    policy=board_return_fixture()
    policy._board_return_pose(False,10.)
    assert policy._board_return_pose(True,12.) is None                # hold here for the lock window
    assert policy._board_return_pose(False,12.4) is None              # brief gap: still paused
    assert policy._board_view_origin=='start'
    # Evidence gone for longer than the pause: resume where the return stopped.
    t=12.+policy.BOARD_RETURN_PAUSE_SEC+.1
    assert policy._board_return_pose(False,t)==pytest.approx(8.)
    assert policy._board_return_pose(False,t+8.) is None and policy._hold_pose=='start'


def test_collection_holds_through_a_brief_load_like_the_policy(monkeypatch):
    """sfp-train-d-smoke: the collector ended at the recover threshold after 6 s."""
    from geometry_msgs.msg import Pose
    import aic_model.CaptureRailViews as collector
    policy=collector.CaptureRailViews.__new__(collector.CaptureRailViews)
    clock={'t':0.};calls=[];logs=[]
    tcp=Pose();tcp.orientation.w=1.
    policy.begin_episode=lambda:None
    policy.time_now=lambda:NS(nanoseconds=int(clock['t']*1e9))
    policy._parent_node=NS(check_policy_execution=lambda:None)
    policy._wait_for_capture_tf=lambda *args:None
    policy._maybe_capture_sample=lambda *args:None
    policy._capture_counter=0
    policy.get_logger=lambda:NS(info=logs.append)
    # Calm start (baseline 20 N), then 8 s over the recover threshold for 1 s.
    policy._parse_observation=lambda observation:NS(force_mag=31. if 8<=clock['t']<9 else 20.,tcp_pose=tcp)
    monkeypatch.setattr(collector.perception,'register_board',lambda *args:object())
    monkeypatch.setattr(collector.motion,'send_motion',lambda *args:None)
    monkeypatch.setattr(collector.time,'sleep',lambda dt:clock.__setitem__('t',clock['t']+.1))

    def step(policy,parsed,task,now,force_limit,abort_limit):
        calls.append((now,force_limit,abort_limit))
        return ('rail_view_force_hold' if parsed.force_mag>=force_limit else 'moving'),tcp
    monkeypatch.setattr(collector,'rail_view_step',step)
    policy.insert_cable(NS(),lambda:object(),None,lambda message:None)
    assert all(abort>limit for _,limit,abort in calls)
    assert max(now for now,_,_ in calls)>9.5
    assert not any('rail_view_contact' in line for line in logs)
