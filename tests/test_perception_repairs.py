"""Regressions for mismatched exposure geometry and partial SC completion."""
from types import SimpleNamespace as NS
from unittest.mock import Mock
import numpy as np
import pytest
from rclpy.time import Time
from std_msgs.msg import Header
from sensor_msgs.msg import CameraInfo
from geometry_msgs.msg import TransformStamped
from aic_model.policy import Policy
from aic_model.policy_perception import synchronized_camera_names, camera_projection_matrix


def header(seconds, frame='camera'):
    return Header(stamp=Time(seconds=seconds).to_msg(), frame_id=frame)


def test_fusion_excludes_stale_future_and_unsynchronized_exposures():
    policy = NS(time_now=lambda: Time(seconds=10))
    obs = NS(image_map={n: np.zeros((1, 1, 3)) for n in ('left','center','right')},
             image_header_map={'left': header(9.99), 'center': header(9.98), 'right': header(9.8)})
    assert set(synchronized_camera_names(policy, obs)) == {'left','center'}
    obs.image_header_map = {'left': header(9.4), 'center': header(10.1), 'right': header(0)}
    assert synchronized_camera_names(policy, obs) == []


def test_projection_uses_image_time_not_latest_or_calibration_time():
    tf = TransformStamped(); tf.transform.rotation.w = 1.0
    buffer = NS(lookup_transform=Mock(return_value=tf))
    policy = NS(_parent_node=NS(_tf_buffer=buffer))
    info = CameraInfo(header=header(1), k=[100.,0.,50.,0.,100.,50.,0.,0.,1.])
    assert camera_projection_matrix(policy, info, None, header(2)) is not None
    assert buffer.lookup_transform.call_args.args[2].nanoseconds == 2_000_000_000
    buffer.lookup_transform.reset_mock()
    assert camera_projection_matrix(policy, info, None, header(2, 'wrong')) is None
    assert camera_projection_matrix(policy, info, None, header(0)) is None
    buffer.lookup_transform.assert_not_called()


def test_capture_ground_truth_uses_the_image_exposure():
    from aic_model.ground_truth import project_port_to_camera, project_frame_origin_to_camera
    tf = TransformStamped(); tf.transform.rotation.w = 1.0
    tf.transform.translation.z = 1.0
    buffer = NS(lookup_transform=Mock(return_value=tf))
    policy = NS(_parent_node=NS(_tf_buffer=buffer))
    info = CameraInfo(header=header(1), width=100, height=100,
                      k=[100.,0.,50.,0.,100.,50.,0.,0.,1.])
    policy._task_mode = lambda _: 'sc'
    policy._port_dimensions_m = lambda _: (.010, .025)
    policy.MODE_CONFIG = Policy.MODE_CONFIG
    task = NS(target_module_name='sc_port_0', port_name='sc_port_base')
    for project, target in ((project_port_to_camera, task),
                            (project_frame_origin_to_camera, 'plug')):
        project(policy, target, info, header(2))
        assert buffer.lookup_transform.call_args.args[2].nanoseconds == 2_000_000_000
        buffer.lookup_transform.reset_mock()
        assert not project(policy, target, info, header(0))['visible']
        assert not project(policy, target, info, header(2, 'wrong'))['visible']
        buffer.lookup_transform.assert_not_called()


@pytest.mark.parametrize('depth,travel,visible,expected', [
    (.0081,.0098,True,False),  # Observed false success in the baseline sample.
    (.016,.010,True,False),   # Perceived depth alone cannot establish success.
    (.016,.016,False,False),  # Lost target cannot accumulate confirmations.
    (float('inf'),.016,True,False),
    (.016,.016,True,True),
])
def test_sc_completion_requires_full_depth_and_travel(depth, travel, visible, expected):
    policy = Policy.__new__(Policy)
    geometry = NS(valid=True, depth_m=depth, xy_m=.001)
    contact = NS(valid=True, state='deep', travel_m=travel)
    assert policy._completion_geometry_ready(geometry, contact, NS(visible=visible),
                                            Policy.MODE_CONFIG['sc']) == expected


@pytest.mark.parametrize("substate", ["deep_seating", "engaged_seating", "face_probe"])
def test_engagement_timeout_precedes_all_motion_branches(monkeypatch, substate):
    from aic_model.policy import InsertState
    from aic_model import policy_state
    policy = Policy.__new__(Policy)
    policy._insert_geometry = Mock(return_value=NS(valid=True))
    contact = NS(valid=True, state='engaged', escaped=False, xy_m=.001,
                 depth_m=.008, travel_m=.010, force_n=8.)
    policy._classify_insert_contact = Mock(return_value=contact)
    policy._log_insert_contact_diag = Mock()
    policy._recover_from_sfp_insert = Mock()
    monkeypatch.setattr(policy_state, 'force_recover_threshold', lambda *_: 30.)
    state = InsertState(insert_substate=substate, engaged_start_time=1.)
    obs = NS(force_mag=8., tcp_error=np.zeros(6))
    # This used to take the pose-seating early return forever despite timeout.
    policy._handle_sfp_insert_phase(Mock(), Mock(), state, obs, NS(),
                                   Policy.MODE_CONFIG['sc'], None, None, 21., None)
    assert policy._recover_from_sfp_insert.call_args.args[0] == 'engaged_seating_timeout'


def test_sc_completion_change_preserves_sfp_contact_gate():
    policy = Policy.__new__(Policy)
    geometry = NS(valid=True, depth_m=.012, xy_m=.0032)
    contact = NS(valid=True, state='deep', travel_m=.019)
    assert policy._completion_geometry_ready(geometry, contact, NS(visible=True),
                                            Policy.MODE_CONFIG['sfp'])


def test_episode_reset_prevents_capture_collisions_and_stale_board_pose():
    policy = Policy.__new__(Policy)
    policy.begin_episode()
    previous = policy._capture_episode_id
    policy._capture_counter = 100
    policy._last_capture_time = 900.0
    policy._board_pose = object()
    policy.begin_episode()
    assert policy._capture_episode_id != previous
    assert policy._capture_counter == 0
    assert policy._last_capture_time < 0  # A reset simulator clock can capture immediately.
    assert policy._board_pose is None
