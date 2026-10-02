"""Close-range SFP yaw lock: samples, gating and the corrected orientation."""

import numpy as np
import pytest
from geometry_msgs.msg import Pose

from aic_model import policy_geometry as geom
from aic_model import policy_state as st
from aic_model.policy import InsertState, TargetEstimate


def port_rotation(yaw_deg):
    """SFP port frame: insertion axis straight down, x axis yawed in the table plane."""
    yaw = np.deg2rad(yaw_deg)
    x = np.array([np.cos(yaw), np.sin(yaw), 0.])
    z = np.array([0., 0., -1.])
    return np.column_stack([x, np.cross(z, x), z])


def estimate(yaw_deg, position=(.0, .0, .2), reason='pose=sfp_face'):
    return TargetEstimate(visible=True, confidence=.9, detection_source='sfp_heatmap',
                          rejection_reason=reason, port_pos_base_link=np.array(position, dtype=float),
                          port_rot_base_link=port_rotation(yaw_deg))


def world_yaw(rotation):
    return float(np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0])))


def test_yaw_about_axis_is_signed_about_the_reference_axis():
    # The SFP axis points down, so a positive world yaw is negative about the port axis.
    assert np.isclose(np.degrees(st.yaw_about_axis(port_rotation(0.), port_rotation(1.5))), -1.5)
    assert np.isclose(np.degrees(st.yaw_about_axis(port_rotation(10.), port_rotation(7.))), 3.)


def test_only_fresh_close_range_estimates_are_recorded():
    state = InsertState()
    assert not st.record_close_yaw(state, estimate(0.), (0., 0., .2+st.CLOSE_YAW_MAX_TCP_DISTANCE_M+.01))
    assert not st.record_close_yaw(state, estimate(0., reason='held_static_lock'), (0., 0., .3))
    assert st.record_close_yaw(state, estimate(0.), (0., 0., .3))
    assert len(state.close_yaw_rotations) == 1


def test_sample_buffer_keeps_the_most_recent_estimates():
    state = InsertState()
    for yaw in range(st.CLOSE_YAW_MAX_SAMPLES+5):
        st.record_close_yaw(state, estimate(float(yaw)), (0., 0., .3))
    assert len(state.close_yaw_rotations) == st.CLOSE_YAW_MAX_SAMPLES
    assert np.isclose(world_yaw(state.close_yaw_rotations[0]), 5.)


def test_lock_takes_median_close_yaw_and_keeps_the_axis():
    state = InsertState()
    for yaw in (1.1, 0.9, 1.0, 9.0):   # one outlier
        st.record_close_yaw(state, estimate(yaw), (0., 0., .3))
    smoothed = geom.matrix_to_quaternion(port_rotation(-1.))
    locked, correction = st.close_range_quat(state, smoothed)
    assert np.isclose(np.degrees(correction), -2.05, atol=1e-6)     # about the downward axis
    rotation = geom.quaternion_to_matrix(locked)
    assert np.allclose(rotation[:, 2], [0., 0., -1.], atol=1e-9)
    assert np.isclose(world_yaw(rotation), 1.05, atol=1e-6)


def test_too_few_or_implausible_samples_keep_the_smoothed_orientation():
    smoothed = geom.matrix_to_quaternion(port_rotation(0.))
    state = InsertState()
    for yaw in (1., 1.):
        st.record_close_yaw(state, estimate(yaw), (0., 0., .3))
    assert st.close_range_quat(state, smoothed) == (smoothed, None)
    far = InsertState()
    for yaw in (20., 20., 20.):
        st.record_close_yaw(far, estimate(yaw), (0., 0., .3))
    assert st.close_range_quat(far, smoothed) == (smoothed, None)


def test_ready_after_enough_samples_or_the_wait_limit():
    state = InsertState()
    assert not st.close_yaw_ready(state, 10.)
    assert not st.close_yaw_ready(state, 10.+st.CLOSE_YAW_MAX_WAIT_SEC-.1)
    assert st.close_yaw_ready(state, 10.+st.CLOSE_YAW_MAX_WAIT_SEC)
    ready = InsertState()
    for _ in range(st.CLOSE_YAW_MIN_SAMPLES):
        st.record_close_yaw(ready, estimate(0.), (0., 0., .3))
    assert st.close_yaw_ready(ready, 0.)


def test_new_acquisition_clears_samples_and_coarse_align_restarts_the_wait():
    state = InsertState()
    st.record_close_yaw(state, estimate(0.), (0., 0., .3))
    state.close_yaw_wait_start = 5.
    st.set_phase(state, 'coarse_align', 6.)
    assert state.close_yaw_wait_start is None and len(state.close_yaw_rotations) == 1
    st.set_phase(state, 'find_target', 7.)
    assert state.close_yaw_rotations == []


def rotation(axis, angle):
    axis = np.asarray(axis, dtype=float)/np.linalg.norm(axis)
    k = np.array([[0., -axis[2], axis[1]], [axis[2], 0., -axis[0]], [-axis[1], axis[0], 0.]])
    return np.eye(3)+np.sin(angle)*k+(1.-np.cos(angle))*(k@k)


def test_rotation_about_axis_measures_turns_about_a_tilted_axis():
    axis = np.array([.1, -.2, -1.])/np.linalg.norm([.1, -.2, -1.])
    reference = rotation([1., 2., 3.], .7)          # arbitrary TCP orientation
    turned = rotation(axis, np.deg2rad(1.5))@reference
    assert np.isclose(np.degrees(st.rotation_about_axis(reference, turned, axis)), 1.5)


def test_yaw_dither_sweeps_then_stops_once_the_plug_is_inside():
    state = InsertState(phase='insert', phase_start_time=10.)
    axis = np.array([0., 0., -1.])
    reference = rotation([0., 0., 1.], .3)
    rate, _, lever = st.yaw_dither(state, reference, (0., 0., .3), (0., .02, .25), axis, 10.)
    assert rate == 0. and np.allclose(lever, [0., -.02, .05])
    rate, _, _ = st.yaw_dither(state, reference, (0., 0., .3), (0., .02, .25), axis,
                               10.+st.YAW_DITHER_PERIOD_SEC/4)
    assert np.isclose(rate, st.YAW_DITHER_MAX_RATE_RAD_S)      # far below +amplitude: clipped
    at_peak = rotation(axis, st.YAW_DITHER_AMPLITUDE_RAD)@reference
    rate, _, _ = st.yaw_dither(state, at_peak, (0., 0., .3), (0., .02, .25), axis, 10.+st.YAW_DITHER_PERIOD_SEC/4)
    assert abs(rate) < 1e-9
    state.max_insert_travel_m = st.YAW_DITHER_MAX_TRAVEL_M+.001
    assert st.yaw_dither(state, reference, (0., 0., .3), (0., .02, .25), axis, 12.)[0] == 0.
    assert st.yaw_dither(state, reference, (0., 0., .3), None, axis, 12.) is None


def test_yaw_dither_reference_restarts_with_each_insert_attempt():
    state = InsertState(phase='insert', phase_start_time=1.)
    st.yaw_dither(state, np.eye(3), (0., 0., .3), (0., 0., .2), (0., 0., -1.), 1.)
    state.phase_start_time = 5.
    st.yaw_dither(state, rotation([0., 0., 1.], .2), (0., 0., .3), (0., 0., .2), (0., 0., -1.), 5.)
    assert np.allclose(state.yaw_dither_reference, rotation([0., 0., 1.], .2))


@pytest.mark.parametrize('frame', ['base_link', 'gripper/tcp'])
def test_dithered_command_pivots_about_the_plug_tip(frame):
    from aic_model import policy_motion as motion
    tcp_rotation = rotation([1., -1., .5], 2.)
    tcp_quat = geom.matrix_to_quaternion(tcp_rotation)
    axis, lever = np.array([0., 0., -1.]), np.array([.01, -.03, .05])
    command = motion.build_twist_command((.001, 0., .002), frame_id=frame)
    dithered = motion.with_yaw_dither(command, (.05, axis, lever), tcp_quat)
    linear = np.array([dithered.twist.linear.x, dithered.twist.linear.y, dithered.twist.linear.z])
    angular = np.array([dithered.twist.angular.x, dithered.twist.angular.y, dithered.twist.angular.z])
    added = linear-np.array([.001, 0., .002])
    if frame == 'gripper/tcp':
        added, angular = tcp_rotation@added, tcp_rotation@angular
    assert np.allclose(angular, .05*axis)
    assert np.allclose(added+np.cross(angular, -lever), 0.)     # the tip does not move
    assert motion.with_yaw_dither(motion.build_pose_command(Pose()), (.05, axis, lever), tcp_quat).twist is None


def test_sc_port_orientation_follows_the_board():
    from aic_model.sc_face_decoder import sc_port_rotation_board
    rotation = sc_port_rotation_board()
    assert np.allclose(rotation.T@rotation, np.eye(3), atol=1e-9)
    assert np.dot(rotation[:, 2], [0., 0., 1.]) < -.9999       # insertion axis into the board


def test_reacquisition_keeps_the_dropped_lock_for_the_attempt_end():
    state = InsertState()
    st.set_phase(state, 'find_target', 1.)
    assert state.last_lock is None
    state.held_target = estimate(2., position=(.1, .2, .3))
    state.locked_port_quat = geom.matrix_to_quaternion(port_rotation(2.))
    st.set_phase(state, 'find_target', 2.)
    assert state.held_target is None
    assert np.allclose(state.last_lock[0], [.1, .2, .3])
