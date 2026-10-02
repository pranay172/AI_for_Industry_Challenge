"""Pre-insert under a steady cable load: the plateau handover to the face search."""
from copy import deepcopy
from types import SimpleNamespace as NS

import numpy as np
from geometry_msgs.msg import Pose

from aic_model import policy_motion, policy_phases
from aic_model.policy import Policy
from aic_model.policy_types import CycleContext, InsertContactState, InsertState, TargetEstimate

# insertion-return-pause-001: the arm held 4.8-5.8 mm off its pre-insert goal.
LOAD = np.array([-.0045, .0032])


def run_pre_insert(monkeypatch, cycles=40, dt=.4, load=lambda step: LOAD, mode='sfp', spring=False):
    """Toy arm: the measured TCP (= plug) settles load(step) short of every commanded goal.

    With spring, the cable holds it load(step) off the port whatever is commanded,
    as in the race runs, where fine centering did not move it."""
    policy = Policy.__new__(Policy)
    logs = []
    policy.get_logger = lambda: NS(info=logs.append, warn=logs.append)
    policy._send_status = lambda *args: None
    policy._log_recover_diag = lambda *args, **kwargs: None
    # SFP requires the locked port orientation; it is locked and matched here.
    policy._requires_pre_insert_orientation_lock = lambda *args: True
    policy._pre_insert_orientation_errors = lambda *args: (0., 0.)
    policy._plug_to_port_orientation_errors = lambda *args: (0., 0.)
    offset_m = .008

    def preinsert_pose(policy, parsed, port, offset, **kwargs):
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = map(float, np.asarray(port)+[0., 0., offset])
        return pose
    monkeypatch.setattr(policy_motion, 'compute_preinsert_pose', preinsert_pose)
    monkeypatch.setattr(policy_phases._st, 'force_recover_threshold', lambda *args: 1e9)
    commands = []
    monkeypatch.setattr(policy_motion, 'send_motion', lambda policy, move, command: commands.append(command))
    mode_cfg = Policy.MODE_CONFIG[mode]
    state = InsertState(); state.phase = 'pre_insert'; state.phase_start_time = 0.
    state.locked_port_quat = Pose().orientation
    port = np.array([.1, .2, .0])
    target = TargetEstimate(visible=True, confidence=.9, detection_source='sfp_heatmap', port_pos_base_link=port)
    tcp = Pose(); tcp.position.x, tcp.position.y, tcp.position.z = .1-load(0)[0], .2-load(0)[1], offset_m
    for step in range(cycles):
        if state.phase != 'pre_insert':
            break
        if commands:
            goal = commands[-1].pose.position
            anchor = port[:2] if spring else (goal.x, goal.y)
            tcp = deepcopy(tcp); tcp.position.x, tcp.position.y = anchor[0]-load(step)[0], anchor[1]-load(step)[1]
        parsed = NS(force_mag=0., speed_mag=0., tcp_pose=tcp)
        plug = np.array([tcp.position.x, tcp.position.y, tcp.position.z-offset_m])
        policy._pre_insert_step(CycleContext(
            task=NS(), mode=mode, mode_cfg=mode_cfg, insert_state=state, parsed_obs=parsed, target=target,
            raw_position=None, plug_pos_base=plug, plug_quat_base=None, now_wall=step*dt,
            move_robot=None, send_feedback=lambda message: None))
    return state, logs, step*dt


def test_a_settled_residual_hands_over_before_the_timeout(monkeypatch):
    state, logs, elapsed = run_pre_insert(monkeypatch)
    assert state.phase == 'insert'
    assert any(line.startswith('[pre_insert] plateaued') for line in logs)
    assert Policy.PRE_INSERT_PLATEAU_MIN_SEC <= elapsed < Policy.MODE_CONFIG['sfp']['pre_insert_timeout_sec']


def test_a_residual_still_shrinking_does_not_count_as_a_plateau(monkeypatch):
    # The load relaxes 0.6 mm per cycle from 12 mm, so the residual keeps closing.
    relaxing = lambda step: np.array([0., max(.012-.0006*step, .0005)])
    state, logs, elapsed = run_pre_insert(monkeypatch, load=relaxing)
    assert not any(line.startswith('[pre_insert] plateaued') for line in logs)
    assert any(line.startswith('[fine_center]') for line in logs)


# An SC race grasp, even with the true grasp, held the arm 5.2 mm off.
RACE_LOAD = np.array([-.0030, .0043])


def test_an_sc_race_stall_hands_over_within_the_contact_guided_range(monkeypatch):
    state, logs, _ = run_pre_insert(monkeypatch, load=lambda step: RACE_LOAD, mode='sc', spring=True)
    assert state.phase == 'insert'
    assert any(line.startswith('[pre_insert] plateaued') for line in logs)


def test_sfp_keeps_the_spiral_radius_and_sc_rejects_beyond_its_range(monkeypatch):
    for mode, load in (('sfp', RACE_LOAD), ('sc', np.array([-.0045, .0057]))):
        state, logs, _ = run_pre_insert(monkeypatch, cycles=15, load=lambda step: load, mode=mode, spring=True)
        assert state.phase == 'pre_insert' and not any('handing over' in line for line in logs)


def face_search_starts(monkeypatch, mode, xy_m, stall_handover=True):
    """One insert cycle with the plug on the face, xy_m off the port, the moment it lands."""
    policy = Policy.__new__(Policy)
    logs = []
    policy.get_logger = lambda: NS(info=logs.append, warn=logs.append)
    policy._send_status = lambda *args: None
    monkeypatch.setattr(policy_motion, 'send_motion', lambda *args: None)
    port = np.array([.1, .2, .0])
    state = InsertState(); state.phase = 'insert'; state.stall_handover = stall_handover
    state.locked_port_quat = Pose().orientation; state.last_progress_time = 5.
    target = TargetEstimate(visible=True, confidence=.9, port_pos_base_link=port)
    contact = InsertContactState(valid=True, state='face_slide', xy_m=xy_m, depth_m=0.)
    parsed = NS(tcp_pose=Pose())
    policy._face_search_step(
        None, lambda message: None, state, parsed, target, Policy.MODE_CONFIG[mode],
        port+[xy_m, 0., 0.], 5., NS(plug_type=mode, port_type=mode), contact)
    return state.face_search is not None


def test_an_sc_plug_landing_off_centre_after_a_stall_handover_starts_the_face_search(monkeypatch):
    # A race trial: handed over at 5.5 mm, landed at 6.2 mm, recentered 6 times.
    assert face_search_starts(monkeypatch, 'sc', .0062)
    # A gate pass that slides, a slide beyond the range, and SFP still recenter.
    assert not face_search_starts(monkeypatch, 'sc', .0062, stall_handover=False)
    assert not face_search_starts(monkeypatch, 'sc', .0090)
    assert not face_search_starts(monkeypatch, 'sfp', .0050)
