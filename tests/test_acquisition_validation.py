"""Acquisition validation runs the live loop and stops before any insertion motion."""
import json
from types import SimpleNamespace as NS
from unittest.mock import Mock

import numpy as np
import pytest
from geometry_msgs.msg import Pose
from rclpy.time import Time
from std_msgs.msg import Header

import aic_model.ValidateAcquisition as module
from aic_model import ground_truth, policy_motion, policy_perception
from aic_model.board_registration import BoardPose
from aic_model.policy import TargetEstimate


class Clock:
    def __init__(self):
        self.ns = 1_000_000_000

    def now(self):
        return Time(nanoseconds=self.ns)

    def sleep(self, seconds):
        self.ns += round(seconds*1e9)


def make_policy(tmp_path, clock):
    policy = module.ValidateAcquisition.__new__(module.ValidateAcquisition)
    policy._parent_node = NS(check_policy_execution=lambda: None)
    policy.get_logger = lambda: NS(info=lambda *_: None, warn=lambda *_: None)
    policy.time_now = clock.now
    policy.sleep_for = clock.sleep
    policy._board_pose = None
    policy._sfp_detector = policy._sc_port_detector = object()
    policy._capture_dir = None
    policy._trace_dir = tmp_path
    policy._parse_observation = lambda _msg: observation(clock)
    # The fake TCP sits at the origin; the return to home has its own tests.
    policy._restore_start_pose = lambda *_: None
    return policy


def observation(clock):
    pose = Pose()
    pose.orientation.w = 1.
    header = Header(frame_id='center_camera', stamp=clock.now().to_msg())
    return NS(tcp_pose=pose, force_mag=0., image_map={'center': np.zeros((2, 2, 3), np.uint8)},
              image_header_map={'center': header}, camera_info_map={})


def task():
    return NS(plug_type='sc', port_type='sc', port_name='sc_port_base', target_module_name='sc_port_0',
              cable_name='cable_1', plug_name='sc_tip', time_limit=180)


def patch_perception(monkeypatch, clock, visible, truth_checks):
    def estimate(policy, parsed, mode, phase, prior_axis_base_link=None, task=None):
        policy._board_pose = policy._board_pose or BoardPose(np.eye(3), np.array([0., 0., .5]), 0.)
        if not visible(clock):
            return TargetEstimate(visible=False, confidence=0., detection_source='sc_heatmap',
                                  rejection_reason='sc_heatmap_no_candidate')
        return TargetEstimate(visible=True, confidence=.9, detection_source='sc_heatmap',
                              source_camera='center+left', port_pos_base_link=np.array([.01, 0., .2]))

    def truth(policy, task):
        truth_checks.append(policy._outcome)
        return np.array([.01, 0., .201]), np.eye(3), None

    monkeypatch.setattr(policy_perception, 'estimate_target', estimate)
    monkeypatch.setattr(policy_perception, 'lookup_plug_tip_in_base', lambda *_: None)
    monkeypatch.setattr(policy_perception, 'synchronized_camera_names', lambda *_: ['center'])
    monkeypatch.setattr(ground_truth, 'port_pos_from_gt_tf', truth)
    monkeypatch.setattr(module, 'HOLD_SECONDS', 1.0)


def test_live_loop_locks_holds_and_never_commands_insertion_motion(tmp_path, monkeypatch):
    clock, truth_checks, commands = Clock(), [], []
    patch_perception(monkeypatch, clock, lambda _: True, truth_checks)
    monkeypatch.setattr(policy_motion, 'send_motion', lambda policy, move, command: commands.append(command))
    policy = make_policy(tmp_path, clock)
    assert policy.insert_cable(task(), lambda: object(), Mock(), Mock()) is False
    trace = json.loads(next(tmp_path.glob('*.json')).read_text())
    assert trace['outcome'] == 'acquired'
    events = [e['event'] for e in trace['events']]
    assert events == ['lock', 'finish']
    # REQUIRED_LOCK_COUNT consecutive visible find_target cycles precede the lock.
    assert trace['rows'][trace['events'][0]['row']]['phase'] == 'coarse_align'
    from validate_acquisition import summarize_trace
    summary = summarize_trace(trace)
    assert summary['lock_window']['cycles'] == 10
    assert all(r['visible'] for r in trace['rows'][:trace['events'][0]['row']])
    assert summary['first_hold']['final_used_error_mm'] == pytest.approx(1.)
    # Every command is a hold at the measured TCP pose: no approach motion.
    assert commands and all(c.pose.position.x == 0. and c.pose.position.z == 0. for c in commands)
    # Ground truth is read once, after the outcome was decided.
    assert truth_checks == ['acquired']
    assert trace['ground_truth']['face_position'] == [.01, 0., .201]
    assert trace['images'] == {'lock': f"{trace['episode_id']}-lock.npz",
                               'final': f"{trace['episode_id']}-final.npz"}


def test_sustained_loss_returns_to_search_like_coarse_align(tmp_path, monkeypatch):
    clock, truth_checks, commands = Clock(), [], []
    start = clock.ns
    # Visible long enough to lock, then lost for good.
    patch_perception(monkeypatch, clock, lambda c: c.ns-start < 700_000_000, truth_checks)
    monkeypatch.setattr(policy_motion, 'send_motion', lambda policy, move, command: commands.append(command))
    monkeypatch.setattr(module, 'BUDGET_SECONDS', 3.0)
    policy = make_policy(tmp_path, clock)
    policy.insert_cable(task(), lambda: object(), Mock(), Mock())
    trace = json.loads(next(tmp_path.glob('*.json')).read_text())
    events = [e['event'] for e in trace['events']]
    assert events[:2] == ['lock', 'lock_lost']
    assert trace['outcome'] == 'validation_budget_exceeded'
    lost = trace['events'][1]['row']
    assert trace['rows'][lost+1]['phase'] == 'find_target'
    assert any(f['message'] == 'perception stale, reacquiring target' for f in trace['feedback'])


def test_trace_is_written_when_the_live_loop_is_cancelled(tmp_path, monkeypatch):
    clock, truth_checks = Clock(), []
    patch_perception(monkeypatch, clock, lambda _: True, truth_checks)
    monkeypatch.setattr(policy_motion, 'send_motion', lambda *_: None)
    policy = make_policy(tmp_path, clock)
    calls = iter([None, None, RuntimeError('cancel')])

    def check():
        result = next(calls)
        if isinstance(result, Exception):
            raise result
    policy._parent_node = NS(check_policy_execution=check)
    with pytest.raises(RuntimeError, match='cancel'):
        policy.insert_cable(task(), lambda: object(), Mock(), Mock())
    trace = json.loads(next(tmp_path.glob('*.json')).read_text())
    assert trace['outcome'] == 'error' and 'cancel' in trace['error']
    assert truth_checks == ['error']


def test_validation_mode_is_explicit(monkeypatch, tmp_path):
    monkeypatch.delenv('AIC_ACQUISITION_VALIDATION', raising=False)
    with pytest.raises(RuntimeError, match='explicit'):
        module.ValidateAcquisition(None)
    monkeypatch.setenv('AIC_ACQUISITION_VALIDATION', '1')
    monkeypatch.delenv('AIC_ACQUISITION_TRACE_DIR', raising=False)
    with pytest.raises(RuntimeError, match='TRACE_DIR'):
        module.ValidateAcquisition(None)


def test_summary_reports_lock_window_hold_stability_and_errors():
    from validate_acquisition import summarize_trace, aggregate
    rows = []
    for i in range(14):
        phase = 'find_target' if i < 10 else 'coarse_align'
        point = [0., 0., .2+(.001 if i == 12 else 0.)]
        rows.append({'t': i*.1, 'wall': i*.1, 'phase': phase, 'board_registered': True, 'visible': True,
                     'filter': 'accepted', 'estimator_stage': 'accepted', 'rejection': '',
                     'source_cameras': 'center+left', 'raw_position': point, 'used_position': [0., 0., .2],
                     'rail_distance_bound': .3, 'used_tcp_distance': .2, 'status': None,
                     'exposures_ns': {'center': i//2}})
    trace = {'rows': rows, 'events': [{'event': 'lock', 't': 1., 'row': 10},
                                      {'event': 'finish', 't': 1.3, 'outcome': 'acquired'}],
             'feedback': [], 'outcome': 'acquired', 'scene_id': 's', 'task': {},
             'constants': {'REQUIRED_LOCK_COUNT': 10}, 'ground_truth': {'face_position': [0., .003, .2]}}
    summary = summarize_trace(trace)
    assert summary['time_to_first_lock_s'] == 1.
    assert summary['lock_window']['cycles'] == 10
    assert summary['lock_window']['distinct_center_exposures'] == 5
    assert summary['lock_window']['used_error_at_lock_mm'] == pytest.approx(3.)
    assert summary['first_hold']['raw_spread_mm'] == pytest.approx(1.)
    assert summary['first_hold']['final_used_error_mm'] == pytest.approx(3.)
    assert summary['first_hold']['final_used_error_components_mm'] == pytest.approx({'lateral': 3., 'vertical': 0.})
    assert aggregate([summary])['outcomes'] == {'acquired': 1}


def test_validation_compose_mounts_frozen_checkpoint_and_disables_capture(tmp_path):
    from validate_acquisition import validation_compose
    run = tmp_path/'scene-0000'
    config = validation_compose(run, 'eval', 'model', 'scene-sha256:x', False,
                                {'AIC_SC_PORT_DETECTOR_PATH': 'sc.pt'})
    policy = config['services']['model']
    assert 'AIC_CAPTURE_DIR' not in policy['environment']
    assert policy['environment']['AIC_SC_PORT_DETECTOR_PATH'] == '/checkpoints/sc.pt'
    assert {'type': 'bind', 'source': str(tmp_path/'checkpoints'), 'target': '/checkpoints',
            'read_only': True} in policy['volumes']
    assert 'policy:=aic_model.ValidateAcquisition' in policy['command']
    captured = validation_compose(run, 'eval', 'model', 'scene-sha256:x', False, {}, capture=True)['services']['model']
    assert captured['environment']['AIC_CAPTURE_DIR'] == '/captures'
    assert {'type': 'bind', 'source': str(run/'captures'), 'target': '/captures'} in captured['volumes']


def test_live_loop_suspends_framing_on_evidence_and_ends_it_at_lock(tmp_path, monkeypatch):
    import aic_model.policy_rail_view as rail
    clock, truth_checks, calls = Clock(), [], []
    start = clock.ns
    patch_perception(monkeypatch, clock, lambda c: c.ns-start >= 300_000_000, truth_checks)
    monkeypatch.setattr(policy_motion, 'send_motion', lambda *_: None)

    def step(policy, parsed, task, now, force_limit, abort_limit=None):
        calls.append('step')
        if policy._rail_view_start is None:
            policy._rail_view_plan, policy._rail_view_start = 'plan', now
        return 'framing_requested_rail', parsed.tcp_pose
    original_suspend = rail.suspend_rail_view
    monkeypatch.setattr(rail, 'rail_view_step', step)
    monkeypatch.setattr(rail, 'suspend_rail_view', lambda p, now: (calls.append('suspend'), original_suspend(p, now)))
    policy = make_policy(tmp_path, clock)
    policy.insert_cable(task(), lambda: object(), Mock(), Mock())
    trace = json.loads(next(tmp_path.glob('*.json')).read_text())
    assert trace['outcome'] == 'acquired'
    assert calls[0] == 'step' and 'suspend' in calls and calls.index('suspend') > calls.index('step')
    assert policy._rail_view_plan is None and policy._rail_view_start is None


def test_holds_keep_one_latched_setpoint_while_the_measured_pose_sags(tmp_path, monkeypatch):
    clock, truth_checks, commands = Clock(), [], []
    start = clock.ns
    patch_perception(monkeypatch, clock, lambda _: True, truth_checks)
    monkeypatch.setattr(policy_motion, 'send_motion', lambda policy, move, command: commands.append(command))
    policy = make_policy(tmp_path, clock)

    def sagging(_msg):
        parsed = observation(clock)
        parsed.tcp_pose.position.z = -.003*(clock.ns-start)/1e9   # 3 mm/s, as measured live
        return parsed
    policy._parse_observation = sagging
    policy.insert_cable(task(), lambda: object(), Mock(), Mock())
    assert len(commands) > 20
    assert {c.pose.position.z for c in commands} == {commands[0].pose.position.z}


def test_framing_motion_releases_the_latch_so_the_next_hold_starts_where_framing_ended(tmp_path, monkeypatch):
    import aic_model.policy_rail_view as rail
    clock, truth_checks, commands = Clock(), [], []
    start = clock.ns
    patch_perception(monkeypatch, clock, lambda c: c.ns-start >= 300_000_000, truth_checks)
    real_send = policy_motion.send_motion
    monkeypatch.setattr(policy_motion, 'send_motion', lambda policy, move, command: (
        commands.append(command), real_send(policy, lambda **_: None, command)))
    monkeypatch.setattr(policy_motion, 'motion_update_from_command', lambda *_: None)
    monkeypatch.setattr(rail, 'rail_view_step', lambda policy, parsed, *_: ('framing_requested_rail', parsed.tcp_pose))
    policy = make_policy(tmp_path, clock)

    def sagging(_msg):
        parsed = observation(clock)
        parsed.tcp_pose.position.z = -.003*(clock.ns-start)/1e9
        return parsed
    policy._parse_observation = sagging
    policy.insert_cable(task(), lambda: object(), Mock(), Mock())
    heights = [c.pose.position.z for c in commands]
    framing = [z for z in heights if z > heights[-1]]
    assert framing, 'framing commands precede the hold'
    # Holds latch the first post-framing measurement, not the episode start.
    assert heights[-1] < 0. and set(heights[len(framing):]) == {heights[-1]}


@pytest.mark.parametrize('existing, failures, builds', [(True, 0, 0), (False, 2, 3)])
def test_verified_build_reuses_matching_tag_and_retries_registry_failures(tmp_path, monkeypatch, existing, failures, builds):
    import subprocess
    import benchmark as collect   # the verified build is shared by collection, validation and benchmarks
    manifest = {'source_sha256': 'a'*64, 'source_files': {'aic_model/aic_model/policy.py': 'h'}}
    attempts = []

    def run(args, **kwargs):
        return NS(returncode=0 if existing else 1)

    def command(args, **kwargs):
        attempts.append(args)
        if len(attempts) <= failures:
            raise subprocess.CalledProcessError(1, args)
    monkeypatch.setattr(collect.subprocess, 'run', run)
    monkeypatch.setattr(collect, 'command', command)
    monkeypatch.setattr(collect.time, 'sleep', lambda _: None)
    monkeypatch.setattr(collect, 'image_info', lambda tag: {'id': 'sha256:x', 'reference': tag})
    monkeypatch.setattr(collect, 'probe_image', lambda *a, **k: NS(stdout=json.dumps({'policy.py': 'h'})))
    info = collect.build_verified_image(tmp_path, manifest, 'aic-collection-model:', 10, labels=('a=b',))
    assert len(attempts) == builds and info['reused_existing_tag'] is existing
    assert all(args[2:4] == ['--label', 'a=b'] for args in attempts)


def run_board_search(tmp_path, monkeypatch, visible_after_s):
    """Marker found after 2 s of search; the target is accepted from `visible_after_s`."""
    clock, truth_checks, commands, statuses = Clock(), [], [], []
    start = clock.ns
    patch_perception(monkeypatch, clock, lambda _: True, truth_checks)
    board = BoardPose(np.eye(3), np.array([0., 0., .5]), 0.)

    def estimate(policy, parsed, mode, phase, prior_axis_base_link=None, task=None):
        if clock.ns-start < 2_000_000_000:           # marker found after 2 s of search
            return TargetEstimate(visible=False, confidence=0., detection_source='board_registration',
                                  rejection_reason='board_marker_unregistered')
        policy._board_pose = board
        if clock.ns-start < visible_after_s*1e9:
            return TargetEstimate(visible=False, confidence=0., detection_source='sfp_face',
                                  rejection_reason='sfp_heatmap_no_candidate')
        return TargetEstimate(visible=True, confidence=.9, detection_source='sc_heatmap',
                              source_camera='center+left', port_pos_base_link=np.array([.01, 0., .2]))
    monkeypatch.setattr(policy_perception, 'estimate_target', estimate)
    monkeypatch.setattr(policy_perception, 'camera_projection_matrix',
                        lambda *_: (np.eye(3), np.eye(3), np.zeros(3)))
    monkeypatch.setattr(policy_motion, 'send_motion', lambda policy, move, command: commands.append(command))
    policy = make_policy(tmp_path, clock)
    real_status = policy._send_status
    policy._send_status = lambda fb, state, message, now: (statuses.append(message), real_status(fb, state, message, now))
    policy.insert_cable(task(), lambda: object(), Mock(), Mock())
    outcome = json.loads(next(tmp_path.glob('*.json')).read_text())['outcome']
    return [c.pose.position.z for c in commands], commands, statuses, outcome


def test_board_search_is_retraced_to_the_start_view_before_acquisition(tmp_path, monkeypatch):
    z, commands, statuses, outcome = run_board_search(tmp_path, monkeypatch, visible_after_s=4.5)
    holds = [c for c in commands if c.is_hold]
    # Search withdraws along -camera Z, retraces at the same rate, then holds at the start pose.
    assert min(z) < -.05
    assert 'returning to start view' in statuses
    assert holds and all(c.pose.position.z == 0. and c.pose.position.y == 0. for c in holds)
    peak = z.index(min(z))
    assert all(b >= a-1e-9 for a, b in zip(z[peak:], z[peak+1:]))   # monotone return, no jump
    assert outcome == 'acquired'


def test_target_accepted_during_the_return_is_locked_where_it_was_seen(tmp_path, monkeypatch):
    # Qualification only guarantees the start view in one camera, sometimes at its
    # edge; a view the return passes through may serve both stereo cameras better.
    z, commands, statuses, outcome = run_board_search(tmp_path, monkeypatch, visible_after_s=2.)
    assert outcome == 'acquired'
    assert statuses.count('returning to start view') <= 1
    # No retrace is commanded: the last moving command is the deepest search pose.
    # (This harness's measured TCP stays at the origin, so holds latch there.)
    moving = [c.pose.position.z for c in commands if not c.is_hold]
    assert min(moving) < -.05 and moving[-1] <= min(moving)+.005


def test_windowed_lock_tolerates_flicker_but_not_inconsistent_positions():
    from aic_model.policy import InsertState, Policy
    from aic_model.policy_state import update_windowed_lock
    policy = NS(REQUIRED_LOCK_COUNT=10, LOCK_WINDOW_CYCLES=15, LOCK_SPREAD_M=.005)

    def run(pattern):
        state = InsertState()
        for index, value in enumerate(pattern):
            target = TargetEstimate(visible=value is not None, confidence=.9,
                                    port_pos_base_link=None if value is None else np.array(value))
            if update_windowed_lock(policy, state, target, None if value is None else np.array(value)):
                return index, state, target
        return None, state, None
    good = [0., 0., .2]
    # Every third cycle rejected: the consecutive rule never locks, the window does.
    flicker = [None if i % 3 == 2 else good for i in range(30)]
    index, state, target = run(flicker)
    assert index == 13 and state.target_lock_count == 10
    np.testing.assert_allclose(state.port_pos_smoothed, good)
    np.testing.assert_allclose(target.port_pos_base_link, good)
    # An accepted outlier 8 mm away blocks lock until it leaves the window.
    outlier = [good]*5 + [[.008, 0., .2]] + [good]*20
    assert run(outlier)[0] == 20
    assert Policy.LOCK_WINDOW_CYCLES == 15 and Policy.LOCK_SPREAD_M == .005


def test_cycles_without_evaluable_cameras_are_not_misses_until_the_gap_is_long():
    from aic_model.policy import InsertState, Policy
    from aic_model.policy_perception import no_evaluable_camera
    from aic_model.policy_state import counts_as_miss
    assert no_evaluable_camera({'stage': 'insufficient_cameras',
                                'cameras': {'center': 'missing_geometry_or_image', 'left': 'missing_geometry_or_image'}})
    assert not no_evaluable_camera({'stage': 'insufficient_cameras',
                                    'cameras': {'center': 'accepted', 'left': 'missing_geometry_or_image'}})
    assert not no_evaluable_camera(None)
    state, policy = InsertState(), NS(PERCEPTION_GAP_MAX_SEC=Policy.PERCEPTION_GAP_MAX_SEC)
    gap = TargetEstimate(visible=False, confidence=0., rejection_reason='no_evaluable_camera')
    reject = TargetEstimate(visible=False, confidence=0., rejection_reason='sc_heatmap_no_candidate')
    assert [counts_as_miss(policy, state, gap, t) for t in (0., .2, .9)] == [False, False, False]
    assert counts_as_miss(policy, state, gap, 1.2)
    assert counts_as_miss(policy, state, reject, 1.3) and state.evidence_gap_start is None


def test_brief_camera_geometry_gap_after_lock_does_not_drop_the_lock(tmp_path, monkeypatch):
    clock, truth_checks = Clock(), []
    start = clock.ns
    patch_perception(monkeypatch, clock, lambda _: True, truth_checks)
    inner = policy_perception.estimate_target

    def estimate(policy, parsed, mode, phase, prior_axis_base_link=None, task=None):
        target = inner(policy, parsed, mode, phase, prior_axis_base_link, task)
        if phase == 'coarse_align' and clock.ns-start < 1_000_000_000:   # 0.5 s TF gap right after lock
            return TargetEstimate(visible=False, confidence=0., detection_source='sc_heatmap',
                                  rejection_reason='no_evaluable_camera')
        return target
    monkeypatch.setattr(policy_perception, 'estimate_target', estimate)
    monkeypatch.setattr(policy_motion, 'send_motion', lambda *_: None)
    policy = make_policy(tmp_path, clock)
    policy.insert_cable(task(), lambda: object(), Mock(), Mock())
    trace = json.loads(next(tmp_path.glob('*.json')).read_text())
    assert [e['event'] for e in trace['events']] == ['lock', 'finish'] and trace['outcome'] == 'acquired'


def test_benchmark_compose_mounts_candidates_and_can_disable_capture(tmp_path):
    from benchmark import compose_config
    config = compose_config(tmp_path, 'eval', 'model', False, {'sfp': 'sfp-x.pt', 'sc': 'sc-y.pt'}, capture=False)
    policy = config['services']['model']
    assert 'AIC_CAPTURE_DIR' not in policy['environment']
    assert policy['environment']['AIC_SFP_DETECTOR_PATH'] == '/checkpoints/sfp-x.pt'
    assert policy['environment']['AIC_SC_PORT_DETECTOR_PATH'] == '/checkpoints/sc-y.pt'
    assert policy['volumes'] == [{'type': 'bind', 'source': str(tmp_path/'checkpoints'), 'target': '/checkpoints',
                                  'read_only': True}]
    assert policy['environment']['AIC_ENABLE_ACL'] == 'true' and 'ground_truth:=false' in config['services']['eval']['command']
    default = compose_config(tmp_path, 'eval', 'model')
    assert default['services']['model']['environment']['AIC_CAPTURE_DIR'] == '/captures'


def test_static_lock_is_held_only_in_approach_phases_and_cleared_on_reacquisition():
    from aic_model.policy import InsertState
    from aic_model.policy_state import hold_static_target, set_phase
    state = InsertState(phase='coarse_align')
    seen = TargetEstimate(visible=True, confidence=.9, source_camera='left', port_pos_base_link=np.array([.1, .2, .3]),
                          port_rot_base_link=np.eye(3))
    missed = TargetEstimate(visible=False, confidence=0., rejection_reason='sfp_heatmap_no_candidate')
    assert hold_static_target(state, seen) is seen
    seen.port_pos_base_link[0] = 9.                       # the held copy is independent of later mutation
    held = hold_static_target(state, missed)
    assert held.visible and held.rejection_reason == 'held_static_lock'
    np.testing.assert_allclose(held.port_pos_base_link, [.1, .2, .3])
    held.port_pos_base_link[0] = 7.
    np.testing.assert_allclose(hold_static_target(state, missed).port_pos_base_link, [.1, .2, .3])
    state.phase = 'settle'
    assert hold_static_target(state, missed).rejection_reason == 'held_static_lock'
    for phase in ('recover', 'find_target'):
        state.phase = phase
        assert hold_static_target(state, missed) is missed
    set_phase(state, 'find_target', 1.)
    state.phase = 'coarse_align'
    assert state.held_target is None and hold_static_target(state, missed) is missed


def test_reacquisition_hold_returns_to_the_lock_pose(tmp_path):
    clock = Clock()
    policy = make_policy(tmp_path, clock)
    policy.begin_episode()
    lock_pose = observation(clock).tcp_pose
    lock_pose.position.z = .25
    command = policy._hold_command(observation(clock), lock_pose)
    assert command.is_hold and command.pose.position.z == .25
    assert policy._hold_command(observation(clock)).pose.position.z == .25    # latched until other motion


def test_orientation_is_board_anchored_only_once_the_board_is_registered():
    from aic_model.board_registration import BoardPose
    from aic_model.policy_perception import board_anchored_orientation
    assert board_anchored_orientation(NS(_board_pose=BoardPose(np.eye(3), np.zeros(3), 0.)))
    assert not board_anchored_orientation(NS(_board_pose=None))


def test_a_hung_scene_is_recorded_and_the_run_continues(tmp_path, monkeypatch):
    """A simulator crash after the episode hung one scene and ended the run."""
    import subprocess
    import yaml
    import validate_acquisition as validation
    destination = tmp_path/'run'

    def prepare(config, destination):
        destination.mkdir()
        (destination/'config.yaml').write_text(yaml.safe_dump({'trials': {'a': {'scene': 1}, 'b': {'scene': 2}}}))
        (destination/'manifest.json').write_text('{}')

    class Compose:
        def __init__(self, run, prefix, timeout):
            self.run = run
            (run/'compose.log').write_text('')

        def __enter__(self):
            if self.run.name == 'scene-0000':
                raise subprocess.TimeoutExpired('docker', 1)
            return self

        def __exit__(self, *_):
            return False
    monkeypatch.setattr(validation, 'prepare', prepare)
    monkeypatch.setattr(validation, 'image_info', lambda image: {'id': 'sha256:e'})
    monkeypatch.setattr(validation, 'build_verified_image', lambda *args: {'id': 'sha256:m'})
    monkeypatch.setattr(validation, 'validation_compose', lambda *args: {})
    monkeypatch.setattr(validation, 'isolated_compose', Compose)
    monkeypatch.setattr(validation, 'load_traces', lambda run: [({}, {'scene': run.name})])
    monkeypatch.setattr(validation, 'aggregate', lambda summaries: {'episodes': len(summaries)})
    manifest = validation.validate(None, destination, 'eval', {})
    assert [r['status'] for r in manifest['scene_runs']] == ['timed_out', 'traced']
    assert manifest['scene_runs'][0]['harness_timeout'] is True
    assert manifest['status'] == 'incomplete' and manifest['summary'] == {'episodes': 2}
