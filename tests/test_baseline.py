import importlib.util
from pathlib import Path
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('benchmark', ROOT / 'scripts/benchmark.py')
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)
from aic_model.aic_model import AicModel, PolicyCancelled, PolicyExecution
from aic_model.policy import Policy
from aic_control_interfaces.msg import MotionUpdate, TargetMode
from rclpy.action import GoalResponse


def score(t3=75, message='Cable insertion successful.', valid=1):
    return {'total': valid + t3, 'trial_1': {
        'tier_1': {'score': valid}, 'tier_2': {'score': 0, 'categories': {}},
        'tier_3': {'score': t3, 'message': message}}}


@pytest.mark.parametrize('t3,message,outcome', [
    (75, 'Cable insertion successful.', 'full_insertion'),
    (45, 'Partial insertion detected with distance of 0.01m.', 'partial_insertion'),
    (-12, 'Cable insertion failed. Incorrect Port.', 'wrong_port'),
    (20, 'No insertion detected. Final plug port distance: 0.01m.', 'proximity'),
    (0, 'Task not completed.', 'execution_or_scoring_failure'),
])
def test_official_outcomes(t3, message, outcome):
    result = benchmark.summarize(score(t3, message), ['trial_1'])
    assert result['complete']
    assert result['trials'][0]['outcome'] == outcome


def test_missing_trial_is_not_success():
    result = benchmark.summarize(score(), ['trial_1', 'trial_2'])
    assert not result['complete']
    assert result['full_insertion_rate'] == 0.5


def test_model_invalid_overrides_insertion():
    assert benchmark.summarize(score(valid=0), ['trial_1'])['trials'][0]['outcome'] == 'model_invalid'


def test_total_mismatch_is_rejected():
    data = score(); data['total'] = 100
    with pytest.raises(ValueError):
        benchmark.summarize(data, ['trial_1'])


def test_compose_keeps_policy_isolated(tmp_path):
    config = benchmark.compose_config(tmp_path, 'sha256:eval', 'sha256:model', True)
    assert not any(key.startswith(('AIC_DEBUG', 'AIC_EXPERIMENTAL'))
                   for key in config['services']['model']['environment'])
    for service in config['services'].values():
        assert service['environment']['AIC_ENABLE_ACL'] == 'true'
        assert service['gpus'] == 'all'
    mounts = config['services']['model']['volumes']
    assert all(m['target'] not in ['/results', '/benchmark/config.yaml'] for m in mounts)
    assert 'ground_truth:=false' in config['services']['eval']['command']


def wrapper():
    # Exercise real wrapper methods without starting a ROS graph or simulator.
    node = AicModel.__new__(AicModel)
    node._command_lock = threading.RLock()
    node._worker_local = threading.local()
    goal = SimpleNamespace(is_active=True, is_cancel_requested=False, request=SimpleNamespace(task=None))
    node._execution = PolicyExecution(goal)
    node._worker_local.execution = node._execution
    node.is_active = True
    node._goal_reserved = False
    node._observation_msg = None
    node._target_mode = TargetMode.MODE_CARTESIAN
    node.motion_update_pub = Mock()
    node.get_logger = Mock(return_value=Mock())
    return node


def test_cancel_blocks_late_commands():
    node = wrapper()
    node.move_robot(MotionUpdate())
    node.stop_policy()
    with pytest.raises(PolicyCancelled):
        node.move_robot(MotionUpdate())
    assert node.motion_update_pub.publish.call_count == 1


def test_previous_execution_cannot_command_new_goal():
    node = wrapper()
    node._execution = PolicyExecution(node._execution.goal)
    with pytest.raises(PolicyCancelled):
        node.move_robot(MotionUpdate())
    node.motion_update_pub.publish.assert_not_called()


def test_concurrent_goal_reservation():
    node = wrapper()
    assert node.insert_cable_goal_callback(None) == GoalResponse.ACCEPT
    assert node.insert_cable_goal_callback(None) == GoalResponse.REJECT


def test_policy_exception_is_failure():
    node = wrapper()
    policy = Mock()
    policy.insert_cable.side_effect = RuntimeError('test failure')
    node.action_thread_func(node._execution, policy)
    assert not node._execution.result
    assert node._execution.error == 'RuntimeError: test failure'
    assert node._execution.stop.is_set()


def test_cancel_interrupts_paused_simulation_sleep():
    node = wrapper()
    from rclpy.time import Time
    node.get_clock = Mock(return_value=SimpleNamespace(now=lambda: Time(seconds=1)))
    policy = Policy.__new__(Policy)
    policy._parent_node = node
    finished = threading.Event()
    run = node._execution
    def worker():
        node._worker_local.execution = run
        try:
            policy.sleep_for(100)
        except PolicyCancelled:
            finished.set()
    run.thread = threading.Thread(target=worker)
    run.thread.start()
    node.stop_policy()
    assert finished.wait(1)
    assert not run.thread.is_alive()


def test_uncooperative_worker_prevents_new_goal():
    node = wrapper()
    done = threading.Event()
    node._execution.thread = threading.Thread(target=lambda: done.wait(1))
    node._execution.thread.start()
    try:
        assert node.insert_cable_goal_callback(None) == GoalResponse.REJECT
    finally:
        done.set()
        node._execution.thread.join()


def test_missing_checkpoints_fail_configuration(monkeypatch):
    for key in ('AIC_SFP_DETECTOR_PATH', 'AIC_SC_PORT_DETECTOR_PATH', 'AIC_CAPTURE_DIR'):
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(RuntimeError, match='Both SFP and SC checkpoints'):
        Policy(Mock())


def test_no_scoring_subscription_in_policy():
    assert '/scoring/' not in (ROOT / 'aic_model/aic_model/policy.py').read_text()


def test_runtime_modules_never_read_ground_truth():
    # Capture (training collection) and ValidateAcquisition (post-hoc) are the only GT users.
    package = ROOT / 'aic_model/aic_model'
    privileged = {'ground_truth.py', 'policy_capture.py', 'ValidateAcquisition.py',
                  'CaptureInitialViews.py', 'CaptureRailViews.py', 'aic_model.py'}
    for path in package.glob('*.py'):
        if path.name in privileged:
            continue
        text = path.read_text()
        assert 'ground_truth' not in text and 'task_board/' not in text, path.name


def test_importing_the_policy_does_not_load_privileged_modules():
    # policy_capture imports ground_truth only when a capture sample is taken.
    import subprocess, sys
    code = (f'import sys; sys.path.insert(0, {str(ROOT / "aic_model")!r}); import aic_model.policy; '
            "print(sorted(m for m in sys.modules if m.split('.')[-1] in "
            "{'ground_truth', 'ValidateAcquisition', 'OracleGrasp', 'CaptureInitialViews', 'CaptureRailViews'}))")
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, check=True, cwd=ROOT)
    assert out.stdout.strip() == '[]'


def test_wait_set_shutdown_race_does_not_hide_live_ros_errors():
    from aic_model.aic_model import spin_until_shutdown, rclpy_implementation
    executor = SimpleNamespace(spin=Mock(side_effect=rclpy_implementation.RCLError('wait set')))
    spin_until_shutdown(executor, SimpleNamespace(ok=lambda: False))
    with pytest.raises(rclpy_implementation.RCLError):
        spin_until_shutdown(executor, SimpleNamespace(ok=lambda: True))


def test_capture_metadata_does_not_require_removed_scoring_state():
    from aic_model.policy import InsertState
    metadata = Policy.__new__(Policy)._insertion_metadata(InsertState())
    assert 'scoring_event' not in metadata


def test_failed_preflight_is_recorded_without_docker(tmp_path, monkeypatch):
    import json
    run = tmp_path / 'run'
    (run / 'source/scripts').mkdir(parents=True)
    script = Path(benchmark.__file__)
    (run / 'source/scripts/benchmark.py').write_bytes(script.read_bytes())
    (run / 'config.yaml').write_text('changed')
    manifest = {'status': 'prepared', 'config_sha256': 'original',
                'source_files': {'scripts/benchmark.py': benchmark.sha256(script)}}
    (run / 'manifest.json').write_text(json.dumps(manifest))
    docker = Mock(side_effect=AssertionError('Docker should not run'))
    monkeypatch.setattr(benchmark, 'image_info', docker)
    with pytest.raises(ValueError, match='configuration changed'):
        benchmark.execute(run, 'unused', False, 10)
    saved = json.loads((run / 'manifest.json').read_text())
    assert saved['status'] == 'failed'
    assert 'configuration changed' in saved['error']
    docker.assert_not_called()


def test_extra_frozen_source_file_is_rejected(tmp_path, monkeypatch):
    import json
    run = tmp_path / 'run'
    (run / 'source/scripts').mkdir(parents=True)
    script = Path(benchmark.__file__)
    (run / 'source/scripts/benchmark.py').write_bytes(script.read_bytes())
    (run / 'source/unexpected.py').write_text('print(1)')
    manifest = {'status': 'prepared', 'source_files': {'scripts/benchmark.py': benchmark.sha256(script)}}
    (run / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='inventory changed'):
        benchmark.execute(run, 'unused', False, 10)



def test_joint_stop_preserves_valid_dimensions_and_removes_feedforward():
    from aic_control_interfaces.msg import JointMotionUpdate, TrajectoryGenerationMode
    node = wrapper()
    node._target_mode = TargetMode.MODE_JOINT
    node.joint_motion_update_pub = Mock()
    command = JointMotionUpdate()
    command.target_stiffness = [90.0] * 6
    command.target_damping = [45.0] * 6
    command.target_state.velocities = [0.1] * 6
    command.target_feedforward_torque = [2.0] * 6
    command.trajectory_generation_mode.mode = TrajectoryGenerationMode.MODE_VELOCITY
    node.move_robot(joint_motion_update=command)
    node.stop_policy()
    hold = node.joint_motion_update_pub.publish.call_args.args[0]
    assert list(hold.target_state.velocities) == [0.0] * 6
    assert list(hold.target_feedforward_torque) == [0.0] * 6
    assert len(hold.target_damping) == len(hold.target_stiffness) == 6
    assert list(command.target_state.velocities) == [0.1] * 6
