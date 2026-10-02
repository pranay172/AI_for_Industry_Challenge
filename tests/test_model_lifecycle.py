"""ROS integration regression; run with an isolated Zenoh router (see docs/solution/evaluation.md)."""
import os
from pathlib import Path
import sys
import threading
import time

import pytest

pytestmark = pytest.mark.skipif(os.environ.get('AIC_ROS_INTEGRATION') != '1', reason='requires isolated ROS graph')


def test_sigterm_stops_model_main_cleanly(tmp_path):
    import subprocess
    code = '''
from rclpy.executors import MultiThreadedExecutor
from aic_model.aic_model import main
original_spin = MultiThreadedExecutor.spin
def announce_spin(self):
    print("SPIN_READY", flush=True)
    return original_spin(self)
MultiThreadedExecutor.spin = announce_spin
main(args=["--ros-args", "-p", "policy:=aic_model.policy"])
'''
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]/'aic_model'))
    log = tmp_path/'model.log'
    with log.open('w') as stream:
        process = subprocess.Popen([sys.executable, '-c', code], env=env,
                                   stdout=stream, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 15
            while 'SPIN_READY' not in log.read_text() and process.poll() is None and time.monotonic() < deadline:
                time.sleep(.02)
            assert 'SPIN_READY' in log.read_text(), log.read_text()
            process.terminate()
            assert process.wait(timeout=10) == 0, log.read_text()
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)


def test_cancel_with_paused_clock_and_next_goal():
    import rclpy
    from rclpy.action import ActionClient
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.lifecycle import TransitionCallbackReturn
    from action_msgs.msg import GoalStatus
    from aic_task_interfaces.action import InsertCable
    from aic_model.aic_model import AicModel
    from aic_model.policy import Policy
    from aic_control_interfaces.msg import MotionUpdate
    from aic_control_interfaces.srv import ChangeTargetMode

    def wait(future, seconds=8):
        deadline = time.monotonic() + seconds
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert future.done(), 'ROS operation timed out'
        return future.result()

    class WaitingPolicy(Policy):
        def __init__(self, parent):
            self._parent_node = parent
            self.started = threading.Event()
        def insert_cable(self, task, get_observation, move_robot, send_feedback):
            move_robot(MotionUpdate())
            self.started.set()
            self.sleep_for(60)  # No /clock is published: cancellation must interrupt.
            return True

    rclpy.init(args=['--ros-args', '-p', 'policy:=aic_model.policy', '-p', 'use_sim_time:=true'])
    node = AicModel()
    node._policy_class = WaitingPolicy
    client_node = rclpy.create_node('baseline_test_client')
    mode_requests = []
    def set_mode(request, response):
        mode_requests.append(request.target_mode.mode)
        response.success = True
        return response
    client_node.create_service(ChangeTargetMode, '/aic_controller/change_target_mode', set_mode)
    client = ActionClient(client_node, InsertCable, 'insert_cable')
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node); executor.add_node(client_node)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    try:
        assert client.wait_for_server(timeout_sec=10)
        assert not wait(client.send_goal_async(InsertCable.Goal())).accepted
        assert node.trigger_configure() == TransitionCallbackReturn.SUCCESS
        assert not wait(client.send_goal_async(InsertCable.Goal())).accepted
        assert node.trigger_activate() == TransitionCallbackReturn.SUCCESS
        for _ in range(2):
            node._policy.started.clear()
            handle = wait(client.send_goal_async(InsertCable.Goal()))
            assert handle.accepted
            assert node._policy.started.wait(5)
            cancel = wait(handle.cancel_goal_async())
            assert cancel.goals_canceling
            result = wait(handle.get_result_async())
            assert result.status == GoalStatus.STATUS_CANCELED
            assert not result.result.success
            assert not node.worker_running()
        node._policy.started.clear()
        handle = wait(client.send_goal_async(InsertCable.Goal()))
        assert handle.accepted and node._policy.started.wait(5)
        assert node.trigger_deactivate() == TransitionCallbackReturn.SUCCESS
        result = wait(handle.get_result_async())
        assert result.status == GoalStatus.STATUS_ABORTED
        assert not node.worker_running()
        assert mode_requests, "Policy must switch controller mode through the service"
        assert node.trigger_cleanup() == TransitionCallbackReturn.SUCCESS
        assert node.trigger_shutdown() == TransitionCallbackReturn.SUCCESS
        assert node.motion_update_pub is None
    finally:
        node.stop_policy()
        executor.shutdown(timeout_sec=3)
        thread.join(timeout=3)
        client.destroy()
        client_node.destroy_node(); node.destroy_node()
        rclpy.shutdown()
