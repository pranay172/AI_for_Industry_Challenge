"""Training-only, stationary initial-view collector; never an insertion policy."""
import os
import time

from rclpy.time import Time

from .policy import Policy, TargetEstimate, InsertState


class CaptureInitialViews(Policy):
    def __init__(self, parent_node):
        if os.environ.get('AIC_TRAINING_COLLECTION') != '1':
            raise RuntimeError('CaptureInitialViews requires explicit training collection mode')
        if not os.environ.get('AIC_CAPTURE_SCENE_ID') or not os.environ.get('AIC_CAPTURE_DIR'):
            raise RuntimeError('Collection requires a scene ID and capture directory')
        super().__init__(parent_node)

    def _wait_for_capture_tf(self, task, parsed, wall_deadline):
        """Let delayed GT publishers catch up to this frozen RGB exposure."""
        frame = f'task_board/{task.target_module_name}/{task.port_name}_link'
        deadline = min(wall_deadline, time.monotonic() + 5.)
        headers = list(parsed.image_header_map.values())
        while headers and time.monotonic() < deadline:
            self._parent_node.check_policy_execution()
            if all(header.stamp.sec or header.stamp.nanosec for header in headers) and all(
                self._parent_node._tf_buffer.can_transform(
                    header.frame_id, frame, Time.from_msg(header.stamp)) for header in headers
            ):
                return
            time.sleep(.01)
        for header in headers:
            ready, reason = self._parent_node._tf_buffer.can_transform(
                header.frame_id, frame, Time.from_msg(header.stamp), return_debug_tuple=True)
            if not ready:
                self.get_logger().warning(f'Capture GT unavailable at exposure: {reason}')

    def insert_cable(self, task, get_observation, move_robot, send_feedback):
        self.begin_episode()
        start = self.time_now().nanoseconds
        wall_deadline = time.monotonic() + 30.
        pending = []
        last_sample_ns = -1_000_000_000
        send_feedback('Training data collection: stationary initial views')
        while time.monotonic() < wall_deadline:
            self._parent_node.check_policy_execution()
            now = self.time_now().nanoseconds
            if now - start >= 3_000_000_000:
                break
            observation = get_observation()
            parsed = self._parse_observation(observation) if observation is not None else None
            if parsed is not None and now - last_sample_ns >= 250_000_000:
                pending.append((now, parsed))
                last_sample_ns = now
            # Wall-time wait also bounds collection when simulation time is paused.
            time.sleep(.05)
        # GT is published more slowly than RGB. Preserve exposure snapshots and
        # flush after acquisition so the next GT update can bracket each image.
        for now, parsed in pending:
            self._parent_node.check_policy_execution()
            self._wait_for_capture_tf(task, parsed, wall_deadline)
            self._maybe_capture_sample(task, parsed,
                TargetEstimate(visible=False, confidence=0., detection_source='training_collection'),
                InsertState(phase='find_target'), now / 1e9)
        send_feedback(f'Training collection finished: {self._capture_counter} captures; no insertion attempted')
        return False
