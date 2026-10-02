"""Training-only rail-view collection; view planning never reads ground truth."""
from collections import deque
import time

from .CaptureInitialViews import CaptureInitialViews
from .policy import TargetEstimate, InsertState
from . import policy_perception as perception
from . import policy_motion as motion
from . import policy_state as state
from .policy_rail_view import rail_view_step


class CaptureRailViews(CaptureInitialViews):
    def insert_cable(self, task, get_observation, move_robot, send_feedback):
        self.begin_episode()
        start = self.time_now().nanoseconds/1e9
        wall_deadline = time.monotonic()+60.
        pending = deque()
        last_sample = -float('inf')
        framed_since = None
        status = 'waiting_for_board'
        force_state = InsertState(startup_time=start)

        def write_sample(item):
            now, parsed, phase = item
            self._wait_for_capture_tf(task,parsed,wall_deadline)
            self._maybe_capture_sample(task,parsed,
                TargetEstimate(visible=False,confidence=0.,detection_source='rail_view_collection',
                               rejection_reason=phase),InsertState(phase='find_target'),now)

        while time.monotonic() < wall_deadline:
            self._parent_node.check_policy_execution()
            now = self.time_now().nanoseconds/1e9
            if now-start > 20.:
                break
            observation = get_observation()
            parsed = self._parse_observation(observation) if observation is not None else None
            if parsed is None:
                time.sleep(.02)
                continue
            # The local simulator carries ~20 N resting gripper/cable load.
            # Calibrate while stationary, using the policy's bounded calm window.
            state.update_startup_force_baseline(self,parsed,force_state,now)
            if now-start < self.FORCE_ABORT_GRACE_SEC:
                if parsed.force_mag > self.FORCE_BASELINE_CALM_N:
                    status = 'rail_view_contact'
                    break
                time.sleep(.02)
                continue
            if not force_state.startup_force_samples:
                status = 'rail_view_contact'
                break
            # As in Policy: a load above the recover threshold holds the plan;
            # rail_view_step ends it on a sustained load or above the abort one.
            force_limit = state.force_recover_threshold(self,force_state)
            abort_limit = state.force_abort_threshold(self,force_state)
            board = perception.register_board(self,parsed)
            if board is None:
                if now-start > 8.:
                    status = 'board_unavailable'
                    break
            else:
                status, pose = rail_view_step(self,parsed,task,now,force_limit=force_limit,abort_limit=abort_limit)
                # Collection has no perception to wait for at an over-range pose.
                if pose is None or status == 'rail_view_distance_limit':
                    break
                motion.send_motion(self,move_robot,motion.build_pose_command(pose))
                framed_since = (now if framed_since is None else framed_since) if status=='framed' else None
            if now-last_sample >= .5:
                pending.append((now,parsed,status))
                last_sample = now
            # Flush older exposures while they remain in TF's history. GT is
            # consulted only here, after the geometry-only motion decision.
            while pending and now-pending[0][0] >= 1.5:
                write_sample(pending.popleft())
            if framed_since is not None and now-framed_since >= 3.:
                break
            time.sleep(.02)
        for sample in pending:
            self._parent_node.check_policy_execution()
            write_sample(sample)
        self.get_logger().info(f'Rail view collection ended: {status}; captures={self._capture_counter}')
        send_feedback(f'Rail view collection: {status}; no insertion attempted')
        return False
