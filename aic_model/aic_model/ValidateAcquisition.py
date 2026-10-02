"""Acquisition-only validation of the live policy; insertion is never attempted.

The unmodified `Policy.insert_cable` loop performs board registration, rail
framing, estimation, filtering and target lock. After lock this subclass keeps
the live policy's latched hold instead of starting `coarse_align` motion, keeps the live
estimator and filter running in that phase, and mirrors `coarse_align`'s
sustained-loss return to `find_target`. Ground truth is read once, after the
episode, for error reporting only.
"""
import json
import math
import os
from pathlib import Path
import time

import numpy as np
from rclpy.time import Time

from .policy import Policy
from . import ground_truth
from . import policy_motion as motion
from . import policy_perception as perception
from . import policy_state as state

HOLD_SECONDS = 5.0
BUDGET_SECONDS = 90.0


def _vector(value):
    return None if value is None else [float(v) for v in np.asarray(value, dtype=float)]


def _json_safe(value):
    """Nonfinite numbers become null so the trace stays strict JSON."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (bool, str)) or value is None:
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    return str(value)


class ValidateAcquisition(Policy):
    def __init__(self, parent_node):
        if os.environ.get('AIC_ACQUISITION_VALIDATION') != '1':
            raise RuntimeError('ValidateAcquisition requires explicit acquisition validation mode')
        trace_dir = os.environ.get('AIC_ACQUISITION_TRACE_DIR', '').strip()
        if not trace_dir:
            raise RuntimeError('Acquisition validation requires AIC_ACQUISITION_TRACE_DIR')
        super().__init__(parent_node)
        # Capture perturbs control-loop timing; only an explicit diagnostic run may enable it.
        self._diagnostic_capture = os.environ.get('AIC_ACQUISITION_DIAGNOSTIC_CAPTURE') == '1'
        if self._capture_dir is not None and not self._diagnostic_capture:
            raise RuntimeError('Disable dataset capture; it perturbs control-loop timing')
        self._trace_dir = Path(trace_dir)

    # ── Live-loop hooks ──────────────────────────────────────────────────────
    def _trace_cycle(self, task, parsed, target, raw_position, filter_status, insert_state, now):
        diagnostics = (getattr(self, '_last_sc_pose_diagnostics', None)
                       or getattr(self, '_last_sfp_pose_diagnostics', None))
        tcp = state.tcp_position_vector(parsed)
        used = target.port_pos_base_link if target.visible else None
        board = self._board_pose
        rail_bound = None
        if board is not None:
            from .rail_view import rail_distance_bound
            try:
                rail_bound = rail_distance_bound(board, task.target_module_name, tcp)
            except ValueError:
                pass
        exposures = {}
        for name, header in parsed.image_header_map.items():
            if header is not None and parsed.image_map.get(name) is not None:
                exposures[name] = Time.from_msg(header.stamp).nanoseconds
        self._rows.append({
            't': now-self._validation_start, 'wall': time.monotonic()-self._wall_start,
            'phase': insert_state.phase, 'board_registered': board is not None,
            'visible': bool(target.visible), 'source': target.detection_source,
            'rejection': target.rejection_reason, 'filter': filter_status,
            'estimator_stage': diagnostics.get('stage') if diagnostics else None,
            'estimator_cameras': dict(diagnostics.get('cameras', {})) if diagnostics else None,
            'estimator_decoder': dict(diagnostics.get('decoder', {})) if diagnostics else None,
            'geometry_residual_m': diagnostics.get('geometry_residual_m') if diagnostics else None,
            'target_confidence': dict(diagnostics.get('target_confidence', {})) if diagnostics else None,
            'source_cameras': target.source_camera, 'confidence': float(target.confidence),
            'raw_position': _vector(raw_position), 'used_position': _vector(used),
            'tcp_position': _vector(tcp), 'force': float(parsed.force_mag),
            'used_tcp_distance': None if used is None else float(np.linalg.norm(np.asarray(used)-tcp)),
            'rail_distance_bound': rail_bound,
            'exposures_ns': exposures, 'synchronized_cameras': perception.synchronized_camera_names(self, parsed),
            'lock_count': int(insert_state.target_lock_count), 'status': None,
        })
        self._last_parsed = parsed

    def _send_status(self, send_feedback, insert_state, message, now):
        if self._rows:
            self._rows[-1]['status'] = message
        super()._send_status(send_feedback, insert_state, message, now)

    def _acquisition_step(self, task, parsed, target, insert_state, move_robot, send_feedback, now):
        elapsed = now-self._validation_start
        if insert_state.phase in {'initialize', 'find_target'}:
            if elapsed > BUDGET_SECONDS:
                return self._finish('validation_budget_exceeded', now)
            return None
        if insert_state.phase != 'coarse_align':
            return self._finish(f'unexpected_phase:{insert_state.phase}', now)
        if self._hold_start is None:
            self._hold_start = now
            self._events.append({'event': 'lock', 't': elapsed, 'row': len(self._rows)-1})
            if self._lock_parsed is None:
                self._lock_parsed = parsed
        if not target.visible or target.port_pos_base_link is None:
            if state.counts_as_miss(self, insert_state, target, now):
                insert_state.perception_stale_count += 1
            if insert_state.perception_stale_count >= self.PERCEPTION_STALE_CYCLES:
                # Same sustained-loss handling as live coarse_align.
                if now-insert_state.phase_start_time >= 1.0:
                    insert_state.align_retry_count += 1
                insert_state.target_lock_count = 0
                state.set_phase(insert_state, 'find_target', now)
                send_feedback('perception stale, reacquiring target')
                self._events.append({'event': 'lock_lost', 't': elapsed, 'row': len(self._rows)-1})
                self._hold_start = None
        else:
            insert_state.perception_stale_count = 0
            insert_state.evidence_gap_start = None
        motion.send_motion(self, move_robot, self._hold_command(parsed))
        if self._hold_start is not None and now-self._hold_start >= HOLD_SECONDS:
            return self._finish('acquired', now)
        return 'continue'

    # ── Episode wrapper ──────────────────────────────────────────────────────
    def _finish(self, outcome, now):
        self._outcome = outcome
        self._events.append({'event': 'finish', 't': now-self._validation_start, 'outcome': outcome})
        return False

    def insert_cable(self, task, get_observation, move_robot, send_feedback):
        self._rows, self._events, self._feedback = [], [], []
        self._outcome = None
        self._hold_start = None
        self._lock_parsed = self._last_parsed = None
        self._validation_start = self.time_now().nanoseconds/1e9
        self._wall_start = time.monotonic()

        def feedback(message):
            self._feedback.append({'t': self.time_now().nanoseconds/1e9-self._validation_start,
                                   'message': message})
            send_feedback(message)

        error = None
        try:
            super().insert_cable(task, get_observation, move_robot, feedback)
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
            raise
        finally:
            self._write_trace(task, error)
        send_feedback(f'Acquisition validation: {self._outcome}; no insertion attempted')
        return False

    def _write_trace(self, task, error):
        if self._outcome is None:
            aborts = [f['message'] for f in self._feedback if f['message'].startswith('abort:')]
            self._outcome = 'policy_abort' if aborts else ('error' if error else 'policy_returned')
        # Post-hoc ground truth only: every control decision above has finished.
        truth = ground_truth.port_pos_from_gt_tf(self, task)
        episode = self._capture_episode_id
        self._trace_dir.mkdir(parents=True, exist_ok=True)
        images = {}
        for label, parsed in (('lock', self._lock_parsed), ('final', self._last_parsed)):
            if parsed is None:
                continue
            path = self._trace_dir/f'{episode}-{label}.npz'
            np.savez_compressed(path, **{f'{name}_image': image for name, image in parsed.image_map.items()
                                         if image is not None})
            images[label] = path.name
        record = {
            'schema_version': 1, 'episode_id': episode, 'outcome': self._outcome, 'error': error,
            'diagnostic_capture': getattr(self, '_diagnostic_capture', False),
            'scene_id': os.environ.get('AIC_ACQUISITION_SCENE_ID', ''),
            'task': {'plug_type': task.plug_type, 'port_type': task.port_type,
                     'port_name': task.port_name, 'target_module_name': task.target_module_name,
                     'time_limit': int(task.time_limit)},
            'mode': self._task_mode(task),
            'detectors': {'sfp': os.environ.get('AIC_SFP_DETECTOR_PATH', ''),
                          'sc': os.environ.get('AIC_SC_PORT_DETECTOR_PATH', '')},
            'hold_seconds': HOLD_SECONDS, 'budget_seconds': BUDGET_SECONDS,
            'constants': {key: getattr(self, key, None) for key in (
                'CONTROL_DT', 'REQUIRED_LOCK_COUNT', 'LOCK_WINDOW_CYCLES', 'LOCK_SPREAD_M',
                'PERCEPTION_STALE_CYCLES', 'SEARCH_TIMEOUT_SEC',
                'PLAUSIBLE_PORT_DISTANCE_MIN_M', 'PLAUSIBLE_PORT_DISTANCE_MAX_M', 'PORT_POS_EMA_ALPHA',
                'SC_PORT_SPATIAL_LOCK_SEARCH_M', 'SC_PORT_HEATMAP_MIN_CONFIDENCE',
                'SC_PORT_HEATMAP_MAX_GEOMETRY_RESIDUAL_M')},
            'events': self._events, 'feedback': self._feedback, 'rows': self._rows,
            'images': images,
            'ground_truth': {'face_position': _vector(truth[0]) if truth else None,
                             'source': 'latest TF after the episode; post-hoc scoring only'},
        }
        path = self._trace_dir/f'{episode}.json'
        path.write_text(json.dumps(_json_safe(record), indent=1, allow_nan=False))
        self.get_logger().info(f'Acquisition validation {self._outcome}: {path}')
