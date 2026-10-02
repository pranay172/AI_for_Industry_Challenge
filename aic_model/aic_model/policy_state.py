#
#  Copyright (C) 2026 Intrinsic Innovation LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#

"""State, phase, force-baseline and abort helpers for the policy.

Free functions take `policy` as the first arg when they need access to
policy state (constants, time_now, get_logger). Pure helpers don't.
"""

from typing import Optional

import numpy as np
from geometry_msgs.msg import Pose

from . import policy_geometry as _geom


def set_phase(insert_state, phase: str, now_wall: float) -> None:
    prev_phase = insert_state.phase
    insert_state.phase = phase
    insert_state.phase_start_time = now_wall
    insert_state.perception_stale_count = 0
    insert_state.face_search = None
    if phase == "find_target":
        held = insert_state.held_target
        if held is not None and held.port_pos_base_link is not None and insert_state.locked_port_quat is not None:
            insert_state.last_lock = (np.array(held.port_pos_base_link, dtype=float), insert_state.locked_port_quat)
        insert_state.lock_history.clear()
        insert_state.held_target = None
        insert_state.close_yaw_rotations.clear()
    if phase == "coarse_align":
        insert_state.close_yaw_wait_start = None
    if prev_phase == "pre_insert" and phase != "pre_insert":
        insert_state.pre_insert_xy_correction = np.zeros(2, dtype=float)
        insert_state.pre_insert_best_residual_m = float("inf")
        insert_state.pre_insert_best_residual_time = None
        insert_state.pre_insert_xy_hold_count = 0
        insert_state.last_pre_insert_orientation_diag_time = 0.0
        insert_state.last_pre_insert_pose_err_m = float("nan")
        insert_state.last_pre_insert_axis_error_rad = float("nan")
        insert_state.last_pre_insert_orientation_error_rad = float("nan")
        insert_state.last_pre_insert_plug_axis_error_rad = float("nan")
        insert_state.last_pre_insert_plug_orientation_error_rad = float("nan")
    if phase != "coarse_align":
        insert_state.coarse_align_success_count = 0
    if phase == "coarse_align":
        insert_state.face_recenter_count = 0
        insert_state.coarse_align_best_pose_err = float("inf")
        insert_state.coarse_align_last_improve_time = 0.0
        # Only reset orientation smoothing if coming from find_target (genuine re-acquisition).
        # Keep the seed across recovery cycles (e.g. recover -> coarse_align) to prevent
        # orientation jumps.
        if (
            prev_phase == "find_target"
            or insert_state.port_insertion_axis_smoothed is None
        ):
            insert_state.port_insertion_axis_smoothed = None
            insert_state.port_quat_smoothed = None
            insert_state.axis_seeding_buffer = []
            insert_state.flip_rejections = 0
            insert_state.diag_prev_raw_axis = None
            insert_state.diag_raw_axis_step_deg_history = []
            insert_state.diag_ema_updates_since_print = 0

        insert_state.diag_prev_port_pos = None
        insert_state.diag_prev_camera_set = None
        insert_state.diag_axis_align_history = []
        insert_state.diag_port_pos_step_history = []
        insert_state.diag_flip_rejected_since_print = 0
    if phase == "insert":
        insert_state.insertion_start_time = now_wall
        insert_state.insert_substate = "face_probe"
        insert_state.insert_substate_start_time = now_wall
        insert_state.engaged_start_time = 0.0
        insert_state.max_insert_depth_m = -float("inf")
        insert_state.max_insert_travel_m = 0.0
        # Per-attempt axial-travel reference. Stale values from a previous
        # attempt (preserved across recover cycles) make travel_m unreliable
        # for face_probe gates; recapture from the current TCP pose.
        insert_state.insertion_start_position = None
        insert_state.insertion_axis = None
        insert_state.settle_deep_confirm_count = 0
        insert_state.settle_travel_plateau_time = 0.0
        insert_state.last_insert_motion_mode = ""
        insert_state.last_insert_recover_reason = ""
        insert_state.face_contact_servo_start_time = 0.0
        insert_state.face_contact_servo_last_improve_time = 0.0
        insert_state.face_contact_servo_best_xy_m = float("inf")
        insert_state.corner_reference_start_time = 0.0
        insert_state.corner_reference_center_start_time = 0.0
        insert_state.corner_reference_position = None
        insert_state.corner_reference_mode = ""
        if hasattr(insert_state, "last_insert_contact"):
            insert_state.last_insert_contact = type(insert_state.last_insert_contact)()
    elif prev_phase == "insert" and phase == "settle":
        insert_state.insert_substate = "settle_confirm"
        insert_state.insert_substate_start_time = now_wall
        insert_state.settle_deep_confirm_count = 0
        if insert_state.engaged_start_time <= 0.0:
            insert_state.engaged_start_time = now_wall
        insert_state.last_insert_motion_mode = "settle_confirm"
        insert_state.face_contact_servo_start_time = 0.0
        insert_state.face_contact_servo_last_improve_time = 0.0
        insert_state.face_contact_servo_best_xy_m = float("inf")
        insert_state.corner_reference_start_time = 0.0
        insert_state.corner_reference_center_start_time = 0.0
        insert_state.corner_reference_position = None
        insert_state.corner_reference_mode = ""
    elif prev_phase == "settle" and phase != "settle":
        insert_state.settle_deep_confirm_count = 0
    elif prev_phase == "insert" and phase != "insert":
        insert_state.insert_substate = ""
        insert_state.insert_substate_start_time = 0.0
        insert_state.engaged_start_time = 0.0
        insert_state.last_insert_motion_mode = ""
        insert_state.face_contact_servo_start_time = 0.0
        insert_state.face_contact_servo_last_improve_time = 0.0
        insert_state.face_contact_servo_best_xy_m = float("inf")
        insert_state.corner_reference_start_time = 0.0
        insert_state.corner_reference_center_start_time = 0.0
        insert_state.corner_reference_position = None
        insert_state.corner_reference_mode = ""
    if phase == "recover":
        # Keep locked orientation for better alignment during recovery
        insert_state.face_recenter_count = 0
        pass
    if phase == "find_target":
        # Genuine target loss: reset smoothed state
        insert_state.insertion_start_position = None
        insert_state.insertion_axis = None
        insert_state.port_pos_smoothed = None
        insert_state.port_insertion_axis_smoothed = None
        insert_state.port_quat_smoothed = None
        insert_state.axis_seeding_buffer = []
        insert_state.locked_port_quat = None
        insert_state.face_recenter_count = 0
        insert_state.face_slide_count = 0
        if hasattr(insert_state, "last_insert_contact"):
            insert_state.last_insert_contact = type(insert_state.last_insert_contact)()


def tcp_position_vector(parsed_obs) -> np.ndarray:
    return np.array(
        [
            parsed_obs.tcp_pose.position.x,
            parsed_obs.tcp_pose.position.y,
            parsed_obs.tcp_pose.position.z,
        ],
        dtype=float,
    )


def tcp_approach_axis_base(parsed_obs) -> np.ndarray:
    """TCP +Z in base_link (gripper approach axis on UR5e Hand-E)."""
    rotation = _geom.quaternion_to_matrix(parsed_obs.tcp_pose.orientation)
    axis = rotation[:, 2]
    norm = np.linalg.norm(axis)
    if norm < 1e-6:
        return np.array([0.0, 0.0, -1.0], dtype=float)
    return axis / norm


def axial_travel_m(parsed_obs, insert_state) -> float:
    """Signed travel along the captured insertion axis (not Euclidean distance).
    Lateral spiral motion does not contribute."""
    if (
        insert_state.insertion_start_position is None
        or insert_state.insertion_axis is None
    ):
        return 0.0
    delta = tcp_position_vector(parsed_obs) - insert_state.insertion_start_position
    return float(np.dot(delta, insert_state.insertion_axis))


def phase_time_budget(policy, phase: str) -> float:
    """Minimum remaining seconds needed for this phase to have any chance of success."""
    return policy.PHASE_TIME_BUDGET.get(phase, 5.0)


def pose_error_m(parsed_obs, pose: Pose) -> float:
    target = np.array([pose.position.x, pose.position.y, pose.position.z], dtype=float)
    return float(np.linalg.norm(tcp_position_vector(parsed_obs) - target))


def rot_error_mag(parsed_obs) -> float:
    """Norm of the RPY rotation error from the controller state (rad)."""
    if parsed_obs.tcp_error.size < 6:
        return 0.0
    return float(np.linalg.norm(parsed_obs.tcp_error[3:]))


def update_startup_force_baseline(
    policy, parsed_obs, insert_state, now_wall: float
) -> None:
    if now_wall - insert_state.startup_time > policy.FORCE_ABORT_GRACE_SEC:
        return
    # Only sample calm readings so accidental early contact doesn't inflate
    # the baseline and raise thresholds for the whole trial.
    if parsed_obs.force_mag > policy.FORCE_BASELINE_CALM_N:
        return
    insert_state.startup_force_samples.append(parsed_obs.force_mag)
    insert_state.startup_force_baseline = float(
        np.median(insert_state.startup_force_samples)
    )


def force_baseline(policy, insert_state) -> float:
    if len(insert_state.startup_force_samples) > 0:
        return insert_state.startup_force_baseline
    return policy.FORCE_BASELINE_FALLBACK_N


def force_recover_threshold(policy, insert_state) -> float:
    base = force_baseline(policy, insert_state)
    margin = policy.FORCE_RECOVER_MARGIN_N
    mode = getattr(policy, "_current_mode", None)
    if mode is not None and mode in policy.MODE_CONFIG:
        margin = float(policy.MODE_CONFIG[mode].get("force_recover_margin_n", margin))
    return base + margin


def force_abort_threshold(policy, insert_state) -> float:
    base = force_baseline(policy, insert_state)
    margin = policy.FORCE_ABORT_MARGIN_N
    mode = getattr(policy, "_current_mode", None)
    if mode is not None and mode in policy.MODE_CONFIG:
        margin = float(policy.MODE_CONFIG[mode].get("force_abort_margin_n", margin))
    return base + margin


def abort_reason(
    policy, parsed_obs, insert_state, deadline, now_wall: float
) -> Optional[str]:
    if policy.time_now() >= deadline:
        return "time_limit"
    remaining_sec = (deadline.nanoseconds - policy.time_now().nanoseconds) / 1e9
    if remaining_sec < phase_time_budget(policy, insert_state.phase):
        return f"deadline_phase_{insert_state.phase}"
    abort_threshold = force_abort_threshold(policy, insert_state)
    contact_phase = insert_state.phase in {
        "pre_insert",
        "insert",
        "recover",
        "settle",
    }
    if (
        contact_phase
        and now_wall - insert_state.startup_time >= policy.FORCE_ABORT_GRACE_SEC
        and parsed_obs.force_mag >= abort_threshold
    ):
        insert_state.force_abort_count += 1
        if insert_state.force_abort_count >= policy.FORCE_ABORT_SUSTAINED_CYCLES:
            policy.get_logger().warn(
                "Aborting insertion: excess_force "
                f"(phase={insert_state.phase}, force={parsed_obs.force_mag:.2f} N, "
                f"baseline={force_baseline(policy, insert_state):.2f} N, "
                f"threshold={abort_threshold:.2f} N, "
                f"count={insert_state.force_abort_count})"
            )
            return "excess_force"
    else:
        # Decay by 1 instead of hard-resetting so that noisy readings
        # alternating near the threshold do not indefinitely block abort.
        insert_state.force_abort_count = max(0, insert_state.force_abort_count - 1)
    if insert_state.retry_count > policy.MAX_RETRIES:
        return "too_many_retries"
    if insert_state.align_retry_count > policy.MAX_ALIGN_RETRIES:
        return "too_many_align_retries"
    if (
        insert_state.phase == "find_target"
        and (now_wall - insert_state.phase_start_time) > policy.SEARCH_TIMEOUT_SEC
    ):
        return "target_not_found"
    return None


def update_windowed_lock(policy, insert_state, target, raw_position, now_wall: float = 0.) -> bool:
    """Record this cycle and report whether recent evidence is plentiful and consistent.

    At lock the smoothed position restarts from the window median, so evidence
    rejected by the spread test never seeds the controller's target.
    """
    history = insert_state.lock_history
    accepted = target.visible and raw_position is not None
    if accepted:
        insert_state.evidence_gap_start = None
    elif not counts_as_miss(policy, insert_state, target, now_wall):
        return False
    history.append(np.array(raw_position, dtype=float) if accepted else None)
    del history[:-policy.LOCK_WINDOW_CYCLES]
    points = [point for point in history if point is not None]
    insert_state.target_lock_count = len(points)
    if len(points) < policy.REQUIRED_LOCK_COUNT:
        return False
    points = np.asarray(points)
    median = np.median(points, axis=0)
    if float(np.max(np.linalg.norm(points-median, axis=1))) > policy.LOCK_SPREAD_M:
        return False
    insert_state.port_pos_smoothed = median.copy()
    if target.visible:
        target.port_pos_base_link = median.copy()
    return True


def counts_as_miss(policy, insert_state, target, now_wall: float) -> bool:
    """Whether a non-visible cycle is evidence against the target.

    Cycles without any evaluable camera are ignored until the gap exceeds
    PERCEPTION_GAP_MAX_SEC; after that, missing geometry counts as a miss.
    """
    if target.rejection_reason != "no_evaluable_camera":
        insert_state.evidence_gap_start = None
        return True
    if insert_state.evidence_gap_start is None:
        insert_state.evidence_gap_start = now_wall
    return now_wall-insert_state.evidence_gap_start > policy.PERCEPTION_GAP_MAX_SEC


# Phases that may act on a held lock: the board and port are static, and the
# approach views were not seen in training, so a missed detection there is not
# evidence that the target moved. Settle judges seating depth against the port,
# which the plug itself hides. Recovery re-centres in image space and never holds.
HELD_TARGET_PHASES = {"coarse_align", "pre_insert", "insert", "settle"}


def hold_static_target(insert_state, target):
    """Return the target to act on: the fresh estimate, or the last accepted one."""
    from dataclasses import replace
    if target.visible and target.port_pos_base_link is not None:
        insert_state.held_target = replace(
            target, port_pos_base_link=np.array(target.port_pos_base_link, dtype=float),
            port_rot_base_link=(None if target.port_rot_base_link is None
                                else np.array(target.port_rot_base_link, dtype=float)))
        return target
    if insert_state.held_target is not None and insert_state.phase in HELD_TARGET_PHASES:
        held = insert_state.held_target
        return replace(held, port_pos_base_link=held.port_pos_base_link.copy(),
                       port_rot_base_link=None if held.port_rot_base_link is None else held.port_rot_base_link.copy(),
                       rejection_reason="held_static_lock")
    return target


# An SFP cage admits only about 1 degree of plug yaw error: official trials jammed
# at 1.3-2.0 degrees and seated at 0.1 degrees (post-hoc bag analysis). Start-view
# card decodes carry 1-2 degrees of yaw noise, so the orientation lock uses the
# median yaw of estimates taken near the hover pose, where the cameras are ~3x closer.
CLOSE_YAW_MAX_TCP_DISTANCE_M = 0.14
CLOSE_YAW_MIN_SAMPLES = 3
CLOSE_YAW_MAX_SAMPLES = 15
CLOSE_YAW_MAX_WAIT_SEC = 4.0
CLOSE_YAW_MAX_CORRECTION_RAD = float(np.deg2rad(6.0))


def yaw_about_axis(reference, rotation):
    """Signed angle (rad) from reference's x axis to rotation's x axis, about reference z."""
    reference = np.asarray(reference, dtype=float); rotation = np.asarray(rotation, dtype=float)
    axis, x_ref = reference[:, 2], reference[:, 0]
    x = rotation[:, 0]-axis*float(np.dot(rotation[:, 0], axis))
    return float(np.arctan2(np.dot(np.cross(x_ref, x), axis), np.dot(x_ref, x)))


def record_close_yaw(insert_state, target, tcp_position):
    """Keep fresh (not held) oriented estimates observed from near the hover pose."""
    if (not target.visible or target.port_rot_base_link is None or target.port_pos_base_link is None
            or target.rejection_reason == "held_static_lock"):
        return False
    distance = float(np.linalg.norm(np.asarray(tcp_position, dtype=float)
                                    - np.asarray(target.port_pos_base_link, dtype=float)))
    if distance > CLOSE_YAW_MAX_TCP_DISTANCE_M:
        return False
    insert_state.close_yaw_rotations.append(np.array(target.port_rot_base_link, dtype=float))
    del insert_state.close_yaw_rotations[:-CLOSE_YAW_MAX_SAMPLES]
    return True


def close_yaw_ready(insert_state, now_wall):
    """Whether enough close-range samples exist, or waiting for them has timed out."""
    if len(insert_state.close_yaw_rotations) >= CLOSE_YAW_MIN_SAMPLES:
        return True
    if insert_state.close_yaw_wait_start is None:
        insert_state.close_yaw_wait_start = now_wall
    return now_wall-insert_state.close_yaw_wait_start >= CLOSE_YAW_MAX_WAIT_SEC


def close_range_quat(insert_state, smoothed_quat):
    """Smoothed port orientation with its yaw replaced by the close-range median.

    Returns (quaternion, correction_rad); the correction is None when there are
    too few samples or they disagree implausibly with the smoothed estimate.
    """
    if len(insert_state.close_yaw_rotations) < CLOSE_YAW_MIN_SAMPLES:
        return smoothed_quat, None
    reference = _geom.quaternion_to_matrix(smoothed_quat)
    correction = float(np.median([yaw_about_axis(reference, rotation)
                                  for rotation in insert_state.close_yaw_rotations]))
    if abs(correction) > CLOSE_YAW_MAX_CORRECTION_RAD:
        return smoothed_quat, None
    axis = reference[:, 2]
    k = np.array([[0., -axis[2], axis[1]], [axis[2], 0., -axis[0]], [-axis[1], axis[0], 0.]])
    turn = np.eye(3)+np.sin(correction)*k+(1.-np.cos(correction))*(k@k)
    return _geom.matrix_to_quaternion(turn@reference), correction


# The SFP cage tolerance (<0.7 degrees) is tighter than the plug's unobserved grasp
# yaw (the spec allows ~0.04 rad): with the port yaw within 0.01 degrees of truth
# an official trial still presented the plug 0.7-1.5 degrees off and jammed six
# times. While the plug is on the face, the TCP yaw therefore sweeps a sinusoid
# about the insertion axis so the aligned yaw is crossed under axial load. The
# duplex SC plug has the same unobserved grasp yaw, so it sweeps too.
YAW_DITHER_AMPLITUDE_RAD = float(np.deg2rad(2.0))
YAW_DITHER_PERIOD_SEC = 6.0
YAW_DITHER_GAIN = 3.0
YAW_DITHER_MAX_RATE_RAD_S = float(np.deg2rad(4.0))
# Past this axial travel the plug is inside the cage; stop turning it.
YAW_DITHER_MAX_TRAVEL_M = 0.008


def rotation_about_axis(reference, rotation, axis):
    """Angle (rad) of rotation@reference.T about the unit base-frame axis."""
    relative = np.asarray(rotation, dtype=float)@np.asarray(reference, dtype=float).T
    vee = .5*np.array([relative[2, 1]-relative[1, 2], relative[0, 2]-relative[2, 0], relative[1, 0]-relative[0, 1]])
    return float(np.arctan2(np.dot(vee, axis), .5*(np.trace(relative)-1.)))


def yaw_dither(insert_state, tcp_rotation, tcp_position, plug_position, axis, now_wall):
    """(rate, axis, lever) turning the plug about the insertion axis through its tip.

    The sweep is measured from this insert attempt's first TCP orientation; the
    lever (TCP minus plug tip) lets the command pivot about the tip, not the TCP.
    """
    if plug_position is None or axis is None:
        return None
    axis = np.asarray(axis, dtype=float)/max(float(np.linalg.norm(axis)), 1e-9)
    if insert_state.yaw_dither_start != insert_state.phase_start_time:
        insert_state.yaw_dither_start = insert_state.phase_start_time
        insert_state.yaw_dither_reference = np.array(tcp_rotation, dtype=float)
    lever = np.asarray(tcp_position, dtype=float)-np.asarray(plug_position, dtype=float)
    if insert_state.max_insert_travel_m > YAW_DITHER_MAX_TRAVEL_M:
        return 0., axis, lever
    elapsed = now_wall-insert_state.phase_start_time
    desired = YAW_DITHER_AMPLITUDE_RAD*np.sin(2.*np.pi*elapsed/YAW_DITHER_PERIOD_SEC)
    measured = rotation_about_axis(insert_state.yaw_dither_reference, tcp_rotation, axis)
    rate = float(np.clip(YAW_DITHER_GAIN*(desired-measured), -YAW_DITHER_MAX_RATE_RAD_S, YAW_DITHER_MAX_RATE_RAD_S))
    return rate, axis, lever
