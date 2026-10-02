#
#  Copyright (C) 2026 Intrinsic Innovation LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
"""Callback signatures and the dataclasses shared by the policy modules."""

import numpy as np

from aic_control_interfaces.msg import JointMotionUpdate, MotionUpdate, TrajectoryGenerationMode
from aic_model_interfaces.msg import Observation
from aic_task_interfaces.msg import Task
from dataclasses import dataclass, field
from geometry_msgs.msg import Pose, Twist
from typing import Callable, Optional, Protocol

GetObservationCallback = Callable[[], Observation]


class MoveRobotCallback(Protocol):
    """Move the robot using either Cartesian or joint-space commands.

    This function is called by a policy to request robot motion. Either
    Cartesian or joint-space commands can be sent, but not both at the
    same time. One of the following must be set:
     - motion_update: cartesian motion commands
     - joint_motion_update: joint-space motion commands

    The MotionUpdate message contains a request to the Cartesian-space
    admittance controller. The details of this message are described
    in its message definition:
      https://github.com/intrinsic-dev/aic/blob/main/aic_interfaces/aic_control_interfaces/msg/MotionUpdate.msg

    The JointMotionUpdate message contains commands to the joint-space
    controller. The details of this message are described in its message definition:
      https://github.com/intrinsic-dev/aic/blob/main/aic_interfaces/aic_control_interfaces/msg/JointMotionUpdate.msg

    As a convenience, a reasonable set of parameters is populated in a MotionUpdate
    message by the create_motion_update(pose) function in the Policy class.
    """

    def __call__(
        self,
        motion_update: MotionUpdate = None,
        joint_motion_update: JointMotionUpdate = None,
    ) -> None: ...


SendFeedbackCallback = Callable[[str], None]


@dataclass
class ParsedObservation:
    image_map: dict
    camera_info_map: dict
    force_vec: np.ndarray
    torque_vec: np.ndarray
    tcp_pose: Pose
    tcp_velocity: Twist
    tcp_error: np.ndarray
    joint_positions: np.ndarray
    force_mag: float
    lateral_force_mag: float
    speed_mag: float
    image_header_map: dict = field(default_factory=dict)


@dataclass
class TargetEstimate:
    visible: bool
    confidence: float
    x_error: float = 0.0
    y_error: float = 0.0
    presence_prob: float = 0.0
    centering_score: float = 0.0
    landmark_score: float = 0.0
    bbox_width_px: float = 0.0
    bbox_height_px: float = 0.0
    z_distance_m: float = 0.0
    # Retained for capture compatibility; current online SFP path does not use it.
    corners_px: Optional[np.ndarray] = None
    detection_source: str = ""
    source_camera: str = ""
    rejection_reason: str = ""
    port_pos_base_link: Optional[np.ndarray] = None
    # 3×3 rotation of port in base_link; None when position-only estimate.
    port_rot_base_link: Optional[np.ndarray] = None


@dataclass
class MotionCommand:
    twist: Optional[Twist] = None
    pose: Optional[Pose] = None
    frame_id: str = "gripper/tcp"
    stiffness: tuple = (90.0, 90.0, 90.0, 50.0, 50.0, 50.0)
    damping: tuple = (50.0, 50.0, 50.0, 20.0, 20.0, 20.0)
    mode: int = TrajectoryGenerationMode.MODE_VELOCITY
    feedforward_wrench: Optional[tuple] = None  # (fx, fy, fz, tx, ty, tz) in frame_id
    is_hold: bool = False  # station-keeping setpoint; see Policy._hold_command


@dataclass
class InsertGeometry:
    valid: bool = False
    xy_m: float = float("inf")
    depth_m: float = -float("inf")
    residual_xy: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))


@dataclass
class InsertContactState:
    valid: bool = False
    state: str = "invalid"
    xy_m: float = float("inf")
    depth_m: float = -float("inf")
    travel_m: float = 0.0
    force_n: float = 0.0
    lateral_force_n: float = 0.0
    # Positive when the measured force is below the startup baseline — i.e.
    # the port is supporting some of the plug/cable weight. Strong signal of
    # genuine engagement that does not rely on lateral force or axial travel.
    force_drop_n: float = 0.0
    residual_xy: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))
    escaped: bool = False


@dataclass
class InsertState:
    phase: str = "initialize"
    phase_start_time: float = 0.0
    startup_time: float = 0.0  # wall time when insert_cable() started
    retry_count: int = 0
    target_lock_count: int = 0
    # Raw accepted positions (None = rejected) for the windowed experimental lock.
    lock_history: list = field(default_factory=list)
    evidence_gap_start: Optional[float] = None
    # Experimental perception: last accepted estimate, acted on while approach
    # views miss the static target, and the TCP pose where lock was obtained.
    held_target: Optional["TargetEstimate"] = None
    reacquire_pose: Optional[Pose] = None
    close_yaw_rotations: list = field(default_factory=list)
    close_yaw_wait_start: Optional[float] = None
    yaw_dither_start: Optional[float] = None
    yaw_dither_reference: Optional[np.ndarray] = None
    # Experimental perception: spiral/yaw search at the port face (face_search.py).
    face_search: Optional[object] = None
    face_search_count: int = 0
    seated: bool = False
    # (port position, port quat) of the last lock dropped by a re-acquisition,
    # where an attempt that ends without a lock parks the plug.
    last_lock: Optional[tuple] = None
    last_progress_time: float = 0.0
    last_feedback_time: float = 0.0
    insertion_start_time: float = 0.0
    force_abort_count: int = 0  # consecutive cycles above the force abort threshold
    align_retry_count: int = (
        0  # coarse_align→find_target transitions (after ≥1 s dwell)
    )
    coarse_align_success_count: int = 0
    startup_force_samples: list = field(default_factory=list)
    startup_force_baseline: float = 0.0
    # Captured on every entry into insert phase (per-attempt axial-travel
    # reference). Cleared by set_phase on insert entry; recaptured by the
    # pre_insert -> insert transition from current TCP pose.
    insertion_start_position: Optional[np.ndarray] = None
    insertion_axis: Optional[np.ndarray] = None
    last_target_x_error: float = 0.0
    last_target_y_error: float = 0.0
    perception_stale_count: int = 0  # consecutive not-visible cycles in close phases
    # EMA-smoothed port position in base_link; reset when target is lost/reacquired
    port_pos_smoothed: Optional[np.ndarray] = None
    # Locked port orientation carried into pre_insert for TCP alignment.
    locked_port_quat: Optional[object] = None
    # Stall detection for coarse_align: track best pose_err and when it last improved
    coarse_align_best_pose_err: float = float("inf")
    coarse_align_last_improve_time: float = 0.0
    # EMA-smoothed port insertion axis; reset each coarse_align entry.
    port_insertion_axis_smoothed: Optional[np.ndarray] = None
    # EMA-smoothed port quaternion; reset each coarse_align entry.
    port_quat_smoothed: Optional[object] = None
    # Last N raw axes considered for SEEDING the EMA. Cleared on coarse_align entry.
    axis_seeding_buffer: list = field(default_factory=list)
    # Diagnostic-only: previous-cycle values for delta logging in coarse_align
    diag_prev_port_pos: Optional[np.ndarray] = None
    diag_prev_raw_axis: Optional[np.ndarray] = None
    diag_prev_camera_set: Optional[str] = None
    diag_axis_align_history: list = field(default_factory=list)
    diag_port_pos_step_history: list = field(default_factory=list)
    diag_raw_axis_step_deg_history: list = field(default_factory=list)
    # Flip rejections net of agreeing updates; past 10 the EMA re-seeds.
    flip_rejections: int = 0
    # Diagnostic counters since last 1-Hz print
    diag_flip_rejected_since_print: int = 0
    diag_ema_updates_since_print: int = 0
    # Fine-center XY integrator: accumulates residual between perceived plug-tip
    # and perceived port position during pre_insert. Compensates for steady-state
    # impedance offset (commanded TCP ≠ actual TCP at finite stiffness) so the
    # plug actually lands centered before descent. Cleared on pre_insert exit.
    pre_insert_xy_correction: np.ndarray = field(
        default_factory=lambda: np.zeros(2, dtype=float)
    )
    # Best plug-port residual this pre_insert entry and when it last improved.
    pre_insert_best_residual_m: float = float("inf")
    pre_insert_best_residual_time: Optional[float] = None
    # This insert attempt began from a pre-insert stall handover, not the gate.
    stall_handover: bool = False
    face_recenter_count: int = 0
    pre_insert_xy_hold_count: int = 0
    last_pre_insert_orientation_diag_time: float = 0.0
    last_pre_insert_gate_diag_time: float = 0.0
    last_pre_insert_pose_err_m: float = float("nan")
    last_pre_insert_axis_error_rad: float = float("nan")
    last_pre_insert_orientation_error_rad: float = float("nan")
    last_pre_insert_plug_axis_error_rad: float = float("nan")
    last_pre_insert_plug_orientation_error_rad: float = float("nan")
    last_pre_insert_goal: Optional[Pose] = None
    last_seating_diag_time: float = 0.0
    insert_substate: str = ""
    insert_substate_start_time: float = 0.0
    engaged_start_time: float = 0.0
    max_insert_depth_m: float = -float("inf")
    max_insert_travel_m: float = 0.0
    settle_deep_confirm_count: int = 0
    settle_travel_plateau_time: float = 0.0
    face_slide_count: int = 0
    last_insert_contact: InsertContactState = field(default_factory=InsertContactState)
    last_insert_motion_mode: str = ""
    last_insert_recover_reason: str = ""
    face_contact_servo_start_time: float = 0.0
    face_contact_servo_last_improve_time: float = 0.0
    face_contact_servo_best_xy_m: float = float("inf")
    corner_reference_start_time: float = 0.0
    corner_reference_center_start_time: float = 0.0
    corner_reference_position: Optional[np.ndarray] = None
    corner_reference_mode: str = ""


@dataclass
class CycleContext:
    """One control cycle's inputs, handed to the phase step methods."""
    task: Task
    mode: str
    mode_cfg: dict
    insert_state: InsertState
    parsed_obs: ParsedObservation
    target: TargetEstimate
    raw_position: Optional[np.ndarray]
    plug_pos_base: Optional[np.ndarray]
    plug_quat_base: Optional[object]
    now_wall: float
    move_robot: MoveRobotCallback
    send_feedback: SendFeedbackCallback
