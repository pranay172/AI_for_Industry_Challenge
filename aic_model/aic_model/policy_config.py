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
"""Policy constants and per-connector MODE_CONFIG."""

import math


class PolicyConfig:
    CONTROL_DT = 0.05
    # Maximum insertion-phase retries (pre_insert contact, insert force/stall,
    # settle instability).  Kept separate from alignment retries so that a noisy
    # detector during approach does not exhaust the insertion budget.
    MAX_RETRIES = 5
    # Maximum times coarse_align may drop the target and re-enter find_target
    # before giving up.  Only counted after ≥ 1 s dwell to filter brief flickers.
    MAX_ALIGN_RETRIES = 6
    # Give the policy enough time to reacquire a target during broad search.
    SEARCH_TIMEOUT_SEC = 60.0
    MAX_INSERT_SEC = 6.0
    # The scoring system tares the FTS at startup with the cable already held, so
    # the baseline is close to 0 N in evaluation.  The dynamic calibration in
    # _update_startup_force_baseline() learns the actual resting load in the first
    # FORCE_ABORT_GRACE_SEC; this fallback is used only if that window yields zero
    # samples (unlikely).  Setting it to 0 is safe: the margins already encode the
    # headroom, and a non-zero resting load will be captured by the calibration.
    FORCE_BASELINE_FALLBACK_N = 0.0
    FORCE_RECOVER_MARGIN_N = 6.0
    FORCE_ABORT_MARGIN_N = 14.0
    # Maximum force (N) accepted for baseline sampling.  Set high enough to pass
    # the resting cable load (~20 N when the FTS is not tared with cable attached)
    # so the baseline calibrates correctly in both local testing and official eval.
    # Only rejects genuine hard-crash transients (> 25 N at startup).
    FORCE_BASELINE_CALM_N = 25.0
    # Grace period after startup before force-abort is evaluated (sim physics transients).
    FORCE_ABORT_GRACE_SEC = 2.5
    # The engine homes the arm between trials (robot.home_joint_positions, the same
    # in qualification.yaml and sample_config.yaml), but it reactivates the
    # controller right after resetting the joints. Sometimes the controller reads
    # the pre-reset state and drives the arm back to the previous trial's final
    # pose (insertion-wide001-{sfp,sc}: 3 of 12 trials). Perception relies on the
    # qualification start view, so an episode that starts away from home returns
    # there first. Rise to at least home height, then move; stop on contact.
    HOME_TCP_POSITION = (-0.3717, 0.1944, 0.3200)
    HOME_TCP_QUAT_XYZW = (1.0, 0.0, 0.0, 0.0)
    HOME_START_TOL_M = 0.02
    HOME_START_TOL_DEG = 5.0
    HOME_RETURN_SPEED_MPS = 0.10
    HOME_RETURN_TURN_DPS = 30.0
    HOME_RETURN_SETTLE_M = 0.005
    HOME_RETURN_MAX_SEC = 12.0
    HOME_RETURN_FORCE_MARGIN_N = 14.0
    HOME_RETURN_LOAD_MARGIN_N = 30.0
    HOME_RETURN_LOAD_SEC = 0.5
    HOME_RETURN_STILL_MPS = 0.005
    HOME_RETURN_STILL_SEC = 0.5
    # The race also shifts the plug in the gripper: the engine attaches the cable
    # at the gripper's pose mid-race, 7-12 deg and several mm off the fixed grasp
    # (docs/solution/insertion.md). The wrist cameras measure the grasp at
    # the start (plug_pose.py). The measurement replaces `_PLUG_OFFSETS` only when
    # consistent frames place the plug further from it than normal grasp variation.
    GRASP_FRAMES = 5
    GRASP_MEASURE_MAX_SEC = 2.0
    GRASP_STILL_MPS = 0.005
    GRASP_FIT_RMS_M = 0.001
    GRASP_FRAME_SPREAD_M = 0.001
    GRASP_FRAME_SPREAD_DEG = 1.5
    GRASP_APPLY_SHIFT_M = 0.0025
    GRASP_APPLY_SHIFT_DEG = 3.0
    # Number of consecutive above-threshold readings before aborting (~0.5 s at 10 Hz).
    # Mirrors the scoring system: penalty only after 1 s above 20 N.
    FORCE_ABORT_SUSTAINED_CYCLES = 10  # Adjusted for 20Hz control rate (0.5s)
    REQUIRED_LOCK_COUNT = 10  # Increased for 20Hz (0.5s re-lock) to ensure stability
    # Experimental perception: REQUIRED_LOCK_COUNT accepted estimates within the
    # last LOCK_WINDOW_CYCLES, all within LOCK_SPREAD_M of their median. Accepted
    # SC poses flicker near decoder margins even from a stationary view; demanding
    # consecutive cycles delayed locks without improving their accuracy.
    LOCK_WINDOW_CYCLES = 15
    LOCK_SPREAD_M = 0.005
    # Per-cable TCP → plug_tip_link transform, expressed as a chain of fixed
    # (translation, rotation) hops applied right-to-left in code
    # (T_tcp_tip = h0 ⊗ h1 …). Rotation is either RPY (3 floats) or a direct
    # quaternion tuple (x, y, z, w):
    #   - hop 0: TCP → plug body  (aic_engine/config/sample_config.yaml
    #            cables.*.pose.gripper_offset + sibling roll/pitch/yaw)
    #   - hop 1: plug body → *_tip_link  (aic_assets/models/SFP Module/model.sdf
    #            and SC Plug/model.sdf — the trailing transform from the body
    #            link to the contact-tip frame, which CheatCode picks up via TF
    #            because task.plug_name = "sfp_tip" / "sc_tip")
    # Without hop 1 the plug-tip orientation is wrong by ~90° (SFP) or ~90°+90°
    # (SC), and the CheatCode q_diff bakes that error into the gripper target,
    # producing an unwanted spin about the vertical. ~2 mm / ~0.04 rad of
    # trial-to-trial grasp variation in hop 0 is documented and absorbed by the
    # impedance controller. FK-derivable — no GT TF at eval time.
    _PLUG_OFFSETS = {
        "sfp": [
            # Measured TCP-local SFP tip offset from repeated TF-debug trials:
            # rel_pos ~= (-0.1, -20.7, +54.1) mm. The previous two-hop SDF chain
            # missed runtime attach transforms and was off by ~45 mm in GT-off
            # runs. rel_rpy was reconstructed from live sfp_tip_link axes:
            # tf_X≈(+1.00,+0.05,+0.02), tf_Z≈(+0.00,+0.35,-0.94) at the
            # canonical start TCP pose.
            ((-0.0001, -0.0207, 0.0541), (0.3560, 0.0187, -0.0503)),
        ],
        "sc": [
            # Measured once from TF on the sample SC task during development
            # (TCP→cable_1/sc_tip_link). A fixed FK constant: nothing reads TF
            # for the plug at runtime.
            ((-0.0004, -0.0131, 0.0165), (0.1592, -0.1663, 0.6946, 0.6816)),
        ],
    }
    STATIC_TRUSTED_PLUG_ORIENTATION_TYPES = {"sfp", "sc"}
    CENTERING_TOL = 0.05
    PRE_INSERT_TIMEOUT_SEC = 4.0
    PRE_INSERT_ORIENTATION_EXTEND_SEC = 8.0
    CAPTURE_MIN_PERIOD_SEC = 0.25
    COARSE_ALIGN_POSE_TOL_M = 0.030
    COARSE_ALIGN_HOLD_CYCLES = 4
    COARSE_ALIGN_STALL_SEC = 4.0
    COARSE_ALIGN_STALL_IMPROVE_M = 0.010
    # Distance at which orientation SLERP toward port axis begins (m).
    # CheatCode-style time-based slerp: orientation ramps 0→1 over this many
    # seconds from coarse_align entry. Decoupled from pose error so the
    # rotation happens smoothly during the approach rather than as a late
    # twist near the goal.
    COARSE_ALIGN_ORIENT_RAMP_SEC = 3.0
    # EMA alpha for smoothing the heatmap-derived insertion axis across frames.
    ORIENT_AXIS_EMA = 0.25
    # Consistency-gate parameters for seeding port_insertion_axis_smoothed.
    AXIS_SEED_REQUIRE_N = 3
    AXIS_SEED_AGREE_COS = 0.9  # ≈ 26°
    # Threshold on the cos-angle between the port outward normal
    # (-port_rot[:,2], since port_link +Z is INTO the card per URDF) and
    # the (port→TCP) direction. This measures "is TCP near the port's
    # outward-normal axis" — it does NOT measure chirality correctness.
    # From a typical pre-alignment starting pose (TCP above a horizontally
    # mounted port), axis_align maxes around +0.3–+0.5. Mirror-flipped
    # Mirror-flipped pose estimates produce strongly negative values (-0.4 to -1.0).
    # 0.3 cleanly separates the two and lets the gate open during early
    # coarse_align so q_diff orientation can guide the rotation rather than
    # waiting for fallback alignment to get TCP "in front" of the port first.
    PORT_ROT_AXIS_ALIGN_MIN = 0.3
    PRE_INSERT_POSE_TOL_M = 0.005
    # On the face-search path a timed-out pre_insert within this pose error, and
    # within the spiral radius laterally, starts the insert instead of recovering.
    PRE_INSERT_HANDOVER_POSE_TOL_M = 0.010
    # The same handover, before the timeout, once the residual has stopped
    # improving by PROGRESS for PLATEAU_SEC (and at least MIN_SEC into the phase).
    PRE_INSERT_PLATEAU_MIN_SEC = 2.5
    PRE_INSERT_PLATEAU_SEC = 1.5
    PRE_INSERT_PLATEAU_PROGRESS_M = 0.0003
    # Fine-center servo: integrator gain on (perceived_port_xy − perceived_plug_xy).
    # Each cycle adds gain × residual to an accumulating world-XY correction that
    # is added to the standoff goal. Closes the loop on impedance steady-state
    # offset so the plug is centered to within PRE_INSERT_XY_RESIDUAL_TOL_M of
    # the port before transitioning to insert.
    PRE_INSERT_FINE_CENTER_ENABLED = False
    PRE_INSERT_FINE_CENTER_GAIN = 0.4
    PRE_INSERT_FINE_CENTER_CAP_M = 0.02
    PRE_INSERT_XY_RESIDUAL_TOL_M = 0.002
    # Reject port detections too close to the TCP (likely a TF lookup failure
    # returning origin-near values). 5 mm is below the achievable insertion
    # tolerance, so a real port detection should never trigger this. Was 5 cm,
    # which rejected every detection once the arm closed in and bounced the
    # phase machine back to find_target indefinitely.
    PLAUSIBLE_PORT_DISTANCE_MIN_M = 0.005
    PLAUSIBLE_PORT_DISTANCE_MAX_M = 0.40
    PERCEPTION_STALE_CYCLES = 5  # consecutive not-visible → fall back to find_target
    # Cycles with no evaluable camera (e.g. exposure TF not yet available) are
    # not misses until the gap lasts this long.
    PERCEPTION_GAP_MAX_SEC = 1.0
    # Per-phase deadline guards: if remaining time < budget, exit gracefully.
    PHASE_TIME_BUDGET = {
        "find_target": 15.0,
        "coarse_align": 8.0,
        "pre_insert": 5.0,
        "insert": 5.0,
        "recover": 5.0,
        "settle": 1.0,
    }
    # EMA alpha for port position smoothing: 0.25 → ~4-cycle time constant at 20 Hz
    PORT_POS_EMA_ALPHA = 0.15
    SFP_CAGE_HEATMAP_MIN_CAMERAS = 2
    SFP_CAGE_HEATMAP_MIN_CONFIDENCE = 0.50
    # Card-template SFP faces are rigid per camera; triangulated faces fit within this.
    SFP_FACE_MAX_GEOMETRY_RESIDUAL_M = 0.004
    # Cameras must agree on the target face implied by their decoded card placement.
    SFP_CROSS_VIEW_FACE_TOLERANCE_M = 0.002
    # With fewer than two agreeing views, take the pose from one camera's decoded card
    # placement in the registered board frame (inherits registration error).
    SFP_SINGLE_VIEW_BOARD_POSE = True
    # Minimum decode margin for a single view (one near-ambiguous decode was 13 mm off).
    SFP_SINGLE_VIEW_MIN_MARGIN = 0.15
    SC_PORT_HEATMAP_MIN_CAMERAS = 2
    SC_PORT_HEATMAP_MIN_CONFIDENCE = 0.45
    SC_PORT_HEATMAP_MAX_GEOMETRY_RESIDUAL_M = 0.020
    SC_PORT_SPATIAL_LOCK_SEARCH_M = 0.030
    SC_PORT_SPATIAL_LOCK_CLOSE_M = 0.004

    MODE_CONFIG = {
        "sfp": {
            "success_xy_tol_m": 0.004,
            # Once the plug tip is far beyond the entrance face, high lateral
            # force is expected from the seated connector/cable load. Treat it
            # as geometry-confirmed insertion instead of recovering on timeout.
            "success_deep_xy_tol_m": 0.003,
            "success_deep_plug_depth_m": 0.008,
            "face_slip_xy_recover_m": 0.0055,
            "face_slip_depth_guard_m": -0.0015,
            "face_recenter_xy_m": 0.009,
            "face_recenter_max_attempts": 3,
            "pre_insert_fine_center_gain": 0.8,
            "pre_insert_fine_center_cap_m": 0.004,
            "pre_insert_xy_tol_m": 0.0020,
            "pre_insert_xy_hold_cycles": 3,
            "pre_insert_axis_angle_tol_rad": math.radians(8.0),
            "pre_insert_orientation_tol_rad": math.radians(10.0),
            "pre_insert_plug_axis_angle_tol_rad": math.radians(6.0),
            "pre_insert_plug_orientation_tol_rad": math.radians(10.0),
            "pre_insert_timeout_sec": 8.0,
            "pre_insert_orientation_extend_sec": 8.0,
            "pre_insert_orientation_extend_pose_tol_m": 0.012,
            "pre_insert_require_locked_orientation": True,
            "engaged_seating_min_lateral_n": 5.0,
            "engaged_seating_min_travel_m": 0.0060,
            "engaged_force_drop_n": 999.0,
            "engaged_seating_forward": 0.004,
            "engaged_seating_forward_mid": 0.006,
            "engaged_seating_forward_high": 0.008,
            "engaged_seating_feedforward_n": 4.0,
            "engaged_seating_feedforward_mid_n": 7.0,
            "engaged_seating_feedforward_high_n": 10.0,
            "engaged_total_timeout_sec": 18.0,
            "centered_face_extend_sec": 5.0,
            # Jam detection during face_probe: if neither depth nor axial travel
            # advances for this long while contact force is present, the plug is
            # mechanically pinned on the port face — re-seat instead of grinding.
            "insert_jam_timeout_sec": 4.0,
            "insert_jam_force_n": 6.0,
            "settle_max_sec": 18.0,
            # Geometry-fallback success requires axial travel to plateau for
            # this long — declaring success while the plug is still creeping in
            # leaves it partially seated.
            "settle_travel_plateau_sec": 1.5,
            "settle_deep_confirm_cycles": 2,
            "settle_geometry_confirm_sec": 6.0,
            "seating_forward": 0.010,
            "seating_feedforward_n": 9.0,
            "seating_target_depth_m": 0.018,
            "insert_use_port_axis_twist": True,
            "face_probe_forward": 0.0030,
            "face_probe_feedforward_n": 6.0,
            "face_probe_lateral_gain": 0.18,
            "face_probe_lateral_cap_mps": 0.0008,
            "face_probe_near_face_depth_m": -0.0040,
            "face_probe_centered_xy_m": 0.0020,
            "face_probe_centered_depth_m": -0.0035,
            "face_probe_centered_extend_sec": 8.0,
            "face_probe_off_face_descending_extend_sec": 4.0,
            "face_probe_centered_forward": 0.0035,
            "face_probe_centered_feedforward_n": 5.0,
            "face_probe_centered_lateral_gain": 0.28,
            "face_probe_centered_lateral_cap_mps": 0.0007,
            "face_probe_recenter_xy_m": 0.0020,
            "face_probe_recenter_y_m": 0.0006,
            "face_probe_recenter_x_m": 0.0012,
            "face_probe_recenter_depth_m": -0.0030,
            "face_probe_recenter_max_xy_m": 0.0090,
            "face_probe_recenter_extend_sec": 4.0,
            "face_probe_timeout_recenter_depth_m": -0.0050,
            "face_probe_timeout_contact_force_n": 6.0,
            "face_probe_recenter_forward": 0.0003,
            "face_probe_recenter_feedforward_n": 0.0,
            "face_probe_recenter_lateral_gain": 0.45,
            "face_probe_recenter_lateral_cap_mps": 0.0020,
            "shallow_breakthrough_start_sec": 1.6,
            "shallow_breakthrough_progress_window_sec": 1.0,
            "shallow_breakthrough_forward": 0.0050,
            "shallow_breakthrough_feedforward_n": 7.0,
            "shallow_breakthrough_lateral_gain": 0.18,
            "shallow_breakthrough_lateral_cap_mps": 0.0010,
            "shallow_breakthrough_dither_mps": 0.0012,
            "shallow_breakthrough_dither_hz": 0.8,
            "shallow_breakthrough_depth_m": 0.0060,
            "engaged_xy_m": 0.0030,
            "engaged_depth_m": 0.0040,
            "deep_depth_m": 0.0080,
            "deep_travel_m": 0.010,
            "escaped_xy_m": 0.0060,
            "escaped_depth_m": 0.0005,
            "engaged_shallow_recenter_depth_m": -0.0030,
            "engaged_shallow_recenter_max_xy_m": 0.0060,
            "engaged_shallow_recenter_y_m": 0.0006,
            "engaged_shallow_recenter_x_m": 0.0012,
            "centered_face_xy_m": 0.0025,
            # Plug-tip standoff during coarse_align. With reliable port
            # rotation, this lands the plug along the entrance normal at
            # this distance from the face. With unreliable rotation, the
            # standoff falls back to the TCP→port line at this distance.
            # Kept modest because any residual direction error scales linearly
            # with offset.
            "approach_offset_m": 0.060,
            # Keep extra clearance during the image-space servo window. Logs
            # from the first heatmap online test showed the actual plug tip
            # only 2-4 mm above the SFP face with the old 5 mm standoff, so
            # lateral correction could scrape the NIC before insertion starts.
            "pre_insert_offset_m": 0.008,
        },
        "sc": {
            "force_recover_margin_n": 14.0,
            "force_abort_margin_n": 28.0,
            "success_xy_tol_m": 0.004,
            "success_deep_xy_tol_m": 0.003,
            "success_deep_plug_depth_m": 0.01564,
            "success_min_travel_m": 0.01564,
            "face_slip_xy_recover_m": 0.0045,
            "face_slip_depth_guard_m": -0.001,
            "face_recenter_xy_m": 0.009,
            "face_recenter_max_attempts": 5,
            "pre_insert_fine_center_gain": 0.7,
            "pre_insert_fine_center_cap_m": 0.003,
            "pre_insert_xy_tol_m": 0.0030,
            "pre_insert_xy_hold_cycles": 1,
            "pre_insert_pose_tol_m": 0.0060,
            "pre_insert_axis_angle_tol_rad": math.radians(8.0),
            "pre_insert_orientation_tol_rad": math.radians(15.0),
            "pre_insert_plug_axis_angle_tol_rad": math.radians(8.0),
            "pre_insert_plug_orientation_tol_rad": math.radians(15.0),
            "pre_insert_timeout_sec": 8.0,
            "pre_insert_orientation_extend_sec": 8.0,
            "pre_insert_orientation_extend_pose_tol_m": 0.012,
            "pre_insert_require_locked_orientation": True,
            # Race grasps tilt the wrist 12-19 deg, and the cable then holds the
            # compliant arm 5-6 mm off its pre-insert goal, beyond the 3.5 mm
            # spiral radius, so the trial looped in recovery (in two race trials,
            # even with the true grasp). The SC face probe guides by contact up
            # to face_contact_guided_max_xy_m, so hand over within that.
            "pre_insert_handover_residual_m": 0.0060,
            # Insert drops the fine-centering correction, so the arm swings
            # back up to ~1 mm when it lands; start the face search on that
            # face_slide instead of recentering (policy_insert.py).
            "face_search_slide_start_max_xy_m": 0.0075,
            "engaged_seating_min_lateral_n": 8.0,
            "engaged_seating_min_travel_m": 0.003,
            "engaged_force_drop_n": 4.0,
            "engaged_depth_m": 0.0030,
            # Contact engagement is not proof of full insertion. Completion uses
            # the asset entrance-to-origin distance and independent axial travel.
            "deep_depth_m": 0.0070,
            "deep_travel_m": 0.0095,
            "centered_face_extend_sec": 1.0,
            "settle_max_sec": 8.0,
            "settle_travel_plateau_sec": 1.0,
            "settle_deep_confirm_cycles": 2,
            "settle_geometry_confirm_sec": 1.0,
            "seating_forward": 0.012,
            "seating_feedforward_n": 25.0,
            "seating_pose_feedforward_n": 12.0,
            "seating_pose_feedforward_along_axis": True,
            # Target -11mm from port face for pose-seating (matches real SC travel).
            "seating_target_depth_m": 0.0110,
            "engaged_seating_forward": 0.008,
            "engaged_seating_feedforward_n": 15.0,
            "engaged_seating_forward_mid": 0.010,
            "engaged_seating_feedforward_mid_n": 20.0,
            "engaged_seating_forward_high": 0.012,
            "engaged_seating_feedforward_high_n": 25.0,
            # Reduced: once in deep_seating, if we don't succeed in 18s something is wrong.
            "engaged_total_timeout_sec": 18.0,
            # Enable pose-seating once well-engaged: drives TCP to a pose that places
            # the plug tip 11mm into the port, which is beyond the ~8mm plateau.
            # This uses seating_target_depth_m as the target offset.
            "deep_seating_pose_enabled": True,
            "deep_seating_pose_after_sec": 1.5,
            "deep_seating_pose_min_depth_m": 0.0055,
            "deep_seating_pose_min_travel_m": 0.0090,
            "deep_seating_pose_max_xy_m": 0.0025,
            "deep_seating_pose_max_force_n": 20.0,
            "settle_use_deep_seating_when_engaged": True,
            "insert_use_port_axis_twist": True,
            "shallow_breakthrough_forward": 0.0040,
            "shallow_breakthrough_feedforward_n": 10.0,
            "shallow_breakthrough_lateral_gain": 0.0,
            "shallow_breakthrough_lateral_cap_mps": 0.0,
            "shallow_breakthrough_dither_mps": 0.0,
            "shallow_breakthrough_dither_hz": 1.2,
            "shallow_breakthrough_depth_m": 0.0060,
            "shallow_breakthrough_min_depth_m": -0.0008,
            "shallow_breakthrough_force_drop_n": 8.0,
            "shallow_breakthrough_start_sec": 0.6,
            "shallow_breakthrough_progress_window_sec": 1.0,
            "insert_jam_force_n": 15.0,
            "insert_jam_timeout_sec": 3.0,
            "face_contact_guided_enabled": True,
            "face_contact_guided_states": (
                "off_face",
                "face_contact",
                "centered_face",
                "face_slide",
            ),
            "face_contact_guided_depth_m": -0.0045,
            "face_contact_guided_max_xy_m": 0.0060,
            "face_contact_guided_min_xy_m": 0.0006,
            "face_contact_guided_forward": 0.0020,
            "face_contact_guided_min_forward": 0.0004,
            "face_contact_guided_feedforward_n": 6.0,
            "face_contact_guided_min_feedforward_n": 0.0,
            "face_contact_guided_lateral_gain": 0.60,
            "face_contact_guided_lateral_cap_mps": 0.0025,
            "face_contact_guided_force_soft_n": 20.0,
            "face_contact_guided_force_hard_n": 24.0,
            "face_contact_guided_improve_m": 0.00025,
            "face_contact_guided_stall_sec": 1.4,
            "face_contact_guided_max_sec": 4.0,
            "corner_reference_enabled": True,
            "corner_reference_after_sec": 0.0,
            "corner_reference_find_sec": 0.8,
            "corner_reference_lateral_mps": 0.0040,
            "corner_reference_feedforward_n": 8.0,
            "corner_reference_force_n": 8.5,
            "corner_reference_require_force": True,
            "corner_reference_min_move_m": 0.0008,
            "corner_reference_center_offset_m": 0.0032,
            "corner_reference_center_use_residual": True,
            "corner_reference_project_to_face": False,
            "corner_reference_center_sec": 3.0,
            "corner_reference_center_lateral_mps": 0.0080,
            "corner_reference_center_min_lateral_mps": 0.0040,
            "corner_reference_center_residual_gain": 2.5,
            "corner_reference_center_feedforward_n": 6.0,
            "corner_reference_center_pose_enabled": True,
            "corner_reference_center_pose_step_m": 0.0025,
            # Tightened from 6mm: at >3mm the plug is too far off-center for CR
            # to reliably converge. Let face_contact_guided handle large offsets.
            "corner_reference_max_xy_m": 0.0035,
            "contact_seated_xy_m": 0.0025,
            "contact_seated_guided_enabled": True,
            "contact_seated_classifier_enabled": True,
            "contact_seated_depth_m": 0.0015,
            "contact_seated_shallow_depth_m": -0.0008,
            "contact_engaged_depth_m": -0.0002,
            "contact_seated_travel_m": 0.0030,
            "contact_seated_force_drop_n": 8.0,
            "escaped_depth_m": -0.0012,
            "face_probe_centered_xy_m": 0.0012,
            "face_probe_centered_depth_m": -0.0035,
            "face_probe_centered_forward": 0.0050,
            "face_probe_centered_feedforward_n": 6.0,
            "face_probe_centered_lateral_gain": 0.0,
            "face_probe_centered_lateral_cap_mps": 0.0,
            "face_probe_near_face_depth_m": -0.0040,
            "face_probe_recenter_depth_m": -0.0040,
            "face_probe_recenter_x_m": 0.0008,
            "face_probe_recenter_y_m": 0.0012,
            "face_probe_recenter_max_xy_m": 0.0060,
            "face_probe_timeout_recenter_depth_m": -0.0060,
            "face_probe_timeout_contact_force_n": 6.0,
            "approach_offset_m": 0.025,
            "pre_insert_offset_m": 0.005,
        },
    }

    _MODE_MAP = {"sfp": "sfp", "sc": "sc", "SFP": "sfp", "SC": "sc"}
    PARK_STANDOFF_M = 0.0015
    PARK_CLEARANCE_M = 0.010
    PARK_LATERAL_TOL_M = 0.001
    PARK_MAX_SEC = 4.0
